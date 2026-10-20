# Copilot instructions for `perception`

## Commands

Use `uv` for all Python commands (the lockfile is `uv.lock`).

```bash
uv sync --all-extras                         # all optional dependencies and dev tools
uv sync --extra ml --extra ui                # pipeline and Gradio UI dependencies

uv run pytest                                # full test suite
uv run pytest tests/<module>.py::test_name   # single test
uv run pytest -m model                       # model/weights-dependent tests
uv run pytest -m "not model"                 # tests without downloaded Hugging Face weights

uv run ruff check .
uv run ruff check --fix .
uv run mypy perception

uv run perception VIDEO.mp4 --out out.mp4 --json out.json
uv run perception-ui
```

Pytest is configured to collect from `tests/`; the `model` marker denotes slow tests that need Hugging Face weights. The current suite supplies shared video fixtures in `tests/conftest.py`, but no test modules yet. Coverage measures `perception` with branches and excludes `perception/app.py`.

## Architecture and data contracts

The dashcam pipeline is `DETR detection -> ByteTrack-lite tracking -> optional Depth Anything V2 relative depth`.

- `video.load_frames()` decodes video to resized **RGB `uint8`** frames, decimates by `stride`, and returns FPS adjusted by that stride. OpenCV writes and drawing paths must convert RGB back to BGR when needed.
- `pipeline.run()` is the model-independent orchestration boundary. It accepts injectable `detect(frame) -> (boxes, scores, labels)` and optional `depth(frame) -> depth_map` callables, so pipeline tests should use fakes rather than load models. It returns `RunResult`, containing flat per-track `Record`s, per-frame detection/timing data, and `uint8` depth maps only when depth is enabled.
- Boxes use NumPy `xyxy` coordinates throughout. `boxes.py` contains pure-NumPy IoU/NMS primitives. `detect.filter_detections()` is likewise model-free: it keeps road classes, removes overly large/low boxes, then applies class-agnostic NMS.
- `Detector` and `DepthEstimator` are Hugging Face wrappers with lazy imports. Device selection is centralized in `pick_device()` and prefers `mps`, then `cuda`, then `cpu`.
- `ByteTrackLite` uses Kalman-predicted tracks, Hungarian matching gated by class, and two confidence stages. Only tracks updated in the current frame and with `hits >= min_hits` are emitted; stale tracks are removed after `since > max_age`.
- Depth values are normalized independently for each frame (`0..1`, higher is nearer), so they are relative and must not be compared across frames or videos. `near_score()` uses the lower half of a clipped tracked box.

The CLI runs the full pipeline and can emit annotated, trail, depth, and JSON outputs. It serializes label IDs as model label names and rounds boxes and `near` values. The Gradio UI reuses process-cached models, renders through the CLI drawing helper, and prefers ffmpeg H.264 output, falling back to OpenCV `mp4v`.

## Repository conventions

- Keep identifiers in English, while source comments and user-facing CLI/UI text are in Polish.
- Preserve the dependency-injection split between pure NumPy/tracking/pipeline code and model-loading wrappers. Add behavior that can be tested without weights to the pure path where practical.
- Retain the `RunResult`/`Record` output shape when extending pipeline stages: records use frame index, track ID, numeric label, `xyxy` box, and optional relative `near`; downstream CLI, UI, and `gate.py` consume it.
- Test pipeline orchestration with injected detector/depth callables. `tests/conftest.py` supplies an RGB, resized, stride-decimated sample-video fixture for integration-oriented tests.
- Use `gate.compute_metrics()`, `check()`, and `markdown_report()` for regression-style pipeline quality thresholds. `check()` returns human-readable violations; an empty list is a passing gate.
- On Linux, `uv` resolves Torch and Torchvision from the explicit CPU-only `pytorch-cpu` index configured in `pyproject.toml`.
