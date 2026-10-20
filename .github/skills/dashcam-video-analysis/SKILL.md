---
name: dashcam-video-analysis
description: Analyze a dashcam video with this project's perception pipeline, export annotated video or tracking trails and JSON detections, and explain results. Use when asked to inspect, process, or visualize a driving video with the repository's models.
---

# Dashcam video analysis

Use the existing `perception` CLI to process user-provided dashcam footage. Do not implement a second inference path or claim that model output is ground truth.

## Workflow

1. Confirm the input video exists and understand which output(s) the user wants. Preserve existing files; choose new output paths rather than overwriting without permission.
2. Use a small frame limit for a quick inspection when the user has not requested a full run. Tell the user when the result is sampled: `--stride N` processes every Nth frame, and `--max-frames N` limits processed frames.
3. Create the requested output directory if needed, then run the CLI from the repository root. For example:

   ```bash
   uv run --extra ml perception path/to/video.mp4 \
     --out output/annotated.mp4 \
     --out-tracks output/tracks.mp4 \
     --json output/detections.json \
     --max-frames 180 --stride 4 --width 960
   ```

   Include only the outputs requested. Use `--device cpu`, `--device mps`, or `--device cuda` when the user requests a device; otherwise allow automatic selection.
4. To export a depth visualization, add `--out-depth output/depth.mp4`. Do not combine `--out-depth` with `--no-depth`. For a quicker detection/tracking-only run, use `--no-depth` and omit the depth output.
5. Report the output paths and run settings. Explain that object classes, tracking IDs, and relative depth (`near`) are model estimates; `near` is not a calibrated distance measurement. Mention frame sampling if used.

## Outputs and troubleshooting

- `--out` writes detection boxes and labels; `--out-tracks` also draws the observed object trails.
- `--json` writes one record per observed object with `frame`, `id`, `label`, `box`, and `near`. `near` is `null` when depth estimation is disabled.
- The CLI requires the ML extra (`uv run --extra ml ...`) to load the detection model. If model dependencies or weights are unavailable, report the error rather than presenting a partial run as successful.
- Output video paths must end in `.mp4` (or have no extension, in which case `.mp4` is added). Ensure the destination directory exists before running.
- If video loading fails, check the input path and format. If writing fails, check the destination path and that the video output is `.mp4`.
