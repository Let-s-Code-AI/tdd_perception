"""Testy cli.py na atrapach detektora/głębi: bez wag, bez sieci, szybkie.

Służą też jako sieć bezpieczeństwa dla testów mutacyjnych (`mutmut`, patrz
pyproject.toml [tool.mutmut]).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from perception.cli import _color, depth_frame, draw, draw_tracks, main, write_video
from perception.pipeline import Record

FIXTURE = Path(__file__).parent / "fixtures" / "clip.mp4"


# --- _color -----------------------------------------------------------------


def test_color_is_deterministic_per_track_id():
    assert _color(1) == _color(1)


def test_color_differs_for_different_ids():
    assert _color(1) != _color(2)


def test_color_known_value_for_track_1():
    # golden value z np.random.default_rng(1).integers(60, 255, 3)
    assert _color(1) == (152, 159, 207)


def test_color_values_within_configured_range():
    for track_id in range(1, 6):
        assert all(60 <= c < 255 for c in _color(track_id))


# --- draw ---------------------------------------------------------------------


def test_draw_returns_copy_without_mutating_input():
    frame = np.zeros((60, 80, 3), np.uint8)
    original = frame.copy()
    out = draw(frame, [], {})
    assert out is not frame
    np.testing.assert_array_equal(frame, original)


def test_draw_paints_box_border_in_track_color_and_leaves_interior():
    frame = np.zeros((60, 80, 3), np.uint8)
    rec = Record(0, 1, 3, np.array([10.0, 10.0, 50.0, 50.0]), None)
    out = draw(frame, [rec], {3: "car"})
    expected = _color(1)
    assert tuple(int(c) for c in out[10, 30]) == expected
    assert tuple(int(c) for c in out[30, 10]) == expected
    assert tuple(int(c) for c in out[30, 30]) == (0, 0, 0)


def test_draw_adds_extra_pixels_when_near_score_present():
    frame = np.zeros((60, 80, 3), np.uint8)
    without_near = Record(0, 1, 3, np.array([10.0, 20.0, 50.0, 50.0]), None)
    with_near = Record(0, 1, 3, np.array([10.0, 20.0, 50.0, 50.0]), 0.42)
    out_without = draw(frame, [without_near], {3: "car"})
    out_with = draw(frame, [with_near], {3: "car"})
    assert not np.array_equal(out_without, out_with)


# --- draw_tracks ---------------------------------------------------------------


def _bottom_center(box: np.ndarray) -> tuple[int, int]:
    return int((box[0] + box[2]) / 2), int(box[3])


def test_draw_tracks_single_frame_matches_plain_draw():
    frame = np.zeros((60, 100, 3), np.uint8)
    box0 = np.array([10.0, 10.0, 30.0, 30.0])
    rec0 = Record(0, 1, 3, box0, None)
    by_frame = {0: [rec0]}
    out = draw_tracks(frame, by_frame, 0, {3: "car"})
    expected = draw(frame, by_frame[0], {3: "car"})
    np.testing.assert_array_equal(out, expected)


def test_draw_tracks_connects_centers_across_frames_up_to_index():
    frame = np.zeros((60, 100, 3), np.uint8)
    box0 = np.array([10.0, 10.0, 30.0, 30.0])  # środek dolny (20, 30)
    box1 = np.array([40.0, 10.0, 60.0, 30.0])  # środek dolny (50, 30)
    box2 = np.array([70.0, 10.0, 90.0, 30.0])  # środek dolny (80, 30)
    by_frame = {
        0: [Record(0, 1, 3, box0, None)],
        1: [Record(1, 1, 3, box1, None)],
        2: [Record(2, 1, 3, box2, None)],
    }

    out = draw_tracks(frame, by_frame, 1, {3: "car"})

    col = _color(1)
    # na odcinku (20,30)-(50,30) powinien być narysowany ślad
    assert tuple(int(c) for c in out[30, 35]) == col
    # klatka 2 jest *po* indeksie 1, więc nie powinna wydłużać śladu
    assert tuple(int(c) for c in out[30, 65]) == (0, 0, 0)


def test_draw_tracks_single_point_draws_no_polyline():
    frame = np.zeros((60, 100, 3), np.uint8)
    box0 = np.array([10.0, 10.0, 30.0, 30.0])
    by_frame = {0: [Record(0, 1, 3, box0, None)], 5: [Record(5, 1, 3, box0, None)]}
    out = draw_tracks(frame, by_frame, 0, {3: "car"})
    expected = draw(frame, by_frame[0], {3: "car"})
    np.testing.assert_array_equal(out, expected)


# --- depth_frame -----------------------------------------------------------


def test_depth_frame_shape_and_dtype():
    depth = np.zeros((10, 10), np.uint8)
    out = depth_frame(depth)
    assert out.shape == (10, 10, 3)
    assert out.dtype == np.uint8


def test_depth_frame_converts_colormap_bgr_to_rgb():
    depth = np.array([[0, 255]], dtype=np.uint8)
    out = depth_frame(depth)
    assert tuple(int(c) for c in out[0, 0]) == (0, 0, 4)
    assert tuple(int(c) for c in out[0, 1]) == (252, 255, 164)


# --- write_video -------------------------------------------------------------


def test_write_video_rejects_non_mp4_extension(tmp_path: Path):
    with pytest.raises(ValueError):
        write_video(tmp_path / "out.avi", [np.zeros((4, 4, 3), np.uint8)], fps=10.0)


def test_write_video_adds_mp4_suffix_when_missing(tmp_path: Path):
    out = write_video(tmp_path / "out", [np.zeros((4, 4, 3), np.uint8)], fps=10.0)
    assert out == tmp_path / "out.mp4"
    assert out.exists()


def test_write_video_raises_runtime_error_when_writer_cannot_open(tmp_path: Path):
    bad_path = tmp_path / "brak_takiego_katalogu" / "out.mp4"
    with pytest.raises(RuntimeError):
        write_video(bad_path, [np.zeros((4, 4, 3), np.uint8)], fps=10.0)


def test_write_video_writes_all_frames_readable(tmp_path: Path):
    frames = [np.full((20, 20, 3), v, np.uint8) for v in (10, 200, 50)]
    out = write_video(tmp_path / "out.mp4", frames, fps=5.0)

    cap = cv2.VideoCapture(str(out))
    try:
        count = 0
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    assert count == len(frames)


# --- main --------------------------------------------------------------------


class FakeDetector:
    """Jeden stały boks klasy 'car', niezależnie od treści obrazu."""

    def __init__(self, device: str | None = None, model_id: str | None = None):
        self.device = device or "cpu"
        self.id2label = {3: "car"}

    def __call__(self, img, thr: float = 0.3):
        return (
            np.array([[10.0, 10.0, 60.0, 60.0]]),
            np.array([0.9]),
            np.array([3]),
        )


class FakeDepthEstimator:
    def __init__(self, device: str | None = None, model_id: str | None = None):
        self.device = device or "cpu"

    def __call__(self, img):
        return np.full(img.shape[:2], 0.5)


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("perception.detect.Detector", FakeDetector)
    monkeypatch.setattr("perception.depth.DepthEstimator", FakeDepthEstimator)


def test_main_missing_video_file_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    code = main([str(tmp_path / "nie_ma.mp4")])
    assert code == 2
    assert "Brak pliku" in capsys.readouterr().err


def test_main_returns_2_when_no_frames_decoded(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"to nie jest wideo")
    code = main([str(bad)])
    assert code == 2
    assert "Nie udało się wczytać klatek" in capsys.readouterr().err


def test_main_rejects_out_depth_without_depth(tmp_path: Path):
    video = tmp_path / "x.mp4"
    video.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                str(video),
                "--no-depth",
                "--out-depth",
                str(tmp_path / "d.mp4"),
            ]
        )
    assert exc.value.code == 2


def test_main_runs_pipeline_and_writes_all_outputs(
    fake_models: None, tmp_path: Path, capsys: pytest.CaptureFixture
):
    out_video = tmp_path / "out.mp4"
    out_tracks = tmp_path / "tracks.mp4"
    out_depth = tmp_path / "depth.mp4"
    out_json = tmp_path / "out.json"

    code = main(
        [
            str(FIXTURE),
            "--max-frames",
            "5",
            "--stride",
            "5",
            "--width",
            "160",
            "--out",
            str(out_video),
            "--out-tracks",
            str(out_tracks),
            "--out-depth",
            str(out_depth),
            "--json",
            str(out_json),
        ]
    )

    assert code == 0
    assert out_video.exists()
    assert out_tracks.exists()
    assert out_depth.exists()

    rows = json.loads(out_json.read_text())
    assert rows  # stały boks przez >= min_hits klatek potwierdza tor
    assert {row["id"] for row in rows} == {1}
    assert all(row["label"] == "car" for row in rows)
    assert all(row["near"] == pytest.approx(0.5, abs=1e-3) for row in rows)

    assert "Unikalne tory: 1" in capsys.readouterr().out


def test_main_no_depth_sets_near_to_null_in_json(fake_models: None, tmp_path: Path):
    out_json = tmp_path / "out.json"
    code = main(
        [
            str(FIXTURE),
            "--max-frames",
            "5",
            "--stride",
            "5",
            "--width",
            "160",
            "--no-depth",
            "--json",
            str(out_json),
        ]
    )
    assert code == 0
    rows = json.loads(out_json.read_text())
    assert rows and all(row["near"] is None for row in rows)


def test_main_returns_2_when_out_path_has_bad_extension(
    fake_models: None, tmp_path: Path, capsys: pytest.CaptureFixture
):
    code = main(
        [
            str(FIXTURE),
            "--max-frames",
            "3",
            "--stride",
            "5",
            "--width",
            "160",
            "--out",
            str(tmp_path / "out.avi"),
        ]
    )
    assert code == 2
    assert "Nie udało się zapisać wideo" in capsys.readouterr().err


def test_main_returns_2_when_tracks_dir_missing(
    fake_models: None, tmp_path: Path, capsys: pytest.CaptureFixture
):
    code = main(
        [
            str(FIXTURE),
            "--max-frames",
            "3",
            "--stride",
            "5",
            "--width",
            "160",
            "--out-tracks",
            str(tmp_path / "brak_katalogu" / "tracks.mp4"),
        ]
    )
    assert code == 2
    assert "Nie udało się zapisać wideo" in capsys.readouterr().err
