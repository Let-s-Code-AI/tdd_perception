from pathlib import Path

import cv2
import numpy as np
import pytest

gradio = pytest.importorskip("gradio")

from perception.app import _write_browser_video


def test_write_browser_video_creates_readable_mp4(tmp_path: Path) -> None:
    frames = [np.full((5, 7, 3), value, dtype=np.uint8) for value in (0, 255)]
    output = tmp_path / "result.mp4"

    _write_browser_video(output, frames, fps=5.0)

    capture = cv2.VideoCapture(str(output))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()

    assert ok
    assert frame.shape[:2] == (6, 8)
