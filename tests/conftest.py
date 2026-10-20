"""Współdzielone fixture'y dla testów pipeline'u percepcji."""

from pathlib import Path

import numpy as np
import pytest

from perception.video import load_frames


@pytest.fixture(scope="session")
def sample_video_path() -> Path:
    """Zwraca ścieżkę do przykładowego nagrania z dashcama."""
    return Path(__file__).parent / "fixtures" / "clip.mp4"


@pytest.fixture(scope="session")
def sample_video_frames(sample_video_path: Path) -> tuple[list[np.ndarray], float]:
    """Zwraca niewielką, deterministyczną próbkę klatek RGB i efektywny FPS."""
    return load_frames(sample_video_path, max_frames=8, stride=10, width=320)
