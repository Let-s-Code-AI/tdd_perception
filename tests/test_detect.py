"""Testy Detector/pick_device na atrapach torch/transformers: bez wag, bez sieci.

Importy torch/transformers w perception.detect są leniwe (dzieją się wewnątrz
funkcji), więc podstawiamy atrapy modułów w sys.modules przed wywołaniem -
tak jak w produkcji, kod dostaje coś z interfejsem HF, ale bez realnych wag.
"""

import contextlib
import sys
import types

import numpy as np
import pytest

from perception.detect import MODEL_ID, Detector, pick_device

DEFAULT_ID2LABEL = {0: "person", 1: "car", 2: "dog", 3: "traffic light"}


def _fake_torch(mps: bool = False, cuda: bool = False) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)
        ),
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        no_grad=lambda: contextlib.nullcontext(),
        tensor=lambda x: x,
    )


class _FakeTensor:
    """Namiastka torch.Tensor: tylko .cpu().numpy(), nic więcej nie jest potrzebne."""

    def __init__(self, array):
        self._array = np.asarray(array, dtype=float)

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeEncoding(dict):
    def to(self, device: str) -> "_FakeEncoding":
        return self


class _FakeProcessor:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.detections: dict[str, _FakeTensor] | None = None
        self.last_threshold: float | None = None

    def __call__(self, images, return_tensors: str) -> _FakeEncoding:
        return _FakeEncoding(pixel_values=_FakeTensor(np.zeros((1, 3, 4, 4))))

    def post_process_object_detection(self, outputs, threshold, target_sizes):
        self.last_threshold = threshold
        return [self.detections]

    @classmethod
    def from_pretrained(cls, model_id: str) -> "_FakeProcessor":
        return cls(model_id)


class _FakeDetrModel:
    def __init__(self, model_id: str, id2label: dict[int, str]):
        self.model_id = model_id
        self.config = types.SimpleNamespace(id2label=id2label)
        self.device: str | None = None

    def to(self, device: str) -> "_FakeDetrModel":
        self.device = device
        return self

    def eval(self) -> "_FakeDetrModel":
        return self

    def __call__(self, **kwargs):
        return "raw_output"

    @classmethod
    def from_pretrained(
        cls, model_id: str, attn_implementation: str = "eager"
    ) -> "_FakeDetrModel":
        return cls(model_id, DEFAULT_ID2LABEL)


def _install_fakes(monkeypatch: pytest.MonkeyPatch, **torch_kwargs) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(**torch_kwargs))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            DetrImageProcessor=_FakeProcessor, DetrForObjectDetection=_FakeDetrModel
        ),
    )


def test_pick_device_prefers_mps(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True, cuda=True))
    assert pick_device() == "mps"


def test_pick_device_prefers_cuda_over_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=False, cuda=True))
    assert pick_device() == "cuda"


def test_pick_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=False, cuda=False))
    assert pick_device() == "cpu"


def test_detector_init_builds_id2label_and_road_ids(monkeypatch):
    _install_fakes(monkeypatch)

    det = Detector(device="cpu")

    assert det.device == "cpu"
    assert det.processor.model_id == MODEL_ID
    assert det.model.model_id == MODEL_ID
    assert det.model.device == "cpu"
    assert det.id2label == DEFAULT_ID2LABEL
    assert det.road_ids == {0, 1, 3}  # person, car, traffic light są w ROAD; dog nie


def test_detector_init_uses_custom_model_id(monkeypatch):
    _install_fakes(monkeypatch)

    det = Detector(device="cpu", model_id="custom/model")

    assert det.processor.model_id == "custom/model"
    assert det.model.model_id == "custom/model"


def test_detector_init_picks_device_when_not_given(monkeypatch):
    _install_fakes(monkeypatch, mps=False, cuda=False)

    det = Detector()

    assert det.device == "cpu"


def test_detector_call_filters_and_passes_threshold(monkeypatch):
    _install_fakes(monkeypatch)
    det = Detector(device="cpu")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    det.processor.detections = {
        "boxes": _FakeTensor(
            [
                [100, 100, 150, 150],  # person, zostaje
                [200, 200, 250, 250],  # dog, poza ROAD
                [0, 0, 600, 400],  # gigantyczny boks (maska auta)
                [101, 101, 151, 151],  # duplikat pierwszego, niższy score -> NMS
            ]
        ),
        "scores": _FakeTensor([0.9, 0.95, 0.99, 0.5]),
        "labels": _FakeTensor([0, 2, 1, 0]),
    }

    boxes, scores, labels = det(img, thr=0.4)

    assert det.processor.last_threshold == 0.4
    assert labels.tolist() == [0]
    assert scores.tolist() == pytest.approx([0.9])
    np.testing.assert_allclose(boxes, [[100, 100, 150, 150]])


def test_detector_call_uses_default_threshold(monkeypatch):
    _install_fakes(monkeypatch)
    det = Detector(device="cpu")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    det.processor.detections = {
        "boxes": _FakeTensor([[10, 10, 20, 20]]),
        "scores": _FakeTensor([0.5]),
        "labels": _FakeTensor([1]),
    }

    det(img)

    assert det.processor.last_threshold == 0.3
