"""Testy charakteryzujące scripts/eval_base.py: bezpieczna sieć przed refaktorem."""

from pathlib import Path

import numpy as np
import pytest

from scripts.eval_base import light_color, map_metrics, prf, read_gt


def test_read_gt_box_format(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("3 0.5 0.5 0.2 0.4\n")
    boxes, cls = read_gt(label, w=100, h=200)
    assert cls.tolist() == [3]
    assert boxes[0] == pytest.approx([40.0, 60.0, 60.0, 140.0])


def test_read_gt_polygon_format_uses_bounding_box(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
    boxes, cls = read_gt(label, w=10, h=10)
    assert cls.tolist() == [1]
    assert boxes[0] == pytest.approx([1.0, 1.0, 9.0, 9.0])


def test_read_gt_skips_blank_and_malformed_lines(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("\n2 0.1\n3 0.5 0.5 0.2 0.4\n")
    boxes, cls = read_gt(label, w=10, h=10)
    assert cls.tolist() == [3]
    assert len(boxes) == 1


def test_read_gt_missing_file_is_empty():
    boxes, cls = read_gt(Path("does/not/exist.txt"), w=10, h=10)
    assert boxes.shape == (0, 4)
    assert cls.shape == (0,)


def test_light_color_red_patch():
    img = np.full((4, 4, 3), (255, 0, 0), dtype=np.uint8)
    assert light_color(img, np.array([0, 0, 4, 4])) == "Red-Traffic-Light"


def test_light_color_green_patch():
    img = np.full((4, 4, 3), (0, 255, 0), dtype=np.uint8)
    assert light_color(img, np.array([0, 0, 4, 4])) == "Green-Traffic-Light"


def test_light_color_empty_crop_defaults_to_red():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert light_color(img, np.array([10, 10, 10, 10])) == "Red-Traffic-Light"


def test_prf_perfect_match_scores_one():
    preds = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9]), np.array([0]))]
    gts = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([0]))]
    out = prf(preds, gts, classes=[0], thr=0.5)
    assert out[0] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "tp": 1,
        "fp": 0,
        "fn": 0,
    }


def test_prf_below_threshold_prediction_counts_as_missed_gt():
    preds = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([0.1]), np.array([0]))]
    gts = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([0]))]
    out = prf(preds, gts, classes=[0], thr=0.5)
    assert out[0]["tp"] == 0
    assert out[0]["fp"] == 0
    assert out[0]["fn"] == 1


def test_prf_prediction_without_matching_gt_is_false_positive():
    preds = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9]), np.array([0]))]
    gts = [(np.zeros((0, 4)), np.zeros(0, dtype=int))]
    out = prf(preds, gts, classes=[0], thr=0.5)
    assert out[0] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 1,
        "fn": 0,
    }


def test_prf_low_iou_counts_as_both_fp_and_fn():
    preds = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9]), np.array([0]))]
    gts = [(np.array([[50, 50, 60, 60]], dtype=float), np.array([0]))]
    out = prf(preds, gts, classes=[0], thr=0.5)
    assert out[0]["tp"] == 0
    assert out[0]["fp"] == 1
    assert out[0]["fn"] == 1


def test_prf_greedy_matching_prefers_higher_score():
    preds = [
        (
            np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float),
            np.array([0.4, 0.95]),
            np.array([0, 0]),
        )
    ]
    gts = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([0]))]
    out = prf(preds, gts, classes=[0], thr=0.0, iou_thr=0.5)
    assert out[0]["tp"] == 1
    assert out[0]["fp"] == 1
    assert out[0]["fn"] == 0


def test_prf_evaluates_classes_independently():
    preds = [
        (
            np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float),
            np.array([0.9, 0.9]),
            np.array([0, 1]),
        )
    ]
    gts = [
        (np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float), np.array([0, 1]))
    ]
    out = prf(preds, gts, classes=[0, 1], thr=0.5)
    assert out[0]["tp"] == 1 and out[0]["fp"] == 0
    assert out[1]["tp"] == 1 and out[1]["fp"] == 0


def test_map_metrics_perfect_match():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    preds = [(boxes, np.array([0.9, 0.8]), np.array([0, 1]))]
    gts = [(boxes, np.array([0, 1]))]
    out = map_metrics(preds, gts, n_classes=2)
    assert out["map"] == pytest.approx(1.0)
    assert out["map50"] == pytest.approx(1.0)
    assert out["per_class_ap"] == {0: pytest.approx(1.0), 1: pytest.approx(1.0)}
