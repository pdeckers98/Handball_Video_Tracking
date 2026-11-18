"""GPU-free tests for handball.vlm (models themselves are exercised by tools/vlm_bakeoff.py)."""

import numpy as np
import pytest
from PIL import Image

from handball.vlm import MODELS, mosaic, parse_point


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"x": 500, "y": 250}', (0.5, 0.25)),
        ('Sure! ```json\n{"x": 120, "y": 900}\n```', (0.12, 0.9)),
        ('[{"point_2d": [300, 600], "label": "ball"}]', (0.3, 0.6)),
        ("[100, 200, 300, 400]", (0.2, 0.3)),  # box -> centre
        ("0.25, 0.75", (0.25, 0.75)),  # already normalized
    ],
)
def test_parse_point(text, expected):
    assert parse_point(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "I cannot see a ball.", "x=5000, y=10", "42"])
def test_parse_point_rejects(text):
    assert parse_point(text) is None


def test_mosaic_keeps_time_order():
    frames = [np.full((10, 20, 3), i * 40, np.uint8) for i in range(6)]
    sheet = np.asarray(mosaic(frames, cols=4))
    assert sheet.shape == (20, 80, 3)
    assert sheet[0, 0, 0] == 0 and sheet[0, 60, 0] == 120 and sheet[10, 20, 0] == 200
    assert isinstance(mosaic([Image.new("RGB", (4, 4))]), Image.Image)


def test_model_registry():
    for name, (repo, loader, quantize) in MODELS.items():
        assert "/" in repo
        assert loader in {"hf", "moondream"}
        assert isinstance(quantize, bool)
