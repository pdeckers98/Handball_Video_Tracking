"""Read the broadcast score overlay (team names, score, half, clock) from a frame.

The overlay (eyecons / SEW streams) sits at a fixed position with a fixed font, so reading it
is plain template matching: threshold the white (or purple) text, split it into glyphs, and
match each glyph against templates harvested from the clips
(``tools/build_scoreboard_templates.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
from PIL import Image

# Layout in pixels at 1920x1080 (y0, y1, x0, x1).
FIELDS = {
    "home_name": (85, 125, 760, 862),
    "home_score": (85, 125, 855, 940),
    "away_score": (85, 125, 985, 1075),
    "away_name": (85, 125, 1075, 1165),
    "half": (145, 188, 875, 940),
    "clock": (145, 188, 945, 1045),
}
# Panel colour probes: (y0, y1, x0, x1) and the expected mean RGB.
PANEL_PROBES = (
    ((73, 84, 760, 1160), (23, 20, 46)),  # navy main bar, above the text
    ((146, 151, 950, 1040), (148, 61, 238)),  # purple clock box
)
PANEL_TOLERANCE = 15
REF_SIZE = (1920, 1080)

GLYPH_SHAPE = (24, 32)  # digit glyphs, height-normalized, aspect ratio kept
NAME_SHAPE = (24, 96)  # whole team-name word
HALF_SHAPE = (16, 48)  # whole "1ST" / "2ND" word
MIN_GLYPH_PIXELS = 25
MIN_GLYPH_HEIGHT = 14
# Mean abs diff to the best template; above = unreadable. Real glyphs: p99 0.03, max 0.065.
MAX_GLYPH_DISTANCE = 0.08
# Best other-label template must be at least this much further (real glyphs: >= 0.027).
MIN_LABEL_MARGIN = 0.01
MAX_WORD_DISTANCE = 0.15


@dataclass(frozen=True)
class Reading:
    """One frame's overlay. Fields are ``None`` when absent or unreadable."""

    present: bool
    home_name: str | None = None
    away_name: str | None = None
    home: int | None = None
    away: int | None = None
    half: str | None = None
    clock: int | None = None
    """Game clock in seconds as shown (counts up)."""

    @property
    def score(self) -> tuple[int, int] | None:
        if self.home is None or self.away is None:
            return None
        return self.home, self.away


@dataclass(frozen=True)
class Templates:
    score: np.ndarray
    score_labels: tuple[str, ...]
    clock: np.ndarray
    clock_labels: tuple[str, ...]
    name: np.ndarray
    name_labels: tuple[str, ...]
    half: np.ndarray
    half_labels: tuple[str, ...]

    def save(self, path) -> None:
        arrays = {}
        for kind in ("score", "clock", "name", "half"):
            arrays[kind] = np.round(getattr(self, kind) * 255).astype(np.uint8)
            arrays[f"{kind}_labels"] = np.array(getattr(self, f"{kind}_labels"))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> Templates:
        with np.load(path) as data:
            kwargs = {}
            for kind in ("score", "clock", "name", "half"):
                kwargs[kind] = data[kind].astype(np.float32) / 255.0
                kwargs[f"{kind}_labels"] = tuple(str(x) for x in data[f"{kind}_labels"])
        return cls(**kwargs)


@lru_cache(maxsize=1)
def default_templates() -> Templates:
    with resources.as_file(
        resources.files("handball") / "data" / "scoreboard_templates.npz"
    ) as path:
        return Templates.load(path)


# --------------------------------------------------------------------------- image helpers


def to_reference_size(image: np.ndarray) -> np.ndarray:
    """Resize a full frame to 1920x1080 if needed (layout is defined at that size)."""
    if (image.shape[1], image.shape[0]) == REF_SIZE:
        return image
    return np.asarray(Image.fromarray(image).resize(REF_SIZE, Image.BILINEAR))


def panel_present(image: np.ndarray) -> bool:
    """True when the overlay's navy and purple panels are visible at the expected place."""
    for (y0, y1, x0, x1), colour in PANEL_PROBES:
        mean = image[y0:y1, x0:x1].reshape(-1, 3).mean(0)
        if np.any(np.abs(mean - np.array(colour)) > PANEL_TOLERANCE):
            return False
    return True


def text_mask(image: np.ndarray, field: str) -> np.ndarray:
    """Binary text mask of a field. White text everywhere except the purple half label."""
    y0, y1, x0, x1 = FIELDS[field]
    hsv = np.asarray(Image.fromarray(image[y0:y1, x0:x1]).convert("HSV")).astype(np.int16)
    if field == "half":
        return hsv[..., 2] > 120
    return (hsv[..., 2] > 170) & (hsv[..., 1] < 90)


def segment(mask: np.ndarray) -> list[np.ndarray]:
    """Split a text mask into glyphs at empty columns, left to right."""
    cols = mask.any(0)
    glyphs, start = [], None
    for i, filled in enumerate([*cols, False]):
        if filled and start is None:
            start = i
        elif not filled and start is not None:
            sub = mask[:, start:i]
            rows = np.nonzero(sub.any(1))[0]
            if sub.sum() >= MIN_GLYPH_PIXELS and rows[-1] - rows[0] + 1 >= MIN_GLYPH_HEIGHT:
                glyphs.append(sub)
            start = None
    return glyphs


def _resize(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.resize((width, height), Image.BILINEAR), dtype=np.float32) / 255.0


def _bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def normalize_glyph(mask: np.ndarray) -> np.ndarray:
    """Scale a glyph to a fixed height, keep its aspect ratio, centre it on a fixed canvas."""
    height, width = GLYPH_SHAPE
    glyph = _bbox(mask)
    scale = (height - 2) / glyph.shape[0]
    new_w = max(1, min(width, int(round(glyph.shape[1] * scale))))
    canvas = np.zeros(GLYPH_SHAPE, np.float32)
    x0 = (width - new_w) // 2
    canvas[1:height - 1, x0:x0 + new_w] = _resize(glyph, new_w, height - 2)
    return canvas


def normalize_word(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Stretch the text bounding box of a whole word to ``shape``."""
    word = _bbox(mask)
    if word is None:
        return np.zeros(shape, np.float32)
    return _resize(word, shape[1], shape[0])


# --------------------------------------------------------------------------- matching


def _nearest(sample: np.ndarray, templates: np.ndarray, labels, max_dist: float) -> str | None:
    """Label of the closest template, or ``None`` if too far or ambiguous between labels."""
    dist = np.abs(templates - sample).reshape(len(templates), -1).mean(1)
    best = int(dist.argmin())
    if dist[best] > max_dist:
        return None
    others = [d for d, label in zip(dist, labels) if label != labels[best]]
    if others and min(others) - dist[best] < MIN_LABEL_MARGIN:
        return None
    return labels[best]


def _read_digits(image: np.ndarray, field: str, templates: np.ndarray, labels) -> str | None:
    glyphs = segment(text_mask(image, field))
    if not glyphs:
        return None
    chars = []
    for glyph in glyphs:
        char = _nearest(normalize_glyph(glyph), templates, labels, MAX_GLYPH_DISTANCE)
        if char is None:
            return None
        chars.append(char)
    return "".join(chars)


def read_frame(image: np.ndarray, templates: Templates | None = None) -> Reading:
    """Read the overlay from a full RGB frame (any resolution, 16:9)."""
    templates = templates or default_templates()
    image = to_reference_size(image)
    if not panel_present(image):
        return Reading(present=False)

    def word(field: str, shape, tmpl, labels) -> str | None:
        sample = normalize_word(text_mask(image, field), shape)
        return _nearest(sample, tmpl, labels, MAX_WORD_DISTANCE)

    def score(field: str) -> int | None:
        text = _read_digits(image, field, templates.score, templates.score_labels)
        return int(text) if text is not None and 1 <= len(text) <= 2 else None

    clock = None
    text = _read_digits(image, "clock", templates.clock, templates.clock_labels)
    if text is not None and len(text) == 4 and int(text[2:]) < 60:
        clock = int(text[:2]) * 60 + int(text[2:])

    return Reading(
        present=True,
        home_name=word("home_name", NAME_SHAPE, templates.name, templates.name_labels),
        away_name=word("away_name", NAME_SHAPE, templates.name, templates.name_labels),
        home=score("home_score"),
        away=score("away_score"),
        half=word("half", HALF_SHAPE, templates.half, templates.half_labels),
        clock=clock,
    )
