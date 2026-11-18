"""Rebuild ``src/handball/data/scoreboard_templates.npz`` from the clips.

1. Harvest glyphs (digits) and whole words (team names, half) from every keyframe of every
   clip where the overlay is visible.
2. Cluster them greedily (deterministic: clip order, then time).
3. Label each cluster from ``LABELS`` (cluster index, sorted by size). The labels were assigned
   by inspecting the contact sheets this script writes with ``--sheets``; if the clips change,
   re-inspect the sheets and update ``LABELS``.

Usage: python tools/build_scoreboard_templates.py [--clips clip1 clip2 ...] [--sheets DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from handball.config import load_config
from handball.scoreboard import (
    HALF_SHAPE,
    NAME_SHAPE,
    Templates,
    normalize_glyph,
    normalize_word,
    panel_present,
    segment,
    text_mask,
    to_reference_size,
)
from handball.video import iter_keyframes

OUT = Path(__file__).resolve().parents[1] / "src" / "handball" / "data" / "scoreboard_templates.npz"
THRESHOLDS = {"score": 0.045, "clock": 0.045, "name": 0.06, "half": 0.06}
MIN_CLUSTER = 5
LABELS = {
    "score": "2 1 4 6 8 9 7 3 0 1 5".split(),
    "clock": "1 2 3 0 4 0 5 2 1 6 9 8 0 7 4 6 8 9 7 5 5 3 2 8 6".split(),
    "name": "SEW PSV VOL VZV E&O E&O SEW E&O PSV E&O".split(),
    "half": "1ST 1ST 2ND 2ND 2ND 2ND 1ST".split(),
}


def harvest(clips: list[Path]) -> dict[str, np.ndarray]:
    groups: dict[str, list[np.ndarray]] = {"score": [], "clock": [], "name": [], "half": []}
    for clip in clips:
        for frame in iter_keyframes(clip):
            image = to_reference_size(frame.image)
            if not panel_present(image):
                continue
            for field in ("home_name", "away_name"):
                groups["name"].append(normalize_word(text_mask(image, field), NAME_SHAPE))
            groups["half"].append(normalize_word(text_mask(image, "half"), HALF_SHAPE))
            digit_fields = (("home_score", "score"), ("away_score", "score"), ("clock", "clock"))
            for field, kind in digit_fields:
                groups[kind].extend(normalize_glyph(g) for g in segment(text_mask(image, field)))
        print(f"{clip.name}: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    return {k: np.array(v) for k, v in groups.items()}


def cluster(samples: np.ndarray, threshold: float) -> list[list[int]]:
    """Greedy leader clustering; returns member lists sorted by size (largest first)."""
    leaders: list[np.ndarray] = []
    members: list[list[int]] = []
    for i, sample in enumerate(samples.reshape(len(samples), -1)):
        if leaders:
            dist = np.abs(np.array(leaders) - sample).mean(1)
            best = int(dist.argmin())
            if dist[best] < threshold:
                members[best].append(i)
                continue
        leaders.append(sample)
        members.append([i])
    return sorted(members, key=len, reverse=True)


def write_sheet(means: list[np.ndarray], labels: list[str], path: Path) -> None:
    tiles = []
    for i, (mean, label) in enumerate(zip(means, labels)):
        h, w = mean.shape
        scale = 2 if w < 64 else 1
        img = Image.fromarray((mean * 255).astype(np.uint8)).resize((w * scale * 2, h * scale * 2))
        tile = Image.new("RGB", (img.width, img.height + 12), "white")
        tile.paste(img.convert("RGB"), (0, 0))
        ImageDraw.Draw(tile).text((2, img.height), f"{i}: {label}", fill="red")
        tiles.append(np.asarray(tile))
    cols = 12 if tiles[0].shape[1] <= 128 else 6
    tiles += [np.full(tiles[0].shape, 255, np.uint8)] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    Image.fromarray(np.vstack(rows)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", nargs="*", default=[f"clip{i}" for i in range(1, 9)])
    parser.add_argument("--sheets", type=Path, help="write labeled contact sheets here")
    args = parser.parse_args()

    config = load_config()
    groups = harvest([config.resolve_clip(c) for c in args.clips])
    result = {}
    for kind, samples in groups.items():
        clusters = [m for m in cluster(samples, THRESHOLDS[kind]) if len(m) >= MIN_CLUSTER]
        labels = LABELS[kind]
        if len(clusters) != len(labels):
            raise SystemExit(f"{kind}: {len(clusters)} clusters but {len(labels)} labels")
        means = [samples[m].mean(0) for m in clusters]
        result[kind] = np.array(means, dtype=np.float32)
        result[f"{kind}_labels"] = tuple(labels)
        print(f"{kind}: " + " ".join(f"{lab}({len(m)})" for lab, m in zip(labels, clusters)))
        if args.sheets:
            args.sheets.mkdir(parents=True, exist_ok=True)
            write_sheet(means, labels, args.sheets / f"templates_{kind}.png")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Templates(**result).save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
