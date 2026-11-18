"""M2: compare local VLMs on two tasks with free labels.

T1 "goal or not": 16 frames (1 fps, 448 px wide, score overlay blacked out) from the 16 s
   before a scoreboard goal update (positive) or from a random stretch >= 20 s away from any
   score change (negative; may contain saves/misses). Metric: ROC-AUC of P(yes), plus balanced
   accuracy at 0.5.
T2 "find the ball": single frames with one hand-labeled ball (labels/player_ball), 1280 px
   wide. Metric: share of answers within 25 px (at 1080p) of the ball centre.

Usage:
    python tools/vlm_bakeoff.py build                 # build + cache test sets
    python tools/vlm_bakeoff.py run smolvlm2-500m qwen3-vl-2b ...
    python tools/vlm_bakeoff.py report
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import traceback

import numpy as np
from PIL import Image

from handball.config import load_config
from handball.video import iter_frames

SEED = 7
T1_PER_CLIP = 6
T1_SECONDS = 16
T1_WIDTH = 448
T2_ITEMS = 40
T2_WIDTH = 1280
T2_HIT_PX = 25  # at 1080p
OVERLAY = (0.375, 0.05, 0.62, 0.185)  # x0, y0, x1, y1 normalized: score overlay to black out

T1_QUESTION = (
    "These are {n} frames from a handball match, one per second, in time order. "
    "Was a goal scored in this sequence (the ball thrown into the net)?"
)
T2_TARGET = "handball (the small ball the players throw)"


def _mask_overlay(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = OVERLAY
    image = image.copy()
    image[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
    return image


def _dirs():
    config = load_config()
    cache = config.paths.cache / "bakeoff"
    out = config.paths.outputs / "bakeoff"
    return config, cache, out


# --------------------------------------------------------------------------- build


def build() -> None:
    config, cache, _ = _dirs()
    rng = random.Random(SEED)
    items = []
    for clip in sorted(config.paths.clips.glob("*.mp4")):
        goals_csv = config.paths.outputs / "goals" / f"{clip.stem}.csv"
        goals = list(csv.DictReader(goals_csv.open()))
        change_times = sorted({float(g["t"]) for g in goals})
        clean = [g for g in goals if not g["flags"] and float(g["t"]) > T1_SECONDS + 1]
        step = max(1, len(clean) // T1_PER_CLIP)
        positives = [float(g["t"]) for g in clean[::step][:T1_PER_CLIP]]
        duration = max(change_times[-1] if change_times else 0, 60)
        negatives = []
        while len(negatives) < T1_PER_CLIP:
            end = rng.uniform(T1_SECONDS + 1, duration)
            if all(not (end - T1_SECONDS - 5 <= c <= end + 20) for c in change_times):
                negatives.append(end)
        for label, ends in ((1, positives), (0, negatives)):
            for end in ends:
                item_id = f"{clip.stem}_{'pos' if label else 'neg'}_{end:.0f}"
                folder = cache / "t1" / item_id
                folder.mkdir(parents=True, exist_ok=True)
                frames = iter_frames(clip, end - T1_SECONDS, end, fps=1, width=T1_WIDTH)
                for k, frame in enumerate(frames):
                    Image.fromarray(_mask_overlay(frame.image)).save(folder / f"{k:02d}.jpg")
                items.append({"id": item_id, "clip": clip.stem, "end": end, "label": label})
        print(f"T1 {clip.stem}: {len(positives)} pos, {len(negatives)} neg")
    (cache / "t1.json").write_text(json.dumps(items, indent=1))

    labels = sorted((config.root / "labels" / "player_ball").glob("*.txt"))
    rng.shuffle(labels)
    (cache / "t2").mkdir(parents=True, exist_ok=True)
    t2 = []
    for path in labels:
        balls = [r.split() for r in path.read_text().splitlines() if r.startswith("1 ")]
        if len(balls) != 1:
            continue
        m = re.match(r"(?:pt2_)?(clip\d+)_frame_?(\d+)$", path.stem)
        clip, index = config.resolve_clip(m.group(1)), int(m.group(2))
        frame = next(iter_frames(clip, index / 50 - 1e-3, index / 50 + 0.03, width=T2_WIDTH))
        Image.fromarray(_mask_overlay(frame.image)).save(cache / "t2" / f"{path.stem}.jpg")
        cx, cy = float(balls[0][1]), float(balls[0][2])
        t2.append({"id": path.stem, "x": cx, "y": cy})
        if len(t2) == T2_ITEMS:
            break
    (cache / "t2.json").write_text(json.dumps(t2, indent=1))
    print(f"T1 {len(items)} items, T2 {len(t2)} items -> {cache}")


# --------------------------------------------------------------------------- run


def auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _mean_seconds(records: list[dict]) -> float | None:
    timed = [r["seconds"] for r in records if r.get("seconds") is not None]
    return round(float(np.mean(timed)), 2) if timed else None


def _summarize(name: str, records: list[dict], t1: list, t2: list) -> dict:
    summary: dict = {"model": name}
    r1 = [r for r in records if r["task"] == "t1"]
    r2 = [r for r in records if r["task"] == "t2"]
    summary["t1_done"] = f"{len(r1)}/{len(t1)}"
    summary["t2_done"] = f"{len(r2)}/{len(t2)}"
    if r1:
        scores = [r["p_yes"] for r in r1]
        labels = [r["label"] for r in r1]
        pos = [s for s, y in zip(scores, labels) if y == 1]
        neg = [s for s, y in zip(scores, labels) if y == 0]
        if pos and neg:
            summary["t1_auc"] = round(auc(scores, labels), 3)
            tpr, tnr = np.mean([s >= 0.5 for s in pos]), np.mean([s < 0.5 for s in neg])
            summary["t1_bal_acc"] = round(float(tpr + tnr) / 2, 3)
            summary["t1_mean_p_goal"] = round(float(np.mean(pos)), 3)
            summary["t1_mean_p_nogoal"] = round(float(np.mean(neg)), 3)
        summary["t1_s_per_q"] = _mean_seconds(r1)
    if r2:
        dists = [
            float(np.hypot((r["point"][0] - r["x"]) * 1920, (r["point"][1] - r["y"]) * 1080))
            for r in r2 if r["point"] is not None
        ]
        summary["t2_hit"] = round(sum(d <= T2_HIT_PX for d in dists) / len(r2), 3)
        summary["t2_answered"] = round(len(dists) / len(r2), 3)
        summary["t2_median_px"] = round(float(np.median(dists))) if dists else None
        summary["t2_s_per_q"] = _mean_seconds(r2)
    return summary


def run_model(name: str) -> dict:
    """Run (or resume) one model; results per item are appended to outputs/bakeoff/<name>.jsonl."""
    import torch

    from handball.vlm import load

    _, cache, out = _dirs()
    out.mkdir(parents=True, exist_ok=True)
    t1 = json.loads((cache / "t1.json").read_text())
    t2 = json.loads((cache / "t2.json").read_text())
    log_path = out / f"{name}.jsonl"
    records = []
    if log_path.exists():
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    done = {(r["task"], r["id"]) for r in records}
    extra: dict = {}
    with log_path.open("w", encoding="utf-8") as log:
        for r in records:
            log.write(json.dumps(r) + "\n")
        try:
            todo1 = [i for i in t1 if ("t1", i["id"]) not in done]
            todo2 = [i for i in t2 if ("t2", i["id"]) not in done]
            if todo1 or todo2:
                vlm = load(name)
                extra.update(load_s=round(vlm.info.load_seconds),
                             vram_load_gb=round(vlm.info.vram_gb, 2))
                torch.cuda.reset_peak_memory_stats()
                for item in todo1:
                    folder = cache / "t1" / item["id"]
                    frames = [Image.open(p) for p in sorted(folder.glob("*.jpg"))]
                    started = time.perf_counter()
                    p = vlm.yes_probability(frames, T1_QUESTION.format(n=len(frames)))
                    rec = {"task": "t1", **item, "p_yes": p,
                           "seconds": time.perf_counter() - started,
                           "tokens": getattr(vlm, "last_input_tokens", None)}
                    records.append(rec)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                for item in todo2:
                    image = Image.open(cache / "t2" / f"{item['id']}.jpg")
                    started = time.perf_counter()
                    point = vlm.point(image, T2_TARGET)
                    rec = {"task": "t2", **item, "point": point,
                           "seconds": time.perf_counter() - started,
                           "answer": vlm.last_answer}
                    records.append(rec)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                sample = sorted((cache / "t1" / t1[0]["id"]).glob("*.jpg"))
                extra["describe"] = vlm.generate(
                    [Image.open(p) for p in sample[::2]],
                    "These frames are from a handball match, in time order. Briefly describe "
                    "what happens.",
                    max_new_tokens=80,
                )
                extra["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                vlm.close()
        except Exception as exc:  # keep going with the next model
            extra["error"] = f"{type(exc).__name__}: {exc}"[:300]
            traceback.print_exc()
    summary = {**_summarize(name, records, t1, t2), **extra}
    with (out / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=1))
    return summary


def report() -> None:
    _, _, out = _dirs()
    rows = {}
    for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row["model"]] = row  # latest run per model
    cols = ["model", "t1_done", "t1_auc", "t1_bal_acc", "t1_s_per_q", "t2_done", "t2_hit",
            "t2_median_px", "t2_s_per_q", "vram_peak_gb", "error"]
    print(" | ".join(cols))
    for row in rows.values():
        print(" | ".join(str(row.get(c, "")) for c in cols))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    run = sub.add_parser("run")
    run.add_argument("models", nargs="+")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "build":
        build()
    elif args.cmd == "run":
        for name in args.models:
            run_model(name)
    else:
        report()


if __name__ == "__main__":
    main()
