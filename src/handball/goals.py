"""Goal events from score-overlay changes.

1. :func:`scan` reads the overlay on every keyframe (every 4 s; ~15 s per half).
2. :func:`stable_states` keeps scores seen on >= 2 consecutive readings, which removes
   one-off misreads and fade animations.
3. :func:`detect_goals` turns each score increase into goal events, refines the change
   to the exact frame, and undoes operator corrections (score going down).

Note: the overlay is updated by an operator, so ``t`` is when the *overlay* changed, typically
a few seconds after the ball crossed the line.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from handball.scoreboard import Reading, Templates, default_templates, read_frame
from handball.video import iter_frames, iter_keyframes

SCAN_VERSION = 2  # bump to invalidate cached scans
MIN_RUN = 2
REFINE_FULL_RATE_SECONDS = 12.0  # longer gaps are refined at REFINE_FPS
REFINE_FPS = 10.0


@dataclass(frozen=True)
class Sample:
    t: float
    index: int
    reading: Reading


@dataclass(frozen=True)
class State:
    score: tuple[int, int]
    first: Sample
    last: Sample
    count: int


@dataclass
class Goal:
    team: str  # "home" | "away"
    team_name: str | None
    t: float  # when the overlay changed (seconds into the clip)
    index: int  # frame number of that change
    score: str  # score after the goal, "home-away"
    clock: int | None  # game clock (seconds) shown at the change
    half: str | None
    flags: list[str] = field(default_factory=list)


@dataclass
class ClipGoals:
    clip: str
    home_name: str | None
    away_name: str | None
    half: str | None
    start_score: tuple[int, int] | None
    end_score: tuple[int, int] | None
    goals: list[Goal]
    corrections: int
    present_ratio: float
    clock_consistency: float | None

    def counts(self) -> tuple[int, int]:
        home = sum(g.team == "home" for g in self.goals)
        return home, len(self.goals) - home

    def consistent(self) -> bool:
        """Start score + detected goals == end score."""
        if self.start_score is None or self.end_score is None:
            return False
        home, away = self.counts()
        return (self.start_score[0] + home, self.start_score[1] + away) == self.end_score


# --------------------------------------------------------------------------- scanning

_READING_FIELDS = [f.name for f in fields(Reading)]


def _cache_path(cache_dir: Path, clip: Path) -> Path:
    return cache_dir / "scoreboard" / f"{clip.stem}.v{SCAN_VERSION}.csv"


def _parse(value: str, kind):
    if value == "":
        return None
    if kind is bool:
        return value == "True"
    return kind(value)


def _write_samples(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "index", *_READING_FIELDS])
        for s in samples:
            values = asdict(s.reading)
            writer.writerow([f"{s.t:.3f}", s.index] + [
                "" if values[k] is None else values[k] for k in _READING_FIELDS
            ])


def _read_samples(path: Path) -> list[Sample]:
    kinds = {"present": bool, "home": int, "away": int, "clock": int}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        Sample(
            t=float(r["t"]),
            index=int(r["index"]),
            reading=Reading(**{k: _parse(r[k], kinds.get(k, str)) for k in _READING_FIELDS}),
        )
        for r in rows
    ]


def scan(
    clip: Path,
    cache_dir: Path | None = None,
    templates: Templates | None = None,
    refresh: bool = False,
) -> list[Sample]:
    """Overlay reading for every keyframe of the clip (cached as CSV if ``cache_dir``)."""
    cache = _cache_path(cache_dir, clip) if cache_dir else None
    if cache and cache.exists() and not refresh:
        return _read_samples(cache)
    templates = templates or default_templates()
    samples = [Sample(f.t, f.index, read_frame(f.image, templates)) for f in iter_keyframes(clip)]
    if cache:
        _write_samples(cache, samples)
    return samples


# --------------------------------------------------------------------------- analysis


def stable_states(samples: Iterable[Sample], min_run: int = MIN_RUN) -> list[State]:
    """Runs of the same score over consecutive *readable* samples, at least ``min_run`` long.

    Unreadable samples (overlay hidden, misread) are skipped, not treated as breaks. Adjacent
    runs with the same score (split by a short misread) are merged.
    """
    runs: list[State] = []
    for sample in samples:
        score = sample.reading.score
        if score is None:
            continue
        if runs and runs[-1].score == score:
            prev = runs[-1]
            runs[-1] = State(score, prev.first, sample, prev.count + 1)
        else:
            runs.append(State(score, sample, sample, 1))
    merged: list[State] = []
    for run in (r for r in runs if r.count >= min_run):
        if merged and merged[-1].score == run.score:
            prev = merged[-1]
            merged[-1] = State(run.score, prev.first, run.last, prev.count + run.count)
        else:
            merged.append(run)
    return merged


def clock_consistency(samples: list[Sample]) -> float | None:
    """Share of consecutive keyframe pairs (4 s apart) whose clock readings agree.

    A running clock advances 3-5 s, a stopped one 0 s. Anything else means a misread (or a
    clock reset). Serves as a cheap accuracy estimate for the digit reader.
    """
    pairs = [
        (a.reading.clock, b.reading.clock)
        for a, b in zip(samples, samples[1:])
        if a.reading.clock is not None and b.reading.clock is not None
    ]
    if not pairs:
        return None
    ok = sum(1 for a, b in pairs if b - a == 0 or 3 <= b - a <= 5)
    return ok / len(pairs)


def _refine(
    clip: Path, before: State, after: State, templates: Templates
) -> dict[int, Sample]:
    """Per changed team (0 = home, 1 = away): the first frame showing that team's new score
    after the last frame showing its old one. Teams are refined separately because the
    operator sometimes updates both within one keyframe interval."""
    start, end = before.last.t, after.first.t
    fps = None if end - start <= REFINE_FULL_RATE_SECONDS else REFINE_FPS
    teams = [i for i in (0, 1) if after.score[i] != before.score[i]]
    first_new: dict[int, Sample | None] = {i: None for i in teams}
    for frame in iter_frames(clip, start, end + 1e-3, fps=fps):
        reading = read_frame(frame.image, templates)
        score = reading.score
        if score is None:
            continue
        for i in teams:
            if score[i] == before.score[i]:
                first_new[i] = None  # old value still showing: reset
            elif score[i] == after.score[i] and first_new[i] is None:
                first_new[i] = Sample(frame.t, frame.index, reading)
    return {i: first_new[i] or after.first for i in teams}


def _majority(values: Iterable[str | None]) -> str | None:
    counts = Counter(v for v in values if v is not None)
    return counts.most_common(1)[0][0] if counts else None


def detect_goals(
    clip: Path,
    samples: list[Sample],
    refine: bool = True,
    templates: Templates | None = None,
) -> ClipGoals:
    templates = templates or default_templates()
    home_name = _majority(s.reading.home_name for s in samples)
    away_name = _majority(s.reading.away_name for s in samples)
    half = _majority(s.reading.half for s in samples)
    states = stable_states(samples)

    goals: list[Goal] = []
    corrections = 0
    for before, after in zip(states, states[1:]):
        deltas = {
            "home": after.score[0] - before.score[0],
            "away": after.score[1] - before.score[1],
        }
        if refine:
            changes = _refine(clip, before, after, templates)
        else:
            changes = {i: after.first for i in (0, 1)}
        gap = after.first.t - before.last.t
        flags = []
        if gap > 2 * REFINE_FULL_RATE_SECONDS:
            flags.append(f"overlay_unreadable_{gap:.0f}s")
        if sum(d for d in deltas.values() if d > 0) > 1:
            flags.append("multiple_goals_one_interval")
        for team, delta in deltas.items():
            if delta < 0:
                # Operator correction: drop the most recent goal(s) of this team.
                for _ in range(-delta):
                    for i in range(len(goals) - 1, -1, -1):
                        if goals[i].team == team:
                            del goals[i]
                            break
                corrections += 1
        increments = sorted((
            (changes[i], i)
            for i, team in enumerate(deltas)
            for _ in range(max(0, deltas[team]))
        ), key=lambda item: (item[0].t, item[1]))
        running = list(before.score)
        for change, i in increments:
            running[i] += 1
            team = "home" if i == 0 else "away"
            goals.append(Goal(
                team=team,
                team_name=home_name if i == 0 else away_name,
                t=change.t,
                index=change.index,
                score=f"{running[0]}-{running[1]}",
                clock=change.reading.clock,
                half=half,
                flags=list(flags),
            ))
    present = sum(s.reading.present for s in samples)
    return ClipGoals(
        clip=clip.stem,
        home_name=home_name,
        away_name=away_name,
        half=half,
        start_score=states[0].score if states else None,
        end_score=states[-1].score if states else None,
        goals=goals,
        corrections=corrections,
        present_ratio=present / len(samples) if samples else 0.0,
        clock_consistency=clock_consistency(samples),
    )


def write_goals_csv(result: ClipGoals, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["clip", "team", "team_name", "t", "index", "score", "clock", "half", "flags"]
        )
        for g in result.goals:
            writer.writerow([
                result.clip, g.team, g.team_name, f"{g.t:.2f}", g.index, g.score,
                "" if g.clock is None else g.clock, g.half or "", ";".join(g.flags),
            ])
