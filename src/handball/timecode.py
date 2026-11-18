"""Parse and format clip timestamps (seconds, ``m:ss.s`` or ``h:mm:ss.s``)."""

from __future__ import annotations


def parse_timecode(value: str | float | int) -> float:
    """Convert ``"75.5"``, ``"1:15.5"`` or ``"0:01:15.5"`` to seconds."""
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        parts = value.strip().split(":")
        if not 1 <= len(parts) <= 3 or any(p == "" for p in parts):
            raise ValueError(f"invalid timecode: {value!r}")
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
    if seconds < 0:
        raise ValueError(f"timecode must be >= 0: {value!r}")
    return seconds


def format_timecode(seconds: float) -> str:
    """Format seconds as ``m:ss.ss`` (or ``h:mm:ss.ss`` past one hour)."""
    total = max(0.0, seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours >= 1:
        return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"
    return f"{int(minutes)}:{secs:05.2f}"
