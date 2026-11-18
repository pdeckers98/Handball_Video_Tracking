"""Project configuration loaded from ``config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("config.yaml")


@dataclass(frozen=True)
class Paths:
    clips: Path
    cache: Path
    outputs: Path


@dataclass(frozen=True)
class Config:
    root: Path
    paths: Paths

    def resolve_clip(self, clip: str | Path) -> Path:
        """Return a clip path from a path or a bare name such as ``clip1``."""
        candidate = Path(clip)
        if candidate.exists():
            return candidate
        for name in (candidate.name, f"{candidate.name}.mp4"):
            path = self.paths.clips / name
            if path.exists():
                return path
        raise FileNotFoundError(f"clip not found: {clip} (looked in {self.paths.clips})")


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    """Load the config file; relative paths are resolved against its folder."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = path.parent
    raw = data.get("paths", {})

    def _path(key: str, default: str) -> Path:
        value = Path(raw.get(key, default))
        return value if value.is_absolute() else root / value

    paths = Paths(
        clips=_path("clips", "clips"),
        cache=_path("cache", "cache"),
        outputs=_path("outputs", "outputs"),
    )
    return Config(root=root, paths=paths)
