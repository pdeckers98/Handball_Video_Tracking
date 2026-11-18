"""Fast, accurate frame access for match clips.

Frames are located by timestamp: seek to the keyframe before ``start`` and decode forward,
so a few-second window costs well under a second regardless of where it sits in the clip.
Images are RGB ``uint8`` arrays of shape ``(height, width, 3)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

_EPS = 1e-6


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    n_frames: int
    width: int
    height: int
    duration: float


@dataclass(frozen=True)
class Region:
    """Rectangle in normalized coordinates (0..1) so it is independent of resolution."""

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError("region width and height must be > 0")
        if self.x < 0 or self.y < 0 or self.x + self.w > 1 + _EPS or self.y + self.h > 1 + _EPS:
            raise ValueError(f"region outside the frame: {self}")

    @classmethod
    def parse(cls, text: str) -> Region:
        """Parse ``"x,y,w,h"`` (normalized floats)."""
        values = [float(v) for v in text.split(",")]
        if len(values) != 4:
            raise ValueError(f"expected x,y,w,h, got {text!r}")
        return cls(*values)

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return ``(x0, y0, x1, y1)`` pixel bounds, at least 1 px in size."""
        x0 = int(round(self.x * width))
        y0 = int(round(self.y * height))
        x1 = max(x0 + 1, min(width, int(round((self.x + self.w) * width))))
        y1 = max(y0 + 1, min(height, int(round((self.y + self.h) * height))))
        return x0, y0, x1, y1


@dataclass(frozen=True)
class Frame:
    t: float
    """Presentation time in seconds."""
    index: int
    """Frame number (``round(t * fps)``)."""
    image: np.ndarray


def _nominal_fps(stream: av.video.stream.VideoStream) -> float:
    """Frame rate snapped to a simple fraction (50.0005 -> 50, keeps 30000/1001)."""
    rate = stream.average_rate or stream.guessed_rate
    if not rate:
        raise ValueError("video stream has no frame rate")
    rate = float(rate)
    if abs(rate - round(rate)) < 0.01:
        return float(round(rate))
    ntsc = round(rate * 1.001)
    if abs(rate - ntsc / 1.001) < 0.01:
        return float(Fraction(ntsc * 1000, 1001))
    return rate


def probe(path: str | Path) -> VideoInfo:
    """Read basic stream properties without decoding."""
    path = Path(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = _nominal_fps(stream)
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = float(container.duration or 0) / av.time_base
        n_frames = stream.frames or int(round(duration * fps))
        return VideoInfo(path, fps, n_frames, stream.width, stream.height, duration)


def _prepare(image: np.ndarray, crop: Region | None, width: int | None) -> np.ndarray:
    if crop is not None:
        x0, y0, x1, y1 = crop.to_pixels(image.shape[1], image.shape[0])
        image = image[y0:y1, x0:x1]
    if width is not None and width != image.shape[1]:
        height = max(1, int(round(image.shape[0] * width / image.shape[1])))
        image = np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))
    return np.ascontiguousarray(image)


def iter_frames(
    path: str | Path,
    start: float = 0.0,
    end: float | None = None,
    fps: float | None = None,
    crop: Region | None = None,
    width: int | None = None,
) -> Iterator[Frame]:
    """Yield frames with ``start <= t < end``.

    Args:
        fps: sample rate; ``None`` keeps every frame. Sampling picks the first frame at or
            after each ``start + k / fps``.
        crop: optional region, applied before resizing.
        width: optional output width in pixels (aspect ratio kept).
    """
    if fps is not None and fps <= 0:
        raise ValueError("fps must be > 0")
    start = max(0.0, start)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        native_fps = _nominal_fps(stream)
        if start > 0:
            container.seek(int(start / stream.time_base), stream=stream, backward=True)
        next_t = start
        index = None
        for frame in container.decode(stream):
            # The clips were cut with a stream copy, which garbled B-frame timestamps (pairs
            # of frames 1 tick apart, then a 2-frame gap). Decode order is still correct, so
            # anchor on each keyframe (or the first frame) and count frames from there.
            if frame.key_frame or index is None:
                if frame.time is None:
                    continue
                index = int(round(float(frame.time) * native_fps))
            else:
                index += 1
            t = index / native_fps
            if t < start - _EPS:
                continue
            if end is not None and t >= end - _EPS:
                break
            if fps is not None:
                if t < next_t - _EPS:
                    continue
                while next_t <= t + _EPS:
                    next_t += 1.0 / fps
            if crop is None and width is not None:
                # Let FFmpeg's scaler resize: much faster than converting at full size.
                height = max(1, int(round(frame.height * width / frame.width)))
                image = frame.to_ndarray(format="rgb24", width=width, height=height)
            else:
                image = _prepare(frame.to_ndarray(format="rgb24"), crop, width)
            yield Frame(t=t, index=index, image=image)


def read_window(
    path: str | Path,
    center: float,
    before: float = 2.0,
    after: float = 2.0,
    **kwargs,
) -> list[Frame]:
    """Return frames in ``[center - before, center + after)``; see :func:`iter_frames`."""
    return list(iter_frames(path, center - before, center + after, **kwargs))


def write_video(frames: Iterable[Frame], out_path: str | Path, fps: float) -> int:
    """Encode frames as browser-playable H.264 MP4. Returns the number of frames written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with av.open(str(out_path), mode="w") as container:
        stream = None
        for frame in frames:
            image = frame.image
            # yuv420p needs even dimensions.
            image = image[: image.shape[0] // 2 * 2, : image.shape[1] // 2 * 2]
            if stream is None:
                stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
                stream.width = image.shape[1]
                stream.height = image.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "20", "preset": "veryfast"}
            video_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
            count += 1
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
    if count == 0:
        out_path.unlink(missing_ok=True)
    return count


def write_images(frames: Iterable[Frame], out_dir: str | Path, ext: str = "jpg") -> list[Path]:
    """Save frames as ``<out_dir>/f<index>_<t>.<ext>``. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for frame in frames:
        path = out_dir / f"f{frame.index:07d}_{frame.t:09.3f}.{ext}"
        Image.fromarray(frame.image).save(path, quality=92)
        written.append(path)
    return written
