"""Command line interface: ``python -m handball <command> ...``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from handball.config import DEFAULT_CONFIG, load_config
from handball.timecode import format_timecode, parse_timecode
from handball.video import Region, probe, read_window, write_images, write_video


def _cmd_info(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    for clip in args.clips:
        info = probe(config.resolve_clip(clip))
        print(
            f"{info.path.name}: {info.width}x{info.height} @ {info.fps:.3f} fps, "
            f"{info.n_frames} frames, {format_timecode(info.duration)}"
        )
    return 0


def _cmd_window(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    clip = config.resolve_clip(args.clip)
    center = parse_timecode(args.at)
    started = time.perf_counter()
    frames = read_window(
        clip,
        center,
        before=args.before,
        after=args.after,
        fps=args.fps,
        crop=Region.parse(args.crop) if args.crop else None,
        width=args.width,
    )
    elapsed = time.perf_counter() - started
    if not frames:
        print(f"no frames around {format_timecode(center)}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else (
        config.paths.outputs / "windows" / f"{clip.stem}_{center:.2f}.mp4"
    )
    if out.suffix.lower() == ".mp4":
        out_fps = args.fps or probe(clip).fps
        write_video(frames, out, out_fps)
    else:
        write_images(frames, out)
    print(
        f"{len(frames)} frames {format_timecode(frames[0].t)}-{format_timecode(frames[-1].t)} "
        f"({frames[0].image.shape[1]}x{frames[0].image.shape[0]}) "
        f"read in {elapsed:.2f}s -> {out}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="handball", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="show clip properties")
    info.add_argument("clips", nargs="+", help="clip names (clip1) or paths")
    info.set_defaults(func=_cmd_info)

    window = sub.add_parser("window", help="extract a short window around a timestamp")
    window.add_argument("clip", help="clip name (clip1) or path")
    window.add_argument("--at", required=True, help="center time: seconds, m:ss or h:mm:ss")
    window.add_argument("--before", type=float, default=2.0, help="seconds before (default 2)")
    window.add_argument("--after", type=float, default=2.0, help="seconds after (default 2)")
    window.add_argument("--fps", type=float, help="sample rate (default: native)")
    window.add_argument("--crop", help="normalized region x,y,w,h, e.g. 0.4,0.0,0.3,0.2")
    window.add_argument("--width", type=int, help="output width in pixels")
    window.add_argument("--out", help="output .mp4 file or a folder for JPEGs")
    window.set_defaults(func=_cmd_window)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
