from fractions import Fraction

import av
import numpy as np
import pytest

from handball.video import Frame, Region, iter_frames, probe, read_window, write_video

FPS = 25
SECONDS = 6
W, H = 320, 240


def _level(i: int) -> int:
    """Brightness that encodes the frame number (survives lossy compression)."""
    return (i * 9) % 200 + 20


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """Synthetic clip: frame i is uniformly grey at _level(i); keyframe every 2 s."""
    path = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(FPS))
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = {"g": str(2 * FPS), "crf": "10"}
        for i in range(FPS * SECONDS):
            image = np.full((H, W, 3), _level(i), dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_probe(video):
    info = probe(video)
    assert (info.width, info.height) == (W, H)
    assert info.fps == pytest.approx(FPS)
    assert info.duration == pytest.approx(SECONDS, abs=0.1)


def test_window_is_frame_accurate(video):
    # Window starts mid-GOP, so this checks seek + decode-forward.
    frames = read_window(video, center=3.3, before=0.5, after=0.5)
    assert len(frames) == FPS  # 1 s of video
    assert frames[0].t == pytest.approx(2.8, abs=1 / FPS)
    for frame in frames:
        assert frame.index == round(frame.t * FPS)
        assert frame.image.mean() == pytest.approx(_level(frame.index), abs=4)


def test_sampling_fps(video):
    frames = list(iter_frames(video, start=1.0, end=3.0, fps=5))
    assert len(frames) == 10
    gaps = np.diff([f.t for f in frames])
    assert np.allclose(gaps, 0.2, atol=1 / FPS)


def test_crop_and_resize(video):
    region = Region(0.5, 0.5, 0.5, 0.5)
    frames = read_window(video, center=1.0, before=0, after=0.2, crop=region, width=80)
    assert frames[0].image.shape == (60, 80, 3)


def test_window_clamped_at_start_and_end(video):
    assert read_window(video, center=0.0, before=1.0, after=0.2)[0].t == pytest.approx(0)
    assert len(read_window(video, center=SECONDS + 5)) == 0


def test_region_validation():
    assert Region.parse("0.1,0.2,0.3,0.4") == Region(0.1, 0.2, 0.3, 0.4)
    assert Region(0, 0, 1, 1).to_pixels(1920, 1080) == (0, 0, 1920, 1080)
    with pytest.raises(ValueError):
        Region(0.8, 0, 0.5, 0.5)
    with pytest.raises(ValueError):
        Region.parse("0.1,0.2,0.3")


def test_write_video_roundtrip(video, tmp_path):
    frames = read_window(video, center=2.0, before=0.4, after=0.4, fps=12.5)
    out = tmp_path / "out.mp4"
    assert write_video(frames, out, fps=12.5) == len(frames)
    info = probe(out)
    assert info.fps == pytest.approx(12.5)
    assert info.n_frames == len(frames)


def test_write_video_odd_size(tmp_path):
    frames = [Frame(t=i / 10, index=i, image=np.zeros((51, 75, 3), np.uint8)) for i in range(3)]
    out = tmp_path / "odd.mp4"
    assert write_video(frames, out, fps=10) == 3
    assert (probe(out).width, probe(out).height) == (74, 50)
