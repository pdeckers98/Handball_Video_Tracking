# Handball Video Tracking

Extract shooting statistics (shots, goals, saves, misses, shot location and placement in the goal)
from amateur handball match video, running fully locally with an open-source vision-language model.

Status: early development (M0-M1 done). See [PLAN.md](PLAN.md).

## Setup

```bash
conda env create -f environment.yml
conda activate handball
flake8 . && pytest -q
```

Match footage is not included in this repository. Put clips in `clips/` (git-ignored).
Existing hand labels are in [labels/](labels/README.md). Paths are set in `config.yaml`.

## Usage

```bash
# Clip properties
python -m handball info clip1 clip2

# 4 s window around 8:00 (2 s before, 2 s after) as H.264 MP4 -> outputs/windows/
python -m handball window clip1 --at 8:00

# 10 fps, downscaled to 640 px wide
python -m handball window clip1 --at 8:00 --fps 10 --width 640

# Crop a region (normalized x,y,w,h) and save JPEGs to a folder
python -m handball window clip1 --at 20:00 --after 0.1 --crop 0.38,0.03,0.24,0.14 --out outputs/sb

# Goal events from the score overlay (all clips; -v lists every goal) -> outputs/goals/<clip>.csv
python -m handball goals -v clip1
```

In Python:

```python
from handball.video import read_window, iter_frames, Region

frames = read_window("clips/clip1.mp4", center=480, before=2, after=2, width=640)
for f in iter_frames("clips/clip1.mp4", start=0, end=60, fps=2):
    ...  # f.t (s), f.index (frame number), f.image (RGB uint8 array)
```

Note: the clips were cut with a stream copy, which garbled their B-frame timestamps. `handball.video`
times frames from keyframes plus decode order, so `t` and `index` are exact; don't use raw PyAV/OpenCV
timestamps on these files.
