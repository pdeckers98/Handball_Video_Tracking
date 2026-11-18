# Handball Video Tracking

Extract shooting statistics (shots, goals, saves, misses, shot location and placement in the goal)
from amateur handball match video, running fully locally with an open-source vision-language model.

Status: planning / early development. See [PLAN.md](PLAN.md).

## Setup

```bash
conda env create -f environment.yml
conda activate handball
flake8 .
```

Match footage is not included in this repository. Put clips in `clips/` (git-ignored).
Existing hand labels are in [labels/](labels/README.md).
