# Handball shot stats from video: plan v2

**Goal:** give the pipeline a clip (SEW-style eyecons stream, 1080p50, score overlay) and get back,
per team: shots, goals, saves, misses/blocks, where each shot was taken, and where it went in the goal.

**Constraints**
- As little hand-labeling as possible.
- **Everything runs locally.** Video understanding uses an **open-source VLM** on a GTX 1660 Super
  (6 GB VRAM, Turing: fp16 only, no bf16, no FlashAttention 2). No paid APIs.
- **Local hardware first, rented GPU last.** Renting a cloud GPU is allowed but not preferred. Long
  runtimes are acceptable (e.g. overnight batch runs, hours of training), as long as they are not
  extreme. Before proposing a rental, use up the local options:
  1. Smaller / edge models, 4-bit quantization, lower resolution, fewer and better-chosen frames.
  2. Cache everything (decoded windows, crops, model outputs) so no work is repeated.
  3. Run batch jobs overnight; process candidates in queues that can be resumed after a stop.
  4. CPU offload and gradient checkpointing for fine-tuning; LoRA/QLoRA on edge models only.
  5. Swap a weak VLM step for classical CV (motion, geometry) or a small local detector/classifier.
  
  A rental is only worth discussing when a specific step is shown (with measured numbers) to need
  more than 6 GB VRAM, or to take unreasonably long locally, *and* it clearly improves test-set
  accuracy. If so, rent for that one step only (e.g. a single fine-tuning run) and run inference
  locally.
- **All code must pass flake8** (config in `.flake8`, also enforced by CI on every push/PR).
- Python env: conda env `handball` (`environment.yml`).

## Why the first attempt stalled
It went bottom-up: detect court, players, ball and handler per frame, then hope stats fall out of
the tracks. Every stage needed hundreds of labeled frames, and the ball is a few pixels in size
and often blurred, so errors stacked up. Shot stats are **events**, not tracks. This plan detects
the events directly and only looks at frames closely around those events.

## What a 6 GB local VLM means for the design
A small VLM is slow (seconds per query) and weaker than API models, so:
- **It never watches the whole clip.** Cheap classical signals find about 60–100 candidate moments
  per half. The VLM only judges those.
- **Inputs are small and focused:** 8–16 frames per query, downscaled (about 448–672 px), plus a
  high-resolution **crop of the goal area**. We pick the frames ourselves; the VLM doesn't have to
  search through time.
- **Questions are narrow, with constrained JSON answers** (one field at a time if needed). Small
  models do much better on multiple-choice questions than on open-ended ones.
- **Model choice is made by measurement, not guessed** (M2). The models below all fit in 6 GB
  (fp16 or 4-bit). We check for newer releases at M2 time.

  | Group | Models | Why it's in the bake-off |
  |---|---|---|
  | Edge / small | **SmolVLM2** (256M / 500M / 2.2B), **Moondream** (~2B) | Fast on the 1660; SmolVLM2 handles video; Moondream has built-in point/detect (ball in the goal crop -> placement); small enough to LoRA fine-tune locally |
  | Mid (2-4B) | Qwen3-VL-2B/4B, Qwen2.5-VL-3B, InternVL3.5-2B/4B | Stronger at reasoning over several frames ("was it a shot, was it saved?"); slower, harder to fine-tune on 6 GB |

  Expected setup: **tiered.** A fast edge model screens candidates and points at the ball/goal. A
  mid-size model is used only for the questions where it measurably beats the edge model. The
  bake-off reports accuracy, seconds per query and VRAM. Backend: HF `transformers` (fp16 or
  `bitsandbytes` 4-bit); llama.cpp/Ollama GGUF or ONNX if faster.

## Pipeline

```
clip.mp4
 ├─ 1. Scoreboard reader (classical)     -> clock + score every second -> GOAL events (exact, free)
 ├─ 2. Possession / candidate finder     -> camera-pan direction flips + goal-area motion
 │     (classical, CPU/GPU)                 -> ~60-100 "attack ended here" moments per half
 ├─ 3. Shot judge (local VLM)            -> per candidate: shot? which team? outcome?
 ├─ 4. Shot detail (local VLM + geometry)-> shooter zone, 3x3 placement (goal-crop frames)
 ├─ 5. Consistency checks                -> VLM "goal" must match a scoreboard goal; flag disagreements
 └─ 6. Report                            -> shots.json/csv + HTML (shot chart, goal heatmap, timeline)
```

1. **Scoreboard reader.** The overlay has a fixed position and font. Crop it and read the digits
   with template matching; a small OCR model is the fallback. A score change gives an exact goal
   time and team. Clock stops mark breaks. This acts as **ground truth for every goal**, which is
   roughly half of all shots, without any labeling.
2. **Candidate finder.** The broadcast camera follows play, so the direction of global camera
   motion (phase correlation or optical flow at 10 fps) shows which half is under attack. A
   sustained pan reversal means an attack ended, with a shot, a turnover or a foul. Add motion
   energy in the goal area just before the reversal. Tune for recall; scoreboard goals give a
   free recall check. Optional booster: a small YOLO ball detector retrained from the 425
   ball/player labels we kept (images re-extracted from the clips, no new labeling).
3. **Shot judge (VLM).** Frames from about 3 s before to 1 s after the candidate. Multiple-choice
   questions: was there a shot on goal? Outcome: goal / save / miss / post / blocked / no shot.
4. **Shot detail.**
   - Zone: left/right wing, left/center/right back (9 m), pivot/6 m, 7 m penalty, fast break / empty goal.
     The VLM answers from the release frame and the wide view.
   - Placement: 3x3 grid seen from the shooter. The VLM answers from goal-area crops at full fps
     around the ball crossing. If the VLM is too weak here, fall back to geometry: detect the goal
     frame corners, find the ball (frame difference plus the ball detector) inside the goal frame,
     and map it onto the grid.
5. **Consistency checks.** If the VLM says "goal" without a score change, or a score change has no
   matching shot, the item goes into a review queue. Merge duplicate candidates.
6. **Report.** One JSON per clip plus an HTML page.

Later: exact x,y shot location. Retrain the court keypoint model (we kept 329 labels), compute a
homography, and project the shooter's feet onto a court diagram.

## Evaluation (the only unavoidable manual work)
- **Goals:** checked automatically by the scoreboard. They are also used to pick the VLM:
  "goal" windows vs random windows gives a free first benchmark.
- **Everything else:** run the pipeline on 1 half and **correct** its output in a small local
  review page (play a 4 s clip, click to fix fields). About 10 s per shot, about 30 shots, so
  5–10 minutes. That half becomes the fixed test set every change is scored against.
- Targets for v1: shot recall ≥ 90%, outcome accuracy ≥ 90%, zone ≥ 80%, placement ≥ 70%.

## Milestones
| # | Deliverable | Check |
|---|---|---|
| M0 | Repo skeleton, `handball` env, flake8 + CI, clip-window extractor (ffmpeg seek, crops) | `flake8` clean; extract any t ± 2 s window quickly |
| M1 | Scoreboard reader -> goals timeline for all 8 clips | Final scores correct |
| M2 | VLM bake-off on goal vs non-goal windows (accuracy, s/query, VRAM) | Pick a model + backend |
| M3 | Candidate finder | Recall vs scoreboard goals ≥ 95% |
| M4 | Review page + correct 1 half -> test set | Test set saved |
| M5 | Shot judge + shot detail + consistency checks | Accuracy vs test set; iterate prompts and frame selection |
| M6 | End-to-end `python -m handball analyze clip.mp4` -> JSON + HTML | Runs on all 8 clips; time per clip logged |
| M7 (opt.) | LoRA fine-tune an edge VLM (e.g. SmolVLM2) on corrected / pseudo-labeled shots | Better test-set accuracy |

## Code standards
- Package in `src/handball/`, entry point `python -m handball`.
- `flake8` must pass with zero warnings before every commit; GitHub Actions runs it on push/PR.
- No hard-coded absolute paths; settings go in a config file / CLI args.
- Videos, extracted frames, model weights and outputs are git-ignored. The repo is public, so no
  footage goes to GitHub.

## Data we have
- `clips/` (local only): 8 halves, 1080p50, ~30–36 min each, with score overlay.
- `labels/`: court keypoints (329), player/ball boxes (425), ball handler (129). See `labels/README.md`.

## Open questions
- Should stats be split by team (SEW vs opponent) or only for SEW?

No stat sheets exist for these games, so ground truth comes only from the scoreboard (goals) and
the corrected review half.
