# Labels

All labels are YOLO-format `.txt`, coordinates normalized to the 1920x1080 frame.
File names are `<clip>_frame<N>.txt` (frame N of `clips/<clip>.mp4`, 50 fps) or
`pt2_<clip>_frame_<N>.txt` (same meaning, second labeling pass).
Images were deleted; re-extract frame N from the clip to rebuild a dataset.

| Folder | Frames | Format | Classes |
|---|---|---|---|
| `court_keypoints/` | 329 | YOLO pose: `cls cx cy w h` + 16 x (`x y v`), v: 0=not labeled, 1=occluded, 2=visible | 0 = court |
| `player_ball/` | 425 | YOLO detect: `cls cx cy w h` | 0 = person, 1 = ball |
| `ball_handler/` | 129 | YOLO detect: `cls cx cy w h` | 0 = handler (player holding the ball) |

Labels were made in CVAT (court keypoints) and X-AnyLabeling (boxes).
