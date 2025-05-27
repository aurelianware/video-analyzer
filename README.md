# Video Analyzer Pipeline

This repository contains a Python script (`video_analyzer_skip_frames.py`) that processes video files to detect objects, motion, and light anomalies using PyTorch and OpenCV, generates cropped snapshots, builds heatmaps, and produces annotated videos.

---

## 📦 Features

✅ Runs object detection using `Faster R-CNN` (with COCO weights)  
✅ Detects:
- People
- Surveillance devices (TV, laptop, camera, cell phone)
- Projectors/screens

✅ Detects motion and light anomalies  
✅ Crops and saves detected regions as images  
✅ Builds and saves per-category heatmaps  
✅ Overlays heatmaps on the original video  
✅ Skips every N frames and limits max frame count for faster runs  
✅ Outputs:
- `annotated_output.mp4` (with red detection boxes)
- `human_heatmap.png`, `motion_heatmap.png`, `light_heatmap.png`
- `annotated_with_heatmaps.mp4` (final heatmap overlay video)

---

## 🛠 Requirements

- Python 3.x  
- `torch`  
- `torchvision`  
- `opencv-python`  
- (optional) `tqdm` for progress bars

Install dependencies:
```bash
pip install torch torchvision opencv-python tqdm
