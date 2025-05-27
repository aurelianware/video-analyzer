import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision import transforms

print("✅ Starting full pipeline...")

print("ℹ If you want progress bars, install tqdm by running: pip install tqdm")
try:
    from tqdm import tqdm
    use_tqdm = True
except ImportError:
    print("⚠ tqdm not installed — running without progress bars")
    use_tqdm = False

# CONFIG
MAX_FRAMES = 200           # cap max frames
SKIP_FRAMES = 2            # process every 2nd frame
CONFIDENCE_THRESHOLD = 0.5 # updated from 0.3

# STEP 1: Run detection on video
video_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_path)
frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('annotated_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
COCO_INSTANCE_CATEGORY_NAMES = weights.meta['categories']
transform = transforms.Compose([transforms.ToTensor()])

for folder in ['cropped_light_anomalies', 'cropped_motion', 'cropped_humans', 'cropped_surveillance', 'cropped_projection']:
    os.makedirs(folder, exist_ok=True)

prev_frame = None
frame_count = 0
processed_count = 0

detection_count = {'light': 0, 'motion': 0, 'humans': 0, 'surveillance': 0, 'projection': 0}

print("▶ Processing video frames...")
frame_iter = tqdm(range(frame_count_total)) if use_tqdm else range(frame_count_total)
for i in frame_iter:
    if processed_count >= MAX_FRAMES:
        break
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for speed
    if i % SKIP_FRAMES != 0:
        continue

    processed_count += 1
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped = frame[y:y + h, x:x + w]
        cv2.imwrite(f'cropped_light_anomalies/frame{frame_count}_anomaly{idx}.jpg', cropped)
        detection_count['light'] += 1

    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, gray)
        _, diff_thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        motion_contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for j, contour in enumerate(motion_contours):
            x, y, w, h = cv2.boundingRect(contour)
            cropped = frame[y:y + h, x:x + w]
            cv2.imwrite(f'cropped_motion/frame{frame_count}_motion{j}.jpg', cropped)
            detection_count['motion'] += 1

    prev_frame = gray
    img_tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    for k, (box, score, label) in enumerate(zip(predictions['boxes'], predictions['scores'], predictions['labels'])):
        if score > CONFIDENCE_THRESHOLD:
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            x1, y1, x2, y2 = map(int, box.tolist())

            # Draw red rectangle on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cropped = frame[y1:y2, x1:x2]
            if class_name == 'person':
                cv2.imwrite(f'cropped_humans/frame{frame_count}_person{k}.jpg', cropped)
                detection_count['humans'] += 1
            elif class_name in ['tv', 'laptop', 'camera', 'cell phone']:
                cv2.imwrite(f'cropped_surveillance/frame{frame_count}_device{k}.jpg', cropped)
                detection_count['surveillance'] += 1
            elif class_name in ['projector', 'screen']:
                cv2.imwrite(f'cropped_projection/frame{frame_count}_proj{k}.jpg', cropped)
                detection_count['projection'] += 1

    out.write(frame)

cap.release()
out.release()
print("✅ Step 1 complete: annotated_output.mp4 saved")

# STEP 2: Generate enhanced heatmaps
heatmap_shape = (frame_height, frame_width)
human_heatmap = np.zeros(heatmap_shape, dtype=np.float32)
motion_heatmap = np.zeros(heatmap_shape, dtype=np.float32)
light_heatmap = np.zeros(heatmap_shape, dtype=np.float32)

print("▶ Building heatmaps...")
for category, folder in [('humans', 'cropped_humans'), ('motion', 'cropped_motion'), ('light', 'cropped_light_anomalies')]:
    files = tqdm(os.listdir(folder)) if use_tqdm else os.listdir(folder)
    for filename in files:
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (frame_width, frame_height))
            boost = 3.0
            if category == 'humans':
                human_heatmap += (img_resized / 255.0) * boost
            elif category == 'motion':
                motion_heatmap += (img_resized / 255.0) * boost
            elif category == 'light':
                light_heatmap += (img_resized / 255.0) * boost

human_heatmap = cv2.normalize(human_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
motion_heatmap = cv2.normalize(motion_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
light_heatmap = cv2.normalize(light_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

human_color = cv2.applyColorMap(human_heatmap, cv2.COLORMAP_JET)
motion_color = cv2.applyColorMap(motion_heatmap, cv2.COLORMAP_JET)
light_color = cv2.applyColorMap(light_heatmap, cv2.COLORMAP_JET)

cv2.imwrite('human_heatmap.png', human_color)
cv2.imwrite('motion_heatmap.png', motion_color)
cv2.imwrite('light_heatmap.png', light_color)
print("✅ Step 2 complete: heatmaps saved")

# STEP 3: Overlay heatmaps onto annotated video
cap = cv2.VideoCapture('annotated_output.mp4')
out = cv2.VideoWriter('annotated_with_heatmaps.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

alpha = 0.4
print("▶ Overlaying heatmaps onto video...")
for _ in tqdm(range(processed_count)) if use_tqdm else range(processed_count):
    ret, frame = cap.read()
    if not ret:
        break
    combined = cv2.addWeighted(frame, 1, human_color, alpha, 0)
    combined = cv2.addWeighted(combined, 1, motion_color, alpha, 0)
    combined = cv2.addWeighted(combined, 1, light_color, alpha, 0)
    out.write(combined)

cap.release()
out.release()
print("✅ Step 3 complete: final video saved as annotated_with_heatmaps.mp4")
print("🎉 Full pipeline completed!")
