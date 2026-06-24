import cv2
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import csv
from datetime import datetime
import os

# Load model and set parameters
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# ========== CONFIGURATION: Edit these values directly ==========
RESOLUTION = "2k"  # Choose: "original", "1080p" (1920x1080), or "2k" (2560x1440)
MIN_CONFIDENCE = 0.7  # Minimum confidence threshold for detections/tracking (0.0 to 1.0)
# ================================================================

model = YOLO(r"C:\Users\rigdo\OneDrive\Documents\YOLO v26\ping_modelv5\best.pt")

# Open video capture - prefer `ball_video.mp4` but fall back to any local video or webcam
video_path = "ball_video.mp4"
cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])
BALL_CLASS_ID = 0 # COCO class for sports ball

# Check if video was successfully opened
if not cap.isOpened():
    # Try to find any video file in the current directory
    candidates = [f for f in os.listdir(os.getcwd()) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if len(candidates) > 0:
        video_path = candidates[0]
        print(f"ball_video.mp4 not found - opening first video file: {video_path}")
        cap = cv2.VideoCapture(video_path)

    # Still not opened? try webcam index 0
    if not cap.isOpened():
        print("No local video files found or failed to open. Trying webcam (index 0)...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open video source (ball_video.mp4, local files, or webcam)")
        print(f"Make sure a video file is present in: {os.getcwd()}, or that a webcam is connected.")
        exit()

# Setup video output
fps = int(cap.get(cv2.CAP_PROP_FPS))
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Determine target resolution
if RESOLUTION == "original":
    target_width, target_height = orig_width, orig_height
elif RESOLUTION == "1080p":
    target_width, target_height = 1920, 1080
else:  # "2k"
    target_width, target_height = 2560, 1440

output_video = os.path.join(os.getcwd(), f"ball_tracking_output_{RESOLUTION}.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (target_width, target_height))

# Setup CSV logging
csv_file = os.path.join(os.getcwd(), "tracking_results.csv")
csv_writer = None
csv_handle = None

print(f"Output will be saved to: {os.getcwd()}")

frame_count = 0
frames_with_detection = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_count += 1

    # Resize frame to target resolution (if requested)
    if (frame.shape[1], frame.shape[0]) != (target_width, target_height):
        try:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"Warning: failed to resize frame to {(target_width, target_height)}: {e}")
    
    # Initialize CSV writer on first frame
    if csv_writer is None:
        csv_handle = open(csv_file, 'w', newline='')
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(['Frame', 'Ball_ID', 'X', 'Y', 'Width', 'Height', 'Velocity', 'Confidence'])

    if frame_count == 1:
        print(f"Original capture resolution: ({orig_width}, {orig_height})")
        print(f"Target processing resolution: ({target_width}, {target_height})")
        print(f"Frame dtype: {frame.dtype}")

    # Perform tracking using the built-in Ultralytics tracker with ByteTrack config
    results = model.track(frame, persist=True, classes=[BALL_CLASS_ID], tracker="bytetrack.yaml", conf=MIN_CONFIDENCE, iou=0.5)
    r = results[0]
    num_boxes = len(r.boxes.xyxy) if getattr(r.boxes, 'xyxy', None) is not None else 0
    print(f"Frame {frame_count}: ball-class track boxes = {num_boxes}")

    if num_boxes == 0 and frame_count <= 5:
        # Debug: run a normal detection pass without class filtering to see what the model detects
        debug_results = model(frame, conf=0.1, iou=0.1)
        debug_boxes = debug_results[0].boxes
        debug_count = len(debug_boxes.xyxy) if getattr(debug_boxes, 'xyxy', None) is not None else 0
        print(f"  debug all-class boxes = {debug_count}")
        if debug_count > 0:
            classes = debug_boxes.cls.int().cpu().tolist()
            confs = debug_boxes.conf.cpu().numpy().tolist()
            print(f"  debug classes = {classes}")
            print(f"  debug confs = {['{:.2f}'.format(c) for c in confs]}")
            print(f"  debug boxes = {debug_boxes.xyxy.cpu().numpy().tolist()}")

    if num_boxes > 0:
        boxes = r.boxes.xywh.cpu().numpy()
        # r.boxes.id can be None when the tracker didn't assign IDs; guard against that
        if getattr(r.boxes, 'id', None) is not None:
            track_ids = r.boxes.id.int().cpu().tolist()
        else:
            # Fallback: create frame-local IDs so each detection is tracked at least within this run
            track_ids = [f"f{frame_count}_{i}" for i in range(len(boxes))]

        confidences = r.boxes.conf.cpu().numpy().tolist() if getattr(r.boxes, 'conf', None) is not None else [0.0] * len(track_ids)

        # Filter out detections/tracks below the minimum confidence threshold
        keep_idxs = [i for i, c in enumerate(confidences) if c is not None and float(c) >= MIN_CONFIDENCE]
        if len(keep_idxs) == 0:
            print(f"  Frame {frame_count}: all {len(confidences)} detections filtered by min_conf {MIN_CONFIDENCE}")
            continue

        # apply filtering
        boxes = boxes[keep_idxs] if isinstance(boxes, np.ndarray) else [boxes[i] for i in keep_idxs]
        track_ids = [track_ids[i] for i in keep_idxs]
        confidences = [confidences[i] for i in keep_idxs]
        frames_with_detection += 1

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            # Draw trajectory trail
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=3)
            cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 255), 2)

            # Draw confidence label above box
            label = f"conf: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_origin = (int(x - w/2), int(y - h/2) - 10)
            cv2.rectangle(frame,
                          (label_origin[0], label_origin[1] - label_size[1] - 4),
                          (label_origin[0] + label_size[0] + 4, label_origin[1] + 2),
                          (0, 255, 255), -1)
            cv2.putText(frame, label, (label_origin[0] + 2, label_origin[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Calculate velocity and log data
            prev_x, prev_y = track[-2] if len(track) > 1 else (x, y)
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            
            # Write to CSV
            csv_writer.writerow([frame_count, track_id, f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}", f"{distance:.2f}", f"{conf:.2f}"])
            
            print(f"Ball {track_id} Position: ({int(x)}, {int(y)}) | Vel: {distance:.2f} px/frame | Conf: {conf:.2f}")

    # Write frame to output video
    out.write(frame)

    # Display frame (GUI) with fallback to matplotlib or file save for headless environments
    try:
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except cv2.error:
        try:
            import matplotlib.pyplot as plt
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.pause(0.001)
            plt.clf()
        except Exception:
            # Last-resort fallback: save frames to disk so user can inspect them
            safe_path = os.path.join(os.getcwd(), f"frame_{frame_count:06d}.jpg")
            try:
                cv2.imwrite(safe_path, frame)
            except Exception:
                pass

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
if csv_handle:
    csv_handle.close()

# Calculate and display detection rate
if frame_count > 0:
    detection_rate = (frames_with_detection / frame_count) * 100
    print(f"\n✓ Detection Summary:")
    print(f"  {detection_rate:.1f}% detection rate. Over {frame_count} frames, {frames_with_detection} had a ball tracked")

print(f"\n✓ Output video saved to: {output_video}")
print(f"✓ Tracking data saved to: {csv_file}")