import os
import time
import argparse
from datetime import datetime
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
os.makedirs(CAPTURES_DIR, exist_ok=True)

def find_model():
    candidates = [
        os.path.join(BASE_DIR, "yolov8n.pt"),
        os.path.join(ROOT_DIR, "yolov8n.pt"),
        "yolov8n.pt"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "yolov8n.pt"

def run_tracker(source=0, conf=0.35, show_hud=True):
    model_path = find_model()
    print(f"🧠 Loading YOLOv8 Model from '{model_path}'...")
    model = YOLO(model_path)

    # Ingest video or webcam
    try:
        source_val = int(source)
    except ValueError:
        source_val = source

    print(f"🎥 Opening video source: {source_val}...")
    cap = cv2.VideoCapture(source_val)

    if not cap.isOpened():
        print(f"❌ Error: Unable to open video source '{source}'.")
        if isinstance(source_val, int) and source_val == 0:
            print("💡 Retrying with camera index 1...")
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("❌ No accessible webcam detected.")
                return

    print("\n" + "="*50)
    print("🚀 YOLOv8 REAL-TIME OBJECT TRACKER ACTIVE")
    print("="*50)
    print("Controls:")
    print("  [Q] or [ESC] : Exit Application")
    print("  [S]          : Save Annotated Snapshot to 'captures/'")
    print("  [P]          : Pause / Resume Stream")
    print("  [H]          : Toggle HUD Analytics")
    print("="*50 + "\n")

    prev_time = time.time()
    paused = False
    last_frame = None

    while True:
        if not paused:
            success, frame = cap.read()
            if not success:
                print("End of video stream or lost connection.")
                break
            
            # Run YOLO Tracking
            results = model.track(frame, persist=True, conf=conf, verbose=False)
            annotated_frame = results[0].plot()

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / max((curr_time - prev_time), 0.001)
            prev_time = curr_time

            # Compute Object Statistics
            class_counts = {}
            active_ids = set()
            
            if results[0].boxes is not None:
                boxes = results[0].boxes
                names = results[0].names
                
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = names.get(cls_id, f"Class {cls_id}")
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    
                    if box.id is not None:
                        active_ids.add(int(box.id[0].item()))

            # Draw HUD Overlay
            if show_hud:
                h, w, _ = annotated_frame.shape
                
                # Top Banner (Dark Translucent Bar)
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 25), -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                # Title and Live FPS
                cv2.putText(annotated_frame, "YOLOv8 Tracker Pro", (15, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 2)
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                
                # Active Track IDs
                cv2.putText(annotated_frame, f"Tracked IDs: {len(active_ids)}", (w - 180, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 50), 1)
                
                # Class Summary Breakdown
                summary_str = " | ".join([f"{k}: {v}" for k, v in list(class_counts.items())[:4]])
                if not summary_str:
                    summary_str = "No objects detected"
                cv2.putText(annotated_frame, summary_str, (210, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)

            last_frame = annotated_frame.copy()
        else:
            # Paused State Overlay
            annotated_frame = last_frame.copy()
            cv2.putText(annotated_frame, "PAUSED - Press [P] to Resume", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("CodeAlpha AI - Object Detection & Tracking", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('s') or key == ord('S'):  # Save Snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            capture_filename = os.path.join(CAPTURES_DIR, f"tracking_{timestamp}.jpg")
            cv2.imwrite(capture_filename, annotated_frame)
            print(f"📸 Snapshot saved to: {capture_filename}")
        elif key == ord('p') or key == ord('P'):  # Pause Toggle
            paused = not paused
        elif key == ord('h') or key == ord('H'):  # Toggle HUD
            show_hud = not show_hud

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Object tracking stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Real-time Object Tracker")
    parser.add_argument("--source", default=0, help="Webcam index (0) or path to video file")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    run_tracker(source=args.source, conf=args.conf)