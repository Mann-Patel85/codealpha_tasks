import os
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

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

model = YOLO(find_model())

def detect_objects_image(image, conf_threshold, iou_threshold):
    if image is None:
        return None, "⚠️ Please upload an image."

    results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    annotated_img = results[0].plot()

    # Calculate statistics
    boxes = results[0].boxes
    names = results[0].names
    counts = {}

    for box in boxes:
        cls_id = int(box.cls[0].item())
        cls_name = names.get(cls_id, f"Class {cls_id}")
        counts[cls_name] = counts.get(cls_name, 0) + 1

    summary_lines = [f"- **{k}:** {v}" for k, v in counts.items()]
    total_objects = sum(counts.values())

    stats_report = (
        f"### 📊 Detection Summary\n"
        f"- **Total Objects Detected:** {total_objects}\n\n"
        f"**Breakdown by Class:**\n" + ("\n".join(summary_lines) if summary_lines else "No objects recognized.")
    )
    return annotated_img, stats_report

custom_css = """
.vision-title {text-align: center; color: #10b981; font-weight: 800; margin-bottom: 2px;}
.vision-subtitle {text-align: center; color: #64748b; font-size: 14px; margin-bottom: 20px;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"), css=custom_css, title="YOLOv8 Vision Studio") as demo:
    gr.Markdown("# 👁️ YOLOv8 Computer Vision Studio", elem_classes="vision-title")
    gr.Markdown("Real-time Object Detection and Multi-Class Localization with YOLOv8 Nano.", elem_classes="vision-subtitle")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload Image or Snapshot")
            
            with gr.Row():
                conf_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.35, step=0.05, label="Confidence Threshold")
                iou_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.45, step=0.05, label="NMS IoU Threshold")
            
            detect_btn = gr.Button("🔍 Detect Objects", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Detections & Bounding Boxes")
            output_stats = gr.Markdown("Upload an image and click **Detect Objects**.")

    detect_btn.click(
        fn=detect_objects_image,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=[output_image, output_stats]
    )

    gr.Markdown("---")
    gr.Markdown("<div style='text-align: center; color: #94a3b8; font-size: 12px;'>Powered by Ultralytics YOLOv8 & OpenCV | Mann Patel (CodeAlpha 2025)</div>")

if __name__ == "__main__":
    demo.launch()
