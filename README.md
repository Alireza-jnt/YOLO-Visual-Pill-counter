# YOLO Visual Pill Counter

Real-time tabletop pill and capsule counting from a smartphone camera video stream, combining a custom-trained [Ultralytics YOLOv8n](https://docs.ultralytics.com/models/yolov8) detector exported to [ONNX Runtime](https://docs.ultralytics.com/modes/export) with a classical [OpenCV Watershed](https://docs.opencv.org/3.4.1/d2/dbd/tutorial_distance_transform.html) baseline for comparison.

The pipeline runs fully offline over a local Wi-Fi hotspot or webcam and achieves real-time CPU inference without requiring a dedicated GPU.

## Project Structure

```
YOLO-Visual-Pill-Counter/
├── main.py                         # Real-time YOLOv8 / ONNX counter entry point
├── data.yaml                       # YOLO dataset configuration
├── requirements.txt                # Python dependencies
├── models/
│   ├── best.onnx                   # Exported ONNX model for fast CPU inference
│   └── best.pt                     # Trained PyTorch checkpoint
├── scripts/
│   ├── capture_dataset.py          # Capture training frames from mobile stream
│   ├── prepare_dataset.py          # 80/20 train/val split and data.yaml generator
│   ├── train.py                    # YOLOv8n training script (CPU-friendly)
│   └── export_onnx.py              # PyTorch-to-ONNX export utility
├── baseline/
│   ├── preprocessing_pipeline.py   # 2x2 OpenCV preprocessing visualizer
│   └── watershed_counter.py        # Classical red-channel + local-maxima Watershed baseline
└── assets/
    ├── results.png                 # Training loss and mAP curves
    └── confusion_matrix.png        # Validation confusion matrix

```

## Quick Start

### 1. Install Dependencies

```
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

```

### 2. Stream Your Phone Camera

Start a local stream using [IP Webcam](https://www.geeksforgeeks.org/python/connect-your-android-phone-camera-to-opencv-python/) (Android) or [DroidCam](https://droidcam.app/) (Android/iOS) over a shared Wi-Fi network or mobile hotspot. Recommended stream resolution is `640x480` or `800x600` to keep latency low.

### 3. Run Real-Time Counting

```
# Using a smartphone HTTP stream
python main.py --source http://<PHONE_IP>:8080/video

# Or test immediately with a local webcam
python main.py --source 0

```

**Controls:** Press `s` to save a snapshot of the annotated frame, or `q` to quit.

**Optional CLI Flags:**

| Flag | Default | Description | 
| ----- | ----- | ----- | 
| `--source` | stream URL | HTTP/RTSP stream URL or webcam index (`0`) | 
| `--model` | `models/best.onnx` | Path to `.onnx` or `.pt` model weights | 
| `--conf` | `0.50` | Minimum detection confidence threshold | 
| `--iou` | `0.45` | NMS IoU threshold to suppress duplicate boxes | 

## Classical Baseline vs. Deep Learning

Before training YOLOv8, we built a classical segmentation baseline (`baseline/watershed_counter.py`) using red-channel extraction, morphological opening, distance transform, and local-maxima Watershed.

* **Why classical segmentation struggles:** Two-tone glossy capsules (white/maroon) reflect overhead lighting and merge into connected blobs when touching, requiring manual threshold and peak-ratio tuning per lighting condition.

* **Why YOLOv8n + ONNX works better:** Learns full capsule geometry and texture end-to-end, stays stable under shadows and glare, separates densely packed capsules via NMS, and runs smoothly on multi-threaded CPUs (tested on [AMD Ryzen 7 7730U](https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-7-7730u.html) with 16 GB RAM).

To run the classical baseline:

```
python baseline/watershed_counter.py --source http://<PHONE_IP>:8080/video

```

## Training From Scratch

1. **Capture frames** across varied spacing, touching clusters, and lighting angles:

   ```
   python scripts/capture_dataset.py --source http://<PHONE_IP>:8080/video --out raw_images
   
   ```

2. **Annotate** bounding boxes in YOLO format using [LabelImg](https://github.com/HumanSignal/labelimg).

3. **Split dataset** into 80% train / 20% validation and generate `data.yaml`:

   ```
   python scripts/prepare_dataset.py --source raw_images --out dataset
   
   ```

4. **Train and export** to ONNX:

   ```
   python scripts/train.py --data data.yaml --epochs 50 --batch 8 --device cpu --export-onnx
   
   ```
## Evaluation Results

Validation curves for the trained YOLOv8n capsule detector (**mAP@50 = 0.995**, **F1 = 1.00 at conf 0.565**):

| F1-Confidence Curve | Precision-Recall Curve |
| :---: | :---: |
| [assets/BoxF1_curve.png] | [assets/BoxPR_curve.png] |

## Contributors

* **Alireza Jannati** — DevOps, Python Backend, IoT Streaming & Model Pipeline

* **Amirreza Jafari** — Frontend Development & Monitoring Interface

* **Ali Abrehdari** — Frontend Development & Stream Client Integration