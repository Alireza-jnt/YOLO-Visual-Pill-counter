"""Real-time pill and capsule counter using YOLOv8 / ONNX Runtime."""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_COLORS = {
    0: (0, 220, 0),
    1: (0, 165, 255),
}
DEFAULT_COLOR = (255, 200, 0)


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def draw_detections(frame, boxes, names):
    """Draw bounding boxes and return per-class object counts."""
    counts = {}
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = names[cls_id]

        counts[label] = counts.get(label, 0) + 1
        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return counts


def draw_hud(frame, counts: dict, fps: float):
    """Render count summary and FPS overlay on the top-left corner."""
    total = sum(counts.values())
    details = " | ".join(f"{k}: {v}" for k, v in counts.items()) or "No objects detected"

    cv2.rectangle(frame, (15, 15), (360, 135), (20, 20, 20), -1)
    cv2.rectangle(frame, (15, 15), (360, 135), (0, 220, 0), 2)

    cv2.putText(frame, f"Total: {total}", (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 220, 0), 2)
    cv2.putText(frame, f"FPS:   {fps:.1f}", (28, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2)
    cv2.putText(frame, details, (28, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)


def main():
    parser = argparse.ArgumentParser(description="Real-time pill and capsule counter.")
    parser.add_argument("--source", default="http://10.231.94.222:8080/video", help="Stream URL or webcam index (e.g. 0).")
    parser.add_argument("--model", default="models/best.onnx", help="Path to .onnx or .pt weights.")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--width", type=int, default=960, help="Display frame width.")
    parser.add_argument("--height", type=int, default=720, help="Display frame height.")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    model = YOLO(args.model)
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    print("Stream started. Press 's' to save a snapshot, 'q' to quit.")
    prev_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (args.width, args.height))
            result = model.predict(source=frame, conf=args.conf, iou=args.iou, verbose=False)[0]
            counts = draw_detections(frame, result.boxes, model.names)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_hud(frame, counts, fps)
            cv2.imshow("Pill & Capsule Counter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                out_file = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(out_file, frame)
                print(f"Saved {out_file}")
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()