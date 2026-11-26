"""Real-time pill and capsule counter using YOLOv8 / ONNX Runtime."""

import argparse
import threading
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


class FrameGrabber:
    """Continuously reads frames on a background thread to eliminate I/O lag."""

    def __init__(self, source, width: int, height: int):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        # Request lower resolution (effective for local webcams; IP streams set this in the camera app)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.lock = threading.Lock()
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def extract_detections(boxes, names):
    """Unpack boxes to plain tuples once so they can be reused across skipped frames."""
    detections = []
    counts = {}

    if boxes is None or len(boxes) == 0:
        return detections, counts

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
        label = names[cls_id]
        counts[label] = counts.get(label, 0) + 1
        detections.append((x1, y1, x2, y2, float(conf), cls_id, label))

    return detections, counts


def draw_detections(frame, detections):
    """Draw cached bounding boxes onto the current frame."""
    for x1, y1, x2, y2, conf, cls_id, label in detections:
        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )


def draw_hud(frame, counts: dict, fps: float):
    """Render compact count summary and FPS overlay."""
    total = sum(counts.values())
    details = " | ".join(f"{k}: {v}" for k, v in counts.items()) or "No objects detected"

    cv2.rectangle(frame, (10, 10), (300, 105), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (300, 105), (0, 220, 0), 2)

    cv2.putText(frame, f"Total: {total}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
    cv2.putText(frame, f"FPS:   {fps:.1f}", (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
    cv2.putText(frame, details, (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)


def main():
    parser = argparse.ArgumentParser(description="Real-time pill and capsule counter.")
    parser.add_argument("--source", default="http://10.231.94.222:8080/video", help="Stream URL or webcam index.")
    parser.add_argument("--model", default="models/best.onnx", help="Path to .onnx, .pt, or OpenVINO model.")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--width", type=int, default=640, help="Display/target frame width.")
    parser.add_argument("--height", type=int, default=480, help="Display/target frame height.")
    parser.add_argument("--imgsz", type=int, default=640, help="Model inference size (must match static ONNX size).")
    parser.add_argument("--skip", type=int, default=1, help="Run inference every N frames (e.g. 2 = half the load).")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    model = YOLO(args.model, task="detect")
    grabber = FrameGrabber(parse_source(args.source), args.width, args.height)

    print("Stream started. Press 's' to save a snapshot, 'q' to quit.")
    prev_time = time.perf_counter()
    fps = 0.0
    frame_idx = 0
    last_detections = []
    last_counts = {}

    try:
        while grabber.running:
            ret, frame = grabber.read()
            if not ret:
                time.sleep(0.005)
                continue

            # Skip resizing if the stream already matches target dimensions
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

            if frame_idx % max(args.skip, 1) == 0:
                result = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    verbose=False,
                )[0]
                last_detections, last_counts = extract_detections(result.boxes, model.names)
            frame_idx += 1

            draw_detections(frame, last_detections)

            now = time.perf_counter()
            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = 0.85 * fps + 0.15 * inst_fps if fps > 0 else inst_fps  # Smoothed FPS readout
            prev_time = now

            draw_hud(frame, last_counts, fps)
            cv2.imshow("Pill & Capsule Counter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                out_file = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(out_file, frame)
                print(f"Saved {out_file}")
            elif key == ord("q"):
                break
    finally:
        grabber.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()