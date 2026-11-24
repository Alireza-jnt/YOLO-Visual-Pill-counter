"""Capture raw dataset images from a live mobile camera stream or webcam."""

import argparse
import time
from pathlib import Path

import cv2


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def main():
    parser = argparse.ArgumentParser(description="Capture dataset frames from video stream.")
    parser.add_argument("--source", default="http://10.231.94.57:8080/video", help="Stream URL or webcam index.")
    parser.add_argument("--out", default="raw_images", help="Output directory for captured images.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(out_dir.glob("*.jpg")))

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    print("Press 's' to capture a frame, 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            preview = cv2.resize(frame, (800, 600))
            cv2.putText(
                preview,
                f"Captured: {count}  |  [S] Save  [Q] Quit",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 230, 230),
                2,
            )
            cv2.imshow("Dataset Capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                filepath = out_dir / f"cap_{int(time.time() * 1000)}.jpg"
                cv2.imwrite(str(filepath), frame)
                count += 1
                print(f"Saved [{count}]: {filepath}")
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()