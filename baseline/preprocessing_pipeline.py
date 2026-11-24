"""Visualize classical OpenCV preprocessing stages in a 2x2 grid."""

import argparse

import cv2
import numpy as np


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def label_tile(img, title: str, color):
    cv2.putText(img, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def main():
    parser = argparse.ArgumentParser(description="Live 2x2 OpenCV preprocessing visualizer.")
    parser.add_argument("--source", default="http://10.88.76.116:8080/video", help="Stream URL or webcam index.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    print("Preprocessing visualizer running. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            orig = cv2.resize(frame, (400, 300))
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            top = np.hstack((
                label_tile(orig, "1. Original", (0, 220, 0)),
                label_tile(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "2. Grayscale", (0, 220, 220)),
            ))
            bottom = np.hstack((
                label_tile(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), "3. Gaussian Blur", (255, 200, 0)),
                label_tile(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "4. Otsu Binary", (0, 165, 255)),
            ))

            cv2.imshow("Preprocessing Pipeline", np.vstack((top, bottom)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()