"""Classical OpenCV Watershed baseline with red-channel segmentation and live tuning."""

import argparse

import cv2
import numpy as np


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def segment_and_count(frame, min_bright: int, sep_ratio: float, min_area: int):
    """Segment touching two-tone capsules using the red channel and local maxima Watershed."""
    output = frame.copy()

    # Use the red channel so both white and dark-red capsule halves stay bright against dark surfaces
    _, _, red_channel = cv2.split(frame)
    blurred = cv2.GaussianBlur(red_channel, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, min_bright, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    # Local maxima isolates individual capsule peaks even when aligned side-by-side
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    local_max = cv2.dilate(dist, peak_kernel)
    peaks = (dist >= local_max - 0.02) & (dist > sep_ratio)
    sure_fg = cv2.morphologyEx(np.uint8(peaks) * 255, cv2.MORPH_CLOSE, kernel, iterations=1)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(frame, markers)

    count = 0
    for label in np.unique(markers):
        if label <= 1:
            continue

        mask = np.zeros(red_channel.shape, dtype=np.uint8)
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
            continue

        count += 1
        (x, y), _ = cv2.minEnclosingCircle(largest)
        cv2.drawContours(output, [largest], -1, (0, 220, 0), 2)
        cv2.circle(output, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.putText(output, f"#{count}", (int(x) - 12, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 230), 2)

    cv2.rectangle(output, (10, 10), (260, 56), (20, 20, 20), -1)
    cv2.putText(output, f"Count: {count}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
    return output, opening, sure_fg


def main():
    parser = argparse.ArgumentParser(description="Classical Watershed pill counter baseline.")
    parser.add_argument("--source", default="http://10.88.76.116:8080/video", help="Stream URL or webcam index.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    cv2.namedWindow("Controls")
    cv2.resizeWindow("Controls", 420, 180)
    cv2.createTrackbar("Min Brightness", "Controls", 75, 255, lambda _: None)
    cv2.createTrackbar("Separation %", "Controls", 55, 90, lambda _: None)
    cv2.createTrackbar("Min Area", "Controls", 120, 1000, lambda _: None)

    print("Watershed baseline running. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            min_bright = cv2.getTrackbarPos("Min Brightness", "Controls")
            sep_ratio = max(cv2.getTrackbarPos("Separation %", "Controls"), 20) / 100.0
            min_area = cv2.getTrackbarPos("Min Area", "Controls")

            result, mask, centers = segment_and_count(frame, min_bright, sep_ratio, min_area)
            cv2.imshow("Watershed Counter", result)
            cv2.imshow("Binary Mask", mask)
            cv2.imshow("Detected Centers", centers)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()