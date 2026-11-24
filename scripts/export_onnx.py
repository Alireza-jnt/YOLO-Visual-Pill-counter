"""Export trained PyTorch (.pt) YOLOv8 weights to ONNX format."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 weights to ONNX.")
    parser.add_argument("--weights", default="models/best.pt", help="Path to trained .pt weights.")
    parser.add_argument("--imgsz", type=int, default=640, help="Export input resolution.")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    export_path = YOLO(str(weights_path)).export(format="onnx", imgsz=args.imgsz)
    print(f"Exported ONNX model to: {export_path}")


if __name__ == "__main__":
    main()