"""Train a YOLOv8 model on the custom pill/capsule dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 pill detector.")
    parser.add_argument("--data", default="data.yaml", help="Path to dataset YAML config.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Base pretrained weights.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader worker threads.")
    parser.add_argument("--device", default="cpu", help="Training device ('cpu' or GPU index like '0').")
    parser.add_argument("--name", default="capsule_yolov8", help="Run output directory name.")
    parser.add_argument("--export-onnx", action="store_true", help="Export best weights to ONNX after training.")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        cache="ram",
        name=args.name,
        patience=15,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best weights saved to: {best_weights}")

    if args.export_onnx and best_weights.exists():
        YOLO(str(best_weights)).export(format="onnx")


if __name__ == "__main__":
    main()