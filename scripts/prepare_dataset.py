"""Split annotated raw images into YOLO train/val directories and write data.yaml."""

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_class_names(source_dir: Path) -> list[str]:
    classes_file = source_dir / "classes.txt"
    if classes_file.exists():
        names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names
    return ["capsule"]


def write_dataset_yaml(yaml_path: Path, dataset_dir: Path, class_names: list[str]):
    lines = [
        f"path: ./{dataset_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO train/val split from LabelImg annotations.")
    parser.add_argument("--source", default="raw_images", help="Directory containing raw images and .txt labels.")
    parser.add_argument("--out", default="dataset", help="Output dataset directory.")
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio (default: 0.8).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    args = parser.parse_args()

    source_dir = Path(args.source)
    dataset_dir = Path(args.out)

    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = [p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.with_suffix(".txt").exists()]
    random.Random(args.seed).shuffle(images)

    cut = int(len(images) * args.split)
    splits = {"train": images[:cut], "val": images[cut:]}

    for split_name, files in splits.items():
        for img_path in files:
            lbl_path = img_path.with_suffix(".txt")
            shutil.copy2(img_path, dataset_dir / "images" / split_name / img_path.name)
            shutil.copy2(lbl_path, dataset_dir / "labels" / split_name / lbl_path.name)

    class_names = load_class_names(source_dir)
    write_dataset_yaml(Path("data.yaml"), dataset_dir, class_names)
    print(f"Prepared {len(splits['train'])} train and {len(splits['val'])} val samples. Updated data.yaml.")


if __name__ == "__main__":
    main()