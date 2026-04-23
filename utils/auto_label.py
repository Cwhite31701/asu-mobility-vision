"""
Auto-labeling pipeline using pretrained YOLOv8.
Maps COCO classes → person (0) and bike_wheel (1).
Produces YOLO-format .txt label files for fine-tuning.
"""

import os
import shutil
from pathlib import Path
import yaml

# COCO class IDs that map to our two classes
PERSON_IDS = {0}           # person
BIKE_WHEEL_IDS = {1, 3, 4, 8}  # bicycle, motorcycle, skateboard (8 is not standard but kept), bus→skip
# COCO: 0=person,1=bicycle,2=car,3=motorcycle,4=airplane,8=truck — we use 1=bicycle,3=motorcycle

# Corrected COCO IDs
BIKE_WHEEL_IDS = {1, 3}    # bicycle, motorcycle (plus we'll catch skateboard via label name)


def auto_label_frames(
    frames_dir: str,
    output_base: str,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.35,
    split_ratio: float = 0.8,
) -> dict:
    """
    Run YOLOv8 inference on all frames and produce YOLO .txt labels.

    Returns:
        dict with paths to train/val image/label dirs and dataset.yaml path
    """
    from ultralytics import YOLO
    import cv2

    frames_dir = Path(frames_dir)
    output_base = Path(output_base)

    # Setup directories
    train_img = output_base / "images" / "train"
    val_img   = output_base / "images" / "val"
    train_lbl = output_base / "labels" / "train"
    val_lbl   = output_base / "labels" / "val"
    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    print(f"[AutoLabel] Loading {model_name} ...")
    model = YOLO(model_name)

    # Gather all frame files
    image_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    print(f"[AutoLabel] Found {len(image_files)} frames to label.")

    n_train = int(len(image_files) * split_ratio)
    labeled_count = 0

    for i, img_path in enumerate(image_files):
        split = "train" if i < n_train else "val"
        img_dir = train_img if split == "train" else val_img
        lbl_dir = train_lbl if split == "train" else val_lbl

        # Copy image
        dest_img = img_dir / img_path.name
        shutil.copy2(img_path, dest_img)

        # Run inference
        results = model(str(img_path), conf=conf_threshold, verbose=False)[0]

        # Read image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Build label file
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        lines = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            # Map to our class IDs
            if cls_id in PERSON_IDS:
                our_cls = 0
            elif cls_id in BIKE_WHEEL_IDS:
                our_cls = 1
            else:
                # Check by name if available
                name = model.names.get(cls_id, "")
                if name in ("skateboard", "bicycle", "motorcycle"):
                    our_cls = 1
                else:
                    continue  # Skip irrelevant classes

            # Get normalized xywh
            x_center, y_center, bw, bh = box.xywhn[0].tolist()
            conf = box.conf[0].item()
            lines.append(f"{our_cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))
        labeled_count += 1

        if (i + 1) % 20 == 0:
            print(f"[AutoLabel] Processed {i+1}/{len(image_files)} frames...")

    print(f"[AutoLabel] Labeled {labeled_count} frames.")

    # Write dataset.yaml
    yaml_path = output_base / "dataset.yaml"
    dataset_config = {
        "path": str(output_base.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["person", "bike_wheel"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"[AutoLabel] Dataset YAML written to {yaml_path}")
    return {
        "yaml": str(yaml_path),
        "train_images": str(train_img),
        "val_images": str(val_img),
        "train_labels": str(train_lbl),
        "val_labels": str(val_lbl),
        "labeled_count": labeled_count,
    }


if __name__ == "__main__":
    import sys
    frames_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/frames"
    output_base = sys.argv[2] if len(sys.argv) > 2 else "../data/dataset"
    result = auto_label_frames(frames_dir, output_base)
    print("Result:", result)
