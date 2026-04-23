"""
train.py — Fine-tune YOLOv8n on the auto-labeled ASU mobility dataset.
Run: python train.py [--skip-label]
"""

import argparse
import os
import sys
from pathlib import Path


BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
FRAMES_DIR   = BASE_DIR.parent / "mobility_demo" / "CIS515-Project" / "frames"
DATASET_DIR  = DATA_DIR / "dataset"
MODELS_DIR   = BASE_DIR / "models"
LOGS_DIR     = BASE_DIR / "logs"


def run_auto_label():
    from utils.auto_label import auto_label_frames
    print("\n=== Step 1: Auto-Labeling Dataset ===")
    result = auto_label_frames(
        frames_dir=str(FRAMES_DIR),
        output_base=str(DATASET_DIR),
        model_name="yolov8n.pt",
        conf_threshold=0.35,
        split_ratio=0.8,
    )
    print(f"[Train] Auto-labeling complete: {result['labeled_count']} frames.")
    return result["yaml"]


def run_training(yaml_path: str, epochs: int = 30, imgsz: int = 640):
    from ultralytics import YOLO
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Step 2: Fine-Tuning YOLOv8n ({epochs} epochs) ===")
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        workers=2,
        project=str(MODELS_DIR),
        name="mobility_v1",
        exist_ok=True,
        device="",          # auto: MPS on Mac, CUDA if available, else CPU
        patience=10,        # early stopping
        save=True,
        plots=True,
        verbose=True,
        # Augmentation (helps with small dataset)
        degrees=5.0,
        fliplr=0.5,
        mosaic=0.5,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
    )

    best_model = MODELS_DIR / "mobility_v1" / "weights" / "best.pt"
    if best_model.exists():
        print(f"\n[Train] ✅ Best model saved: {best_model}")
    else:
        best_model = MODELS_DIR / "mobility_v1" / "weights" / "last.pt"
        print(f"\n[Train] ⚠️  Using last checkpoint: {best_model}")

    return str(best_model)


def main():
    parser = argparse.ArgumentParser(description="ASU Mobility Vision — Training Pipeline")
    parser.add_argument("--skip-label", action="store_true", help="Skip auto-labeling (use existing dataset)")
    parser.add_argument("--epochs",     type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--imgsz",      type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--yaml",       type=str, default=None, help="Path to dataset.yaml (if skipping labeling)")
    args = parser.parse_args()

    yaml_path = args.yaml

    if not args.skip_label:
        if not FRAMES_DIR.exists():
            print(f"[Train] ERROR: Frames directory not found at {FRAMES_DIR}")
            print("[Train] Please ensure the dataset is extracted to: mobility_demo/CIS515-Project/frames/")
            sys.exit(1)
        yaml_path = run_auto_label()
    else:
        if yaml_path is None:
            yaml_path = str(DATASET_DIR / "dataset.yaml")
        if not Path(yaml_path).exists():
            print(f"[Train] ERROR: dataset.yaml not found at {yaml_path}")
            print("[Train] Run without --skip-label to generate labels first.")
            sys.exit(1)

    best_model = run_training(yaml_path, epochs=args.epochs, imgsz=args.imgsz)
    print(f"\n🎉 Training complete! Model: {best_model}")
    print(f"   Launch dashboard: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
