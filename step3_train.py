"""
STEP 3: YOLOv8 Training — Optimized for Fire Detection
========================================================
Run AFTER step2_augmentation.py
Trains YOLOv8s with fire-optimized hyperparameters
"""

import os
import yaml
import torch
import shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # Model
    "model":        "yolov8s.pt",   # s = small: best speed/accuracy for deadline
                                     # Options: yolov8n (fastest) | yolov8s | yolov8m (best accuracy)
    "data_yaml":    "dataset/fire.yaml",
    "project":      "runs/fire_detection",
    "run_name":     f"train_{datetime.now().strftime('%Y%m%d_%H%M')}",

    # Training
    "epochs":       100,
    "imgsz":        640,
    "batch":        -1,             # -1 = auto batch size (uses ~60% GPU VRAM)
    "workers":      4,              # data loading workers (reduce to 2 on Colab)

    # Optimizer
    "optimizer":    "AdamW",        # AdamW > SGD for smaller datasets
    "lr0":          0.001,          # initial learning rate
    "lrf":          0.01,           # final lr = lr0 * lrf
    "momentum":     0.937,
    "weight_decay": 0.0005,
    "warmup_epochs":3.0,

    # Augmentation (YOLOv8 built-in — ON TOP of our Albumentations)
    "mosaic":       1.0,            # YOLOv8 mosaic augmentation probability
    "mixup":        0.15,           # mixup augmentation
    "copy_paste":   0.1,            # copy-paste augmentation

    # Loss weights — tuned for fire detection
    "box":          7.5,            # bounding box loss weight
    "cls":          0.5,            # classification loss
    "dfl":          1.5,            # distribution focal loss

    # Fire-specific settings
    "conf":         0.25,           # confidence threshold for inference
    "iou":          0.45,           # NMS IoU threshold
    "max_det":      50,             # max detections per image

    # Hardware
    "device":       "0" if torch.cuda.is_available() else "cpu",
    "amp":          True,           # automatic mixed precision (faster training)

    # Early stopping
    "patience":     20,             # stop if no improvement for 20 epochs

    # Save
    "save_period":  10,             # save checkpoint every N epochs
}


# ─────────────────────────────────────────────
# PRE-TRAINING CHECKS
# ─────────────────────────────────────────────

def pre_training_checks(cfg: dict):
    """Verify everything is in place before training."""
    print("🔍 Pre-training checks...")

    # Check YAML
    yaml_path = Path(cfg["data_yaml"])
    assert yaml_path.exists(), f"❌ data yaml not found: {yaml_path}"

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    base = Path(data.get("path", "."))
    train_dir = base / data.get("train", "train/images")
    val_dir   = base / data.get("val",   "val/images")
    test_dir  = base / data.get("test",  "test/images")

    assert train_dir.exists(), f"❌ Train dir not found: {train_dir}"
    assert val_dir.exists(),   f"❌ Val dir not found: {val_dir}"

    n_train = len(list(train_dir.glob("*")))
    n_val   = len(list(val_dir.glob("*")))
    n_test  = len(list(test_dir.glob("*"))) if test_dir.exists() else 0

    print(f"  ✅ Train images:  {n_train}")
    print(f"  ✅ Val images:    {n_val}")
    print(f"  ✅ Test images:   {n_test}")
    print(f"  ✅ Classes:       {data.get('names')}")
    print(f"  ✅ Device:        {cfg['device']}")
    print(f"  ✅ GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✅ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    return True


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train(cfg: dict):
    """Launch YOLOv8 training with fire-optimized settings."""

    print("\n🔥 Starting YOLOv8 Fire Detection Training")
    print("=" * 50)

    model = YOLO(cfg["model"])

    results = model.train(
        data          = cfg["data_yaml"],
        epochs        = cfg["epochs"],
        imgsz         = cfg["imgsz"],
        batch         = cfg["batch"],
        workers       = cfg["workers"],

        # Optimizer
        optimizer     = cfg["optimizer"],
        lr0           = cfg["lr0"],
        lrf           = cfg["lrf"],
        momentum      = cfg["momentum"],
        weight_decay  = cfg["weight_decay"],
        warmup_epochs = cfg["warmup_epochs"],

        # Augmentation
        mosaic        = cfg["mosaic"],
        mixup         = cfg["mixup"],
        copy_paste    = cfg["copy_paste"],

        # Loss weights
        box           = cfg["box"],
        cls           = cfg["cls"],
        dfl           = cfg["dfl"],

        # Inference settings
        conf          = cfg["conf"],
        iou           = cfg["iou"],
        max_det       = cfg["max_det"],

        # Hardware
        device        = cfg["device"],
        amp           = cfg["amp"],

        # Training control
        patience      = cfg["patience"],
        save_period   = cfg["save_period"],

        # Output
        project       = cfg["project"],
        name          = cfg["run_name"],
        exist_ok      = True,

        # Extras
        verbose       = True,
        plots         = True,   # save training plots
        val           = True,   # validate after each epoch
    )

    return results


# ─────────────────────────────────────────────
# POST-TRAINING: FIND BEST WEIGHTS
# ─────────────────────────────────────────────

def get_best_weights(cfg: dict) -> str:
    """Return path to best.pt from the training run."""
    best_pt = Path(cfg["project"]) / cfg["run_name"] / "weights" / "best.pt"
    if not best_pt.exists():
        # fallback: find latest run
        runs = sorted(Path(cfg["project"]).glob("*/weights/best.pt"))
        if runs:
            best_pt = runs[-1]

    assert best_pt.exists(), f"❌ best.pt not found at {best_pt}"
    print(f"✅ Best weights: {best_pt}")
    return str(best_pt)


# ─────────────────────────────────────────────
# QUICK VALIDATION AFTER TRAINING
# ─────────────────────────────────────────────

def validate_model(weights_path: str, data_yaml: str, split: str = "val"):
    """
    Run validation on val or test split.
    split: 'val' for validation set, 'test' for BowFire test set
    """
    print(f"\n📊 Validating on {split} split...")
    model = YOLO(weights_path)

    metrics = model.val(
        data    = data_yaml,
        split   = split,
        imgsz   = 640,
        conf    = 0.25,
        iou     = 0.45,
        plots   = True,
        verbose = True,
    )

    print("\n📈 Results:")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")

    # Per-class metrics
    if hasattr(metrics.box, 'ap_class_index'):
        print("\n  Per-class mAP@0.5:")
        class_names = ["fire", "smoke"]
        for i, ap in enumerate(metrics.box.ap50):
            name = class_names[i] if i < len(class_names) else f"class_{i}"
            print(f"    {name}: {ap:.4f}")

    return metrics


# ─────────────────────────────────────────────
# EXPORT MODEL (for deployment on ESP32 companion)
# ─────────────────────────────────────────────

def export_model(weights_path: str, format: str = "onnx"):
    """
    Export to ONNX for deployment / faster inference.
    format options: 'onnx', 'tflite', 'ncnn', 'torchscript'
    """
    print(f"\n📦 Exporting model to {format}...")
    model = YOLO(weights_path)
    model.export(
        format    = format,
        imgsz     = 640,
        simplify  = True,   # ONNX simplification
        opset     = 12,     # ONNX opset version
    )
    print(f"✅ Model exported to {format}")


# ─────────────────────────────────────────────
# RESUME TRAINING (if interrupted)
# ─────────────────────────────────────────────

def resume_training(cfg: dict):
    """Resume from last checkpoint if training was interrupted."""
    last_pt = Path(cfg["project"]) / cfg["run_name"] / "weights" / "last.pt"
    if last_pt.exists():
        print(f"🔄 Resuming from: {last_pt}")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        print("⚠️  No checkpoint found, starting fresh training...")
        train(cfg)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    cfg = CONFIG

    print("🔥 Fire Detection — Step 3: YOLOv8 Training")
    print("=" * 50)

    # Pre-flight checks
    pre_training_checks(cfg)

    # Train
    print("\n🚀 Launching training...")
    results = train(cfg)

    # Get best weights
    best_weights = get_best_weights(cfg)

    # Validate on val set
    print("\n── Validation Set Performance ──")
    val_metrics = validate_model(best_weights, cfg["data_yaml"], split="val")

    # Validate on BowFire test set
    print("\n── BowFire Test Set Performance ──")
    test_metrics = validate_model(best_weights, cfg["data_yaml"], split="test")

    # Save metrics summary
    summary = {
        "best_weights": best_weights,
        "val": {
            "mAP50":      float(val_metrics.box.map50),
            "mAP50_95":   float(val_metrics.box.map),
            "precision":  float(val_metrics.box.mp),
            "recall":     float(val_metrics.box.mr),
        },
        "test_bowfire": {
            "mAP50":      float(test_metrics.box.map50),
            "mAP50_95":   float(test_metrics.box.map),
            "precision":  float(test_metrics.box.mp),
            "recall":     float(test_metrics.box.mr),
        }
    }

    import json
    summary_path = Path(cfg["project"]) / cfg["run_name"] / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Metrics saved: {summary_path}")

    # Export to ONNX
    export_model(best_weights, format="onnx")

    print("\n✅ Step 3 Complete! Run step4_evaluate.py next.")
    print(f"   Best weights: {best_weights}")

    return best_weights


if __name__ == "__main__":
    main()
