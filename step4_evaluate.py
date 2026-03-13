"""
STEP 4: Evaluation, Metrics & Visualizations
=============================================
Run AFTER step3_train.py
Generates all figures needed for IEEE paper:
  - mAP table, PR curve, confusion matrix, F1 curve
  - YOLOv8 alone vs YOLOv8+TVL comparison
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import cv2
import torch


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "weights":        "runs/detect/runs/fire_detection/train_20260312_1229/weights/best.pt",  # update path
    "data_yaml":      "dataset/fire.yaml",
    "test_images":    "dataset/test/images",
    "test_labels":    "dataset/test/labels",
    "output_dir":     "evaluation_results",
    "class_names":    ["fire", "smoke"],
    "conf_threshold": 0.25,
    "iou_threshold":  0.45,
    "image_size":     640,
}


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model(weights_path: str) -> YOLO:
    assert Path(weights_path).exists(), f"❌ Weights not found: {weights_path}"
    model = YOLO(weights_path)
    print(f"✅ Loaded model: {weights_path}")
    return model


# ─────────────────────────────────────────────
# RUN INFERENCE ON TEST SET
# ─────────────────────────────────────────────

def run_inference_on_test_set(model: YOLO, test_images_dir: str, conf: float, iou: float, imgsz: int):
    """
    Run inference on all test images.
    Returns dict: {image_stem: [{"bbox": [x1,y1,x2,y2], "conf": float, "cls": int}]}
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in Path(test_images_dir).iterdir() if f.suffix.lower() in image_extensions]

    predictions = {}

    for img_file in images:
        results = model(str(img_file), conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "bbox": box.xyxy[0].tolist(),
                    "conf": float(box.conf[0]),
                    "cls":  int(box.cls[0]),
                })
        predictions[img_file.stem] = detections

    print(f"✅ Inference complete on {len(images)} images")
    return predictions


# ─────────────────────────────────────────────
# LOAD GROUND TRUTH
# ─────────────────────────────────────────────

def load_ground_truth(labels_dir: str, images_dir: str, class_names: list):
    """
    Load YOLO format ground truth labels.
    Returns dict: {image_stem: [{"bbox": [x1,y1,x2,y2], "cls": int}]}
    """
    ground_truth = {}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for lbl_file in Path(labels_dir).glob("*.txt"):
        stem = lbl_file.stem
        # Find matching image to get dimensions
        img_path = None
        for ext in image_extensions:
            candidate = Path(images_dir) / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        detections = []
        for line in lbl_file.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, bw, bh = [float(x) for x in parts[1:]]
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            detections.append({"bbox": [x1, y1, x2, y2], "cls": cls})

        ground_truth[stem] = detections

    return ground_truth


# ─────────────────────────────────────────────
# IoU CALCULATION
# ─────────────────────────────────────────────

def compute_iou(box1, box2) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


# ─────────────────────────────────────────────
# PRECISION-RECALL CALCULATION
# ─────────────────────────────────────────────

def compute_precision_recall(predictions: dict, ground_truth: dict, iou_threshold: float = 0.5, class_id: int = 0):
    """
    Compute precision-recall curve for a given class.
    Returns sorted arrays of precision and recall values.
    """
    all_detections = []  # (confidence, is_tp)
    total_gt = 0

    for stem, gt_boxes in ground_truth.items():
        gt_for_class = [g for g in gt_boxes if g["cls"] == class_id]
        total_gt += len(gt_for_class)
        matched = [False] * len(gt_for_class)

        preds = sorted(
            [p for p in predictions.get(stem, []) if p["cls"] == class_id],
            key=lambda x: -x["conf"]
        )

        for pred in preds:
            best_iou = 0
            best_idx = -1
            for i, gt in enumerate(gt_for_class):
                if not matched[i]:
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

            is_tp = best_iou >= iou_threshold and best_idx >= 0
            if is_tp:
                matched[best_idx] = True
            all_detections.append((pred["conf"], int(is_tp)))

    if not all_detections:
        return np.array([1.0]), np.array([0.0]), 0.0

    all_detections.sort(key=lambda x: -x[0])
    confs = np.array([d[0] for d in all_detections])
    tps   = np.array([d[1] for d in all_detections])

    tp_cumsum = np.cumsum(tps)
    fp_cumsum = np.cumsum(1 - tps)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls    = tp_cumsum / (total_gt + 1e-6)

    # Add sentinel values
    precisions = np.concatenate([[1.0], precisions])
    recalls    = np.concatenate([[0.0], recalls])

    # Compute AP (area under PR curve) using trapezoidal rule
    ap = np.trapz(precisions, recalls)

    return precisions, recalls, ap


# ─────────────────────────────────────────────
# FULL METRICS COMPUTATION
# ─────────────────────────────────────────────

def compute_all_metrics(predictions: dict, ground_truth: dict, class_names: list, iou_threshold: float = 0.5):
    """Compute mAP, precision, recall, F1 for all classes."""
    metrics = {}
    all_aps = []

    for class_id, class_name in enumerate(class_names):
        precs, recs, ap = compute_precision_recall(predictions, ground_truth, iou_threshold, class_id)

        # Find best F1 threshold
        f1_scores = 2 * precs * recs / (precs + recs + 1e-6)
        best_f1_idx = np.argmax(f1_scores)

        metrics[class_name] = {
            "precisions": precs,
            "recalls":    recs,
            "ap":         ap,
            "best_precision": precs[best_f1_idx],
            "best_recall":    recs[best_f1_idx],
            "best_f1":        f1_scores[best_f1_idx],
        }
        all_aps.append(ap)

    metrics["mAP"] = np.mean(all_aps)
    return metrics


# ─────────────────────────────────────────────
# PLOT: PR CURVE
# ─────────────────────────────────────────────

def plot_pr_curve(metrics: dict, class_names: list, output_dir: str, title: str = "Precision-Recall Curve"):
    """Generate publication-quality PR curve for the paper."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = ["#FF4500", "#808080", "#2196F3", "#4CAF50"]  # fire=orange, smoke=gray
    linestyles = ["-", "--", "-.", ":"]

    for idx, class_name in enumerate(class_names):
        if class_name not in metrics:
            continue
        m   = metrics[class_name]
        ap  = m["ap"]
        prec = m["precisions"]
        rec  = m["recalls"]

        ax.plot(rec, prec,
                color=colors[idx % len(colors)],
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2.0,
                label=f"{class_name.capitalize()} (AP = {ap:.3f})")
        ax.fill_between(rec, prec, alpha=0.1, color=colors[idx % len(colors)])

    mAP = metrics.get("mAP", 0)
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(f"{title}\nmAP@0.5 = {mAP:.3f}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    save_path = Path(output_dir) / "pr_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ PR curve saved: {save_path}")


# ─────────────────────────────────────────────
# PLOT: CONFUSION MATRIX
# ─────────────────────────────────────────────

def compute_and_plot_confusion_matrix(
    predictions: dict,
    ground_truth: dict,
    class_names: list,
    output_dir: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
):
    """Generate normalized confusion matrix."""
    n_classes = len(class_names) + 1  # +1 for background/FP
    labels_ext = class_names + ["background"]

    cm = np.zeros((n_classes, n_classes), dtype=int)

    for stem, gt_boxes in ground_truth.items():
        matched_gt = [False] * len(gt_boxes)
        preds = [p for p in predictions.get(stem, []) if p["conf"] >= conf_threshold]

        for pred in preds:
            best_iou = 0
            best_gt_idx = -1
            for i, gt in enumerate(gt_boxes):
                if not matched_gt[i]:
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

            pred_cls = pred["cls"]

            if best_iou >= iou_threshold:
                gt_cls = gt_boxes[best_gt_idx]["cls"]
                cm[gt_cls][pred_cls] += 1
                matched_gt[best_gt_idx] = True
            else:
                # False positive → background row
                cm[n_classes - 1][pred_cls] += 1

        # Missed ground truths (false negatives)
        for i, gt in enumerate(gt_boxes):
            if not matched_gt[i]:
                cm[gt["cls"]][n_classes - 1] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm_norm / row_sums, 0)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=labels_ext,
        yticklabels=labels_ext,
        ax=ax,
        linewidths=0.5,
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Normalized Confusion Matrix\n(YOLOv8s Fire Detection)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved: {save_path}")
    return cm


# ─────────────────────────────────────────────
# PLOT: F1-CONFIDENCE CURVE
# ─────────────────────────────────────────────

def plot_f1_confidence_curve(model: YOLO, data_yaml: str, output_dir: str):
    """
    Plot F1 vs confidence threshold — helps choose optimal conf for deployment.
    """
    conf_range = np.arange(0.1, 0.95, 0.05)
    f1_scores = []

    for conf in conf_range:
        metrics = model.val(data=data_yaml, conf=float(conf), iou=0.45, verbose=False, plots=False)
        p = metrics.box.mp
        r = metrics.box.mr
        f1 = 2 * p * r / (p + r + 1e-6)
        f1_scores.append(f1)

    best_idx  = np.argmax(f1_scores)
    best_conf = conf_range[best_idx]
    best_f1   = f1_scores[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(conf_range, f1_scores, "b-o", linewidth=2, markersize=4)
    ax.axvline(x=best_conf, color="red", linestyle="--", label=f"Best conf={best_conf:.2f} (F1={best_f1:.3f})")
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("F1-Confidence Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = Path(output_dir) / "f1_confidence_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ F1-Confidence curve saved: {save_path}")
    print(f"   → Optimal confidence threshold: {best_conf:.2f} (F1 = {best_f1:.3f})")
    return best_conf


# ─────────────────────────────────────────────
# PRINT METRICS TABLE (IEEE FORMAT)
# ─────────────────────────────────────────────

def print_metrics_table(metrics_val: dict, metrics_test: dict, class_names: list):
    """Print IEEE-style results table."""
    print("\n" + "=" * 65)
    print("📊 RESULTS TABLE (for IEEE paper)")
    print("=" * 65)
    print(f"{'Dataset':<25} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10} {'mAP@0.5:0.95':>13}")
    print("-" * 65)

    for label, m in [("Multi-Scale (val)", metrics_val), ("BowFire (test)", metrics_test)]:
        p  = m.get("mp", m.get("best_precision", 0))
        r  = m.get("mr", m.get("best_recall", 0))
        m50 = m.get("map50", m.get("mAP", 0))
        m5095 = m.get("map", 0)
        print(f"{label:<25} {p:>10.4f} {r:>10.4f} {m50:>10.4f} {m5095:>13.4f}")

    print("=" * 65)

    # Per-class
    print("\nPer-Class mAP@0.5:")
    for cls in class_names:
        if cls in metrics_val:
            v_ap = metrics_val[cls].get("ap", 0)
            t_ap = metrics_test.get(cls, {}).get("ap", 0)
            print(f"  {cls:<10}: val={v_ap:.4f} | test={t_ap:.4f}")


# ─────────────────────────────────────────────
# SAVE METRICS TO JSON
# ─────────────────────────────────────────────

def save_metrics_json(metrics_val: dict, metrics_test: dict, output_dir: str):
    """Save all metrics to JSON for the paper and step5 (TVL comparison)."""
    def serialize(m):
        out = {}
        for k, v in m.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, dict):
                out[k] = serialize(v)
            else:
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = v
        return out

    data = {
        "val":  serialize(metrics_val),
        "test": serialize(metrics_test),
    }
    save_path = Path(output_dir) / "metrics.json"
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Metrics saved: {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    cfg = CONFIG
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔥 Fire Detection — Step 4: Evaluation & Metrics")
    print("=" * 50)

    # Load model
    model = load_model(cfg["weights"])

    # Run inference on test set
    print("\n🔍 Running inference on test set (BowFire)...")
    predictions = run_inference_on_test_set(
        model, cfg["test_images"],
        cfg["conf_threshold"], cfg["iou_threshold"], cfg["image_size"]
    )

    # Load ground truth
    print("\n📂 Loading ground truth labels...")
    ground_truth = load_ground_truth(
        cfg["test_labels"], cfg["test_images"], cfg["class_names"]
    )

    # Compute metrics
    print("\n📊 Computing metrics...")
    metrics_test = compute_all_metrics(predictions, ground_truth, cfg["class_names"])

    # Get val metrics from ultralytics (more accurate)
    print("\n📊 Running official val on validation set...")
    val_results = model.val(data=cfg["data_yaml"], split="val", verbose=False)
    metrics_val = {
        "mp":    float(val_results.box.mp),
        "mr":    float(val_results.box.mr),
        "map50": float(val_results.box.map50),
        "map":   float(val_results.box.map),
    }

    # Plots
    print("\n📈 Generating plots for paper...")
    plot_pr_curve(metrics_test, cfg["class_names"], str(out_dir))
    compute_and_plot_confusion_matrix(
        predictions, ground_truth, cfg["class_names"], str(out_dir)
    )

    # F1-Confidence curve (helps choose deployment threshold)
    print("\n📈 Computing F1-Confidence curve...")
    best_conf = plot_f1_confidence_curve(model, cfg["data_yaml"], str(out_dir))

    # Print table
    print_metrics_table(metrics_val, metrics_test, cfg["class_names"])

    # Save metrics JSON (used by step5)
    save_metrics_json(metrics_val, metrics_test, str(out_dir))

    print(f"\n✅ Step 4 Complete!")
    print(f"   Output dir: {out_dir.absolute()}")
    print(f"   Files: pr_curve.png | confusion_matrix.png | f1_confidence_curve.png | metrics.json")
    print(f"\n   → Use conf={best_conf:.2f} in your pipeline (best F1 threshold)")
    print("\n✅ Run step5_temporal_validation.py next.")

    return metrics_val, metrics_test


if __name__ == "__main__":
    main()
