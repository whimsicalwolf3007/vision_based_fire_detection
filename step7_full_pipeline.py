"""
STEP 7: Full End-to-End Pipeline
==================================
Camera → YOLOv8 → TVL → Distance → Control Logic → Logging
Run this for live inference or video demo for the paper
"""

import cv2
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# Import our modules
from step5_temporal_validation import TemporalValidationLayer
from step6_distance_estimation import FusedDistanceEstimator, draw_distance_overlay, DistanceEstimate


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "weights":          "runs/detect/runs/fire_detection/train_20260312_1229/weights/best.pt",
    "source":           0,          # 0=webcam | "path/to/video.mp4" | "path/to/images/"
    "output_dir":       "pipeline_output",
    "save_video":       True,
    "display":          True,

    # Detection
    "conf":             0.25,
    "iou":              0.45,
    "imgsz":            640,
    "class_names":      ["fire", "smoke"],

    # TVL (use values from step5 grid search)
    "tvl_window":       5,
    "tvl_threshold":    3,

    # Distance estimation
    "focal_length_px":  700.0,
    "fire_width_ref_m": 0.5,
    "smoke_width_ref_m":1.5,
    "image_width_px":   640,
    "image_height_px":  480,
    "use_midas":        False,       # set True for paper demo (slower)
    "midas_model":      "MiDaS_small",

    # Alert thresholds
    "alert_zone":       "NEAR",      # alert if fire within this zone or closer
    "alert_cooldown_s": 5.0,         # min seconds between repeated alerts

    # Logging
    "log_file":         "pipeline_output/fire_log.json",
}

ZONE_PRIORITY = {"CRITICAL": 4, "NEAR": 3, "MEDIUM": 2, "FAR": 1, "OUT_OF_RANGE": 0}


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def setup_logging(output_dir: str) -> logging.Logger:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("FireDetection")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(out / "pipeline.log")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────
# ALERT SYSTEM
# ─────────────────────────────────────────────

class AlertSystem:
    def __init__(self, alert_zone: str, cooldown_s: float, log_file: str):
        self.alert_zone   = alert_zone
        self.cooldown_s   = cooldown_s
        self.log_file     = log_file
        self.last_alert_t = 0
        self.alert_log    = []
        self.total_alerts = 0

    def check_and_alert(self, detections: list, logger: logging.Logger) -> bool:
        """Check if alert should fire. Returns True if alert triggered."""
        if not detections:
            return False

        highest_priority = max(
            ZONE_PRIORITY.get(d.distance.zone if d.distance else "OUT_OF_RANGE", 0)
            for d in detections
        )
        alert_priority = ZONE_PRIORITY.get(self.alert_zone, 2)

        if highest_priority < alert_priority:
            return False

        now = time.time()
        if now - self.last_alert_t < self.cooldown_s:
            return False

        self.last_alert_t = now
        self.total_alerts += 1

        # Find closest fire
        fire_dets = [d for d in detections if d.distance]
        if fire_dets:
            closest = min(fire_dets, key=lambda d: d.distance.distance_m)
            msg = (
                f"🔥 FIRE ALERT #{self.total_alerts} | "
                f"Distance: {closest.distance.distance_m:.1f}m | "
                f"Zone: {closest.distance.zone} | "
                f"Pump: {closest.distance.pump_pressure:.0%}"
            )
            logger.warning(msg)
            self.alert_log.append({
                "time": datetime.now().isoformat(),
                "alert_id": self.total_alerts,
                "distance_m": closest.distance.distance_m,
                "zone": closest.distance.zone,
                "pump_pressure": closest.distance.pump_pressure,
            })
            self._save_log()

        return True

    def _save_log(self):
        with open(self.log_file, "w") as f:
            json.dump(self.alert_log, f, indent=2)


# ─────────────────────────────────────────────
# DISPLAY OVERLAY
# ─────────────────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    fps: float,
    frame_count: int,
    tvl_state: dict,
    alert_active: bool,
    pump_pressure: float,
) -> np.ndarray:
    """Draw HUD overlay with FPS, TVL state, pump pressure."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Frame count
    cv2.putText(frame, f"Frame: {frame_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # TVL state
    fire_count  = tvl_state.get("class_0_count", 0)
    smoke_count = tvl_state.get("class_1_count", 0)
    cv2.putText(frame, f"TVL: fire={fire_count}/5 smoke={smoke_count}/5",
                (w // 2 - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    # Pump pressure bar
    bar_x, bar_y, bar_w, bar_h = w - 160, 10, 140, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    filled = int(bar_w * pump_pressure)
    color  = (0, 0, 255) if pump_pressure > 0.7 else (0, 165, 255) if pump_pressure > 0.4 else (0, 255, 0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
    cv2.putText(frame, f"PUMP: {pump_pressure:.0%}", (bar_x, bar_y + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Alert flash
    if alert_active:
        h_f, w_f = frame.shape[:2]
        alert_overlay = frame.copy()
        cv2.rectangle(alert_overlay, (0, 0), (w_f, h_f), (0, 0, 255), -1)
        frame = cv2.addWeighted(alert_overlay, 0.15, frame, 0.85, 0)
        cv2.putText(frame, "⚠ FIRE DETECTED", (w_f//2 - 120, h_f - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame


# ─────────────────────────────────────────────
# GENERATE DEMO VIDEO FROM IMAGES
# ─────────────────────────────────────────────

def generate_demo_video(
    images_dir: str,
    output_path: str,
    model: YOLO,
    tvl: TemporalValidationLayer,
    estimator: FusedDistanceEstimator,
    cfg: dict,
    logger: logging.Logger,
    fps: int = 10,
):
    """
    Generate a demo video from test images for the paper.
    Simulates the full pipeline on BowFire test images.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([
        f for f in Path(images_dir).iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not images:
        print(f"⚠️  No images in {images_dir}")
        return

    # Read first image for dimensions
    first = cv2.imread(str(images[0]))
    h, w  = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    alert_sys = AlertSystem(cfg["alert_zone"], cfg["alert_cooldown_s"], cfg["log_file"])
    max_pressure = 0.0

    print(f"\n🎬 Generating demo video ({len(images)} frames)...")

    for frame_idx, img_file in enumerate(images):
        frame = cv2.imread(str(img_file))
        if frame is None:
            continue

        # YOLOv8 inference
        results = model(frame, conf=cfg["conf"], iou=cfg["iou"],
                       imgsz=cfg["imgsz"], verbose=False)

        raw_dets = []
        for r in results:
            for box in r.boxes:
                raw_dets.append({
                    "cls":  int(box.cls[0]),
                    "conf": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                })

        # TVL filter
        validated_dets = tvl.update(raw_dets)

        # Distance estimation + draw
        fire_detections = []
        current_max_pressure = 0.0

        for det in validated_dets:
            dist_est = estimator.estimate(frame, det["bbox"], det["cls"])
            cls_name = cfg["class_names"][det["cls"]] if det["cls"] < len(cfg["class_names"]) else "unknown"
            frame = draw_distance_overlay(frame, det["bbox"], dist_est, cls_name, det["conf"])

            from step6_distance_estimation import FireDetection
            fd = FireDetection(
                bbox_xyxy  = det["bbox"],
                cls_id     = det["cls"],
                cls_name   = cls_name,
                yolo_conf  = det["conf"],
                distance   = dist_est,
            )
            fire_detections.append(fd)
            current_max_pressure = max(current_max_pressure, dist_est.pump_pressure)

        max_pressure = max(max_pressure, current_max_pressure)
        alert_active = alert_sys.check_and_alert(fire_detections, logger)

        # HUD
        frame = draw_hud(frame, fps, frame_idx, tvl.state_summary(),
                        alert_active, current_max_pressure)

        writer.write(frame)

    writer.release()
    print(f"✅ Demo video saved: {output_path}")
    print(f"   Total alerts: {alert_sys.total_alerts}")
    print(f"   Max pump pressure: {max_pressure:.0%}")

    return alert_sys.alert_log


# ─────────────────────────────────────────────
# LIVE CAMERA INFERENCE
# ─────────────────────────────────────────────

def run_live(cfg: dict, logger: logging.Logger):
    """Run real-time fire detection on webcam or video stream."""

    model = YOLO(cfg["weights"])
    tvl   = TemporalValidationLayer(
        window_size=cfg["tvl_window"],
        threshold=cfg["tvl_threshold"],
        class_count=len(cfg["class_names"]),
    )
    estimator  = FusedDistanceEstimator(cfg)
    alert_sys  = AlertSystem(cfg["alert_zone"], cfg["alert_cooldown_s"], cfg["log_file"])

    cap = cv2.VideoCapture(cfg["source"])
    if not cap.isOpened():
        logger.error(f"Cannot open source: {cfg['source']}")
        return

    # Video writer (optional)
    writer = None
    if cfg["save_video"]:
        out_path = Path(cfg["output_dir"]) / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))

    logger.info("🔥 Fire detection pipeline started")
    frame_count = 0
    fps_deque   = deque(maxlen=30)
    t_prev      = time.time()
    pump_pressure = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            t_now = time.time()
            fps_deque.append(1.0 / (t_now - t_prev + 1e-9))
            t_prev = t_now
            current_fps = np.mean(fps_deque)

            # Inference
            results = model(frame, conf=cfg["conf"], iou=cfg["iou"],
                           imgsz=cfg["imgsz"], verbose=False)
            raw_dets = []
            for r in results:
                for box in r.boxes:
                    raw_dets.append({
                        "cls":  int(box.cls[0]),
                        "conf": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                    })

            # TVL
            validated = tvl.update(raw_dets)

            # Distance + draw
            fire_dets = []
            pump_pressure = 0.0
            for det in validated:
                dist_est = estimator.estimate(frame, det["bbox"], det["cls"])
                cls_name = cfg["class_names"][det["cls"]] if det["cls"] < len(cfg["class_names"]) else "?"
                frame = draw_distance_overlay(frame, det["bbox"], dist_est, cls_name, det["conf"])
                pump_pressure = max(pump_pressure, dist_est.pump_pressure)

                from step6_distance_estimation import FireDetection
                fire_dets.append(FireDetection(det["bbox"], det["cls"], cls_name, det["conf"], dist_est))

            alert = alert_sys.check_and_alert(fire_dets, logger)

            # HUD
            frame = draw_hud(frame, current_fps, frame_count,
                            tvl.state_summary(), alert, pump_pressure)

            if writer:
                writer.write(frame)

            if cfg["display"]:
                cv2.imshow("Fire Detection Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info(f"Pipeline ended | Frames: {frame_count} | Alerts: {alert_sys.total_alerts}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    cfg = CONFIG
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    logger = setup_logging(cfg["output_dir"])
    print("🔥 Fire Detection — Step 7: Full Pipeline")
    print("=" * 50)

    model = YOLO(cfg["weights"])
    tvl   = TemporalValidationLayer(
        window_size  = cfg["tvl_window"],
        threshold    = cfg["tvl_threshold"],
        class_count  = len(cfg["class_names"]),
    )
    estimator = FusedDistanceEstimator(cfg)

    # Generate demo video from BowFire test images (for paper)
    demo_path = str(Path(cfg["output_dir"]) / "demo_bowfire.mp4")
    alert_log = generate_demo_video(
        "dataset/test/images",
        demo_path,
        model, tvl, estimator, cfg, logger,
        fps=10,
    )

    print(f"\n✅ Pipeline complete!")
    print(f"   Demo video: {demo_path}")
    print(f"   Alert log:  {cfg['log_file']}")
    print(f"\n   → Submit to IEEE with:")
    print(f"     - evaluation_results/pr_curve.png")
    print(f"     - evaluation_results/confusion_matrix.png")
    print(f"     - evaluation_results/tvl_comparison.png")
    print(f"     - evaluation_results/distance_estimation_analysis.png")
    print(f"     - pipeline_output/demo_bowfire.mp4")


if __name__ == "__main__":
    main()
