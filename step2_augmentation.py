"""
STEP 2: Data Augmentation Pipeline
====================================
Run AFTER step1_dataset_setup.py
Augments training data + adds hard negatives to reduce false positives
"""

import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm
import shutil
import random

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "dataset_base":         "dataset",
    "train_images":         "dataset/train/images",
    "train_labels":         "dataset/train/labels",
    "augmented_output":     "dataset/train_augmented",
    "hard_negatives_dir":   "raw/hard_negatives",  # folder with non-fire images (sunset, lamps etc.)
    "augment_multiplier":   3,      # each image generates N augmented versions
    "image_size":           640,
    "seed":                 42,
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ─────────────────────────────────────────────
# AUGMENTATION PIPELINES
# ─────────────────────────────────────────────

def get_train_transforms(image_size: int):
    """
    Strong augmentation pipeline for fire/smoke training data.
    Fire detection requires robustness to: lighting, scale, occlusion, noise.
    """
    return A.Compose([
        # ── Spatial ──────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.RandomResizedCrop(size=(640, 640), 
            height=image_size,
            width=image_size,
            scale=(0.7, 1.0),   # simulate multi-scale fire at different distances
            ratio=(0.75, 1.33),
            p=0.4
        ),

        # ── Color / Photometric ───────────────────────────────────
        # Keep fire colors intact — constrained HSV shift
        A.HueSaturationValue(
            hue_shift_limit=8,     # SMALL — fire hue must stay orange/red
            sat_shift_limit=25,
            val_shift_limit=30,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.6
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
            p=0.3
        ),
        A.CLAHE(clip_limit=4.0, p=0.2),  # enhance local contrast — helps in dark scenes

        # ── Noise / Blur ──────────────────────────────────────────
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.3),

        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),   # simulate camera movement
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),

        # ── Occlusion / Environment ────────────────────────────────
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.3
        ),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, alpha_coef=0.1, p=0.15),
        A.RandomRain(
            slant_lower=-10,
            slant_upper=10,
            drop_length=10,
            drop_width=1,
            blur_value=3,
            p=0.1
        ),
        A.CoarseDropout(
            max_holes=4,
            max_height=40,
            max_width=40,
            min_holes=1,
            fill_value=0,
            p=0.2
        ),

        # ── Final Resize ──────────────────────────────────────────
        A.Resize(image_size, image_size, p=1.0),

    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,   # drop bbox if <30% visible after crop
        check_each_transform=False,
    ))


def get_mild_transforms(image_size: int):
    """
    Mild augmentation — only for generating more test variety from BowFire.
    Does NOT alter fire colors significantly.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.GaussNoise(var_limit=(5, 20), p=0.2),
        A.Resize(image_size, image_size, p=1.0),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ))


# ─────────────────────────────────────────────
# CORE AUGMENTATION FUNCTIONS
# ─────────────────────────────────────────────

def load_yolo_labels(label_path: str):
    """Load YOLO label file → list of (class_id, x_c, y_c, w, h)."""
    labels, classes = [], []
    path = Path(label_path)
    if not path.exists() or path.stat().st_size == 0:
        return labels, classes

    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls = int(parts[0])
            bbox = [float(x) for x in parts[1:]]
            # Clamp to [0,1]
            bbox = [min(max(v, 0.0), 1.0) for v in bbox]
            labels.append(bbox)
            classes.append(cls)

    return labels, classes


def save_yolo_labels(label_path: str, bboxes: list, classes: list):
    """Save augmented YOLO labels to file."""
    with open(label_path, "w") as f:
        for cls, bbox in zip(classes, bboxes):
            x_c, y_c, w, h = bbox
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def augment_single_image(
    img_path: str,
    lbl_path: str,
    out_img_dir: str,
    out_lbl_dir: str,
    transform,
    n_augments: int = 3,
    prefix: str = "aug",
):
    """Apply N augmentations to a single image+label pair."""
    img = cv2.imread(img_path)
    if img is None:
        return 0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, classes = load_yolo_labels(lbl_path)

    stem = Path(img_path).stem
    ext  = Path(img_path).suffix
    generated = 0

    for i in range(n_augments):
        try:
            result = transform(
                image=img_rgb,
                bboxes=bboxes,
                class_labels=classes,
            )
        except Exception as e:
            print(f"⚠️  Augmentation failed for {img_path}: {e}")
            continue

        aug_img  = result["image"]
        aug_bboxes  = result["bboxes"]
        aug_classes = result["class_labels"]

        # Skip if all bboxes were lost during augmentation
        if bboxes and not aug_bboxes:
            continue

        out_name  = f"{prefix}_{stem}_{i:03d}"
        img_out   = Path(out_img_dir) / f"{out_name}{ext}"
        lbl_out   = Path(out_lbl_dir) / f"{out_name}.txt"

        # Save image (convert back to BGR for cv2)
        cv2.imwrite(str(img_out), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        save_yolo_labels(str(lbl_out), aug_bboxes, aug_classes)
        generated += 1

    return generated


# ─────────────────────────────────────────────
# HARD NEGATIVES — Reduce False Positives
# ─────────────────────────────────────────────

def add_hard_negatives(hard_neg_dir: str, out_img_dir: str, out_lbl_dir: str, image_size: int):
    """
    Copy non-fire images (sunsets, red lights, lamps) as hard negatives.
    Labels are EMPTY .txt files → YOLO learns "not fire" for these scenes.

    How to collect hard negatives:
    - Download sunset images from Unsplash/Flickr (free license)
    - Screenshot red/orange illuminated indoor scenes
    - Grab frames from non-fire surveillance footage
    Aim for 200–400 hard negative images.
    """
    hard_dir = Path(hard_neg_dir)
    if not hard_dir.exists():
        print(f"⚠️  Hard negatives dir not found: {hard_neg_dir}")
        print("    Create the folder and add non-fire images (sunsets, lamps, red lights)")
        print("    Skipping hard negatives for now...")
        return 0

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in hard_dir.iterdir() if f.suffix.lower() in image_extensions]

    resize = A.Resize(image_size, image_size)
    added = 0

    for img_file in tqdm(images, desc="Adding hard negatives"):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        img_resized = resize(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"]

        out_img = Path(out_img_dir) / f"hn_{img_file.name}"
        out_lbl = Path(out_lbl_dir) / f"hn_{img_file.stem}.txt"

        cv2.imwrite(str(out_img), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        open(str(out_lbl), "w").close()  # empty label = no fire
        added += 1

    print(f"✅ Hard negatives added: {added}")
    return added


# ─────────────────────────────────────────────
# MOSAIC AUGMENTATION (manual implementation)
# ─────────────────────────────────────────────

def create_mosaic(img_label_pairs: list, image_size: int):
    """
    YOLOv8 mosaic: combine 4 images into one.
    Massively helps with multi-scale detection.
    """
    assert len(img_label_pairs) == 4, "Need exactly 4 image-label pairs"

    half = image_size // 2
    mosaic_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    mosaic_labels = []
    mosaic_classes = []

    positions = [
        (0, 0, half, half),         # top-left
        (half, 0, image_size, half), # top-right
        (0, half, half, image_size), # bottom-left
        (half, half, image_size, image_size),  # bottom-right
    ]
    norm_offsets = [
        (0.0, 0.0),   # top-left
        (0.5, 0.0),   # top-right
        (0.0, 0.5),   # bottom-left
        (0.5, 0.5),   # bottom-right
    ]

    for idx, (img_path, lbl_path) in enumerate(img_label_pairs):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (half, half))

        x1, y1, x2, y2 = positions[idx]
        mosaic_img[y1:y2, x1:x2] = img

        bboxes, classes = load_yolo_labels(lbl_path)
        ox, oy = norm_offsets[idx]

        for bbox, cls in zip(bboxes, classes):
            xc, yc, w, h = bbox
            # Scale bbox to half size, then offset
            new_xc = ox + xc * 0.5
            new_yc = oy + yc * 0.5
            new_w  = w * 0.5
            new_h  = h * 0.5
            mosaic_labels.append([new_xc, new_yc, new_w, new_h])
            mosaic_classes.append(cls)

    return mosaic_img, mosaic_labels, mosaic_classes


def generate_mosaics(
    images_dir: str,
    labels_dir: str,
    out_img_dir: str,
    out_lbl_dir: str,
    n_mosaics: int,
    image_size: int,
):
    """Generate N mosaic images from training data."""
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    all_pairs = []
    for img_file in img_dir.iterdir():
        if img_file.suffix.lower() not in image_extensions:
            continue
        lbl_file = lbl_dir / (img_file.stem + ".txt")
        if lbl_file.exists():
            all_pairs.append((str(img_file), str(lbl_file)))

    if len(all_pairs) < 4:
        print("⚠️  Not enough images for mosaic generation")
        return

    generated = 0
    for i in tqdm(range(n_mosaics), desc="Generating mosaics"):
        selected = random.sample(all_pairs, 4)
        mosaic_img, labels, classes = create_mosaic(selected, image_size)

        out_img = Path(out_img_dir) / f"mosaic_{i:04d}.jpg"
        out_lbl = Path(out_lbl_dir) / f"mosaic_{i:04d}.txt"

        cv2.imwrite(str(out_img), mosaic_img)
        save_yolo_labels(str(out_lbl), labels, classes)
        generated += 1

    print(f"✅ Mosaics generated: {generated}")


# ─────────────────────────────────────────────
# AUGMENT BOWFIRE TEST SET (mild only)
# ─────────────────────────────────────────────

def augment_bowfire_test(
    test_img_dir: str,
    test_lbl_dir: str,
    n_augments: int = 1,
    image_size: int = 640,
):
    """
    Mildly augment BowFire test set to increase size from ~226 to ~450+ images.
    Uses MILD transforms only — preserves test set integrity.
    """
    mild_transform = get_mild_transforms(image_size)
    img_dir = Path(test_img_dir)
    lbl_dir = Path(test_lbl_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in image_extensions]

    total = 0
    for img_file in tqdm(images, desc="Augmenting BowFire test"):
        lbl_file = lbl_dir / (img_file.stem + ".txt")
        total += augment_single_image(
            str(img_file), str(lbl_file),
            str(img_dir), str(lbl_dir),
            mild_transform,
            n_augments=n_augments,
            prefix="bf_aug",
        )

    print(f"✅ BowFire test augmented: +{total} images")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    cfg = CONFIG
    print("🔥 Fire Detection — Step 2: Data Augmentation")
    print("=" * 50)

    # Setup output dirs
    aug_img = Path(cfg["augmented_output"]) / "images"
    aug_lbl = Path(cfg["augmented_output"]) / "labels"
    aug_img.mkdir(parents=True, exist_ok=True)
    aug_lbl.mkdir(parents=True, exist_ok=True)

    transform = get_train_transforms(cfg["image_size"])

    # ── Copy original training data first ────────────────
    print("\n📂 Copying original training data...")
    src_imgs = Path(cfg["train_images"])
    src_lbls = Path(cfg["train_labels"])
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    original_images = list(src_imgs.glob("*"))
    for img_file in tqdm(original_images, desc="Copying originals"):
        if img_file.suffix.lower() not in image_extensions:
            continue
        shutil.copy2(str(img_file), str(aug_img / img_file.name))
        lbl = src_lbls / (img_file.stem + ".txt")
        if lbl.exists():
            shutil.copy2(str(lbl), str(aug_lbl / lbl.name))

    # ── Generate augmented versions ───────────────────────
    print(f"\n🔄 Generating {cfg['augment_multiplier']}x augmented versions...")
    total_aug = 0
    for img_file in tqdm(original_images, desc="Augmenting"):
        if img_file.suffix.lower() not in image_extensions:
            continue
        lbl_file = src_lbls / (img_file.stem + ".txt")
        total_aug += augment_single_image(
            str(img_file), str(lbl_file),
            str(aug_img), str(aug_lbl),
            transform,
            n_augments=cfg["augment_multiplier"],
        )
    print(f"✅ Generated {total_aug} augmented images")

    # ── Generate mosaics ──────────────────────────────────
    n_mosaics = min(500, len(original_images) // 2)
    print(f"\n🔲 Generating {n_mosaics} mosaic images...")
    generate_mosaics(
        cfg["train_images"], cfg["train_labels"],
        str(aug_img), str(aug_lbl),
        n_mosaics, cfg["image_size"],
    )

    # ── Add hard negatives ────────────────────────────────
    print("\n🚫 Adding hard negatives...")
    add_hard_negatives(
        cfg["hard_negatives_dir"],
        str(aug_img), str(aug_lbl),
        cfg["image_size"],
    )

    # ── Augment BowFire test set (mild) ───────────────────
    print("\n🎯 Mildly augmenting BowFire test set...")
    augment_bowfire_test(
        f"{cfg['dataset_base']}/test/images",
        f"{cfg['dataset_base']}/test/labels",
        n_augments=1,
        image_size=cfg["image_size"],
    )

    # ── Update fire.yaml to point to augmented train ──────
    yaml_path = Path(cfg["dataset_base"]) / "fire.yaml"
    if yaml_path.exists():
        content = yaml_path.read_text()
        content = content.replace("train: train/images", f"train: {cfg['augmented_output']}/images")
        yaml_path.write_text(content)
        print(f"\n✅ fire.yaml updated to use augmented training data")

    # Summary
    total_train = len(list(aug_img.glob("*")))
    total_val   = len(list(Path(f"{cfg['dataset_base']}/val/images").glob("*"))) if Path(f"{cfg['dataset_base']}/val/images").exists() else 0
    total_test  = len(list(Path(f"{cfg['dataset_base']}/test/images").glob("*"))) if Path(f"{cfg['dataset_base']}/test/images").exists() else 0

    print("\n📊 Final Dataset Summary:")
    print(f"  Train (augmented): {total_train} images")
    print(f"  Val:               {total_val} images")
    print(f"  Test (BowFire):    {total_test} images")
    print("\n✅ Step 2 Complete! Run step3_train.py next.")


if __name__ == "__main__":
    main()
