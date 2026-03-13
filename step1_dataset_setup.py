"""
STEP 1: Dataset Setup & BowFire Conversion
==========================================
Run this FIRST before anything else.
Sets up folder structure + converts BowFire XML → YOLO format
"""

import os
import cv2
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG — Edit these paths to match your setup
# ─────────────────────────────────────────────
CONFIG = {
    "base_dir":            "/home/admincit/fire_project/dataset",
    "multiscale_images":   "/home/admincit/fire_project/raw/multiscale/Multi-Scale-Fire-Smoke-and-Flame-Dataset/train/images",
    "multiscale_labels":   "/home/admincit/fire_project/raw/multiscale/Multi-Scale-Fire-Smoke-and-Flame-Dataset/train/labels",
    "bowfire_images":      "/home/admincit/fire_project/raw/bowfire",
    "bowfire_annotations": "/home/admincit/fire_project/raw/bowfire",
    "train_ratio":         0.80,
    "val_ratio":           0.20,
    "class_names":         ["fire", "smoke"],
    "bowfire_class_id":    0,
}

# ─────────────────────────────────────────────
# FOLDER STRUCTURE
# ─────────────────────────────────────────────

def create_folder_structure(base: str):
    """Create clean train/val/test folder structure."""
    splits = ["train", "val", "test"]
    subdirs = ["images", "labels"]
    for split in splits:
        for sub in subdirs:
            path = Path(base) / split / sub
            path.mkdir(parents=True, exist_ok=True)
    # Hard negatives folder (non-fire images to reduce false positives)
    (Path(base) / "train" / "hard_negatives").mkdir(parents=True, exist_ok=True)
    print(f"✅ Folder structure created under '{base}/'")


# ─────────────────────────────────────────────
# BOWFIRE: XML → YOLO FORMAT CONVERTER
# ─────────────────────────────────────────────

def get_image_size(img_path: str):
    """Read image dimensions."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    return w, h


def convert_xml_to_yolo(xml_path: str, img_w: int, img_h: int, class_id: int = 0):
    """
    Convert a single BowFire Pascal VOC XML annotation to YOLO format.
    Returns list of YOLO label strings: "class x_c y_c w h"
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"⚠️  XML parse error in {xml_path}: {e}")
        return []

    yolo_labels = []

    for obj in root.findall("object"):
        name = obj.find("name")
        if name is not None:
            label = name.text.strip().lower()
            # Map BowFire class names to IDs
            if label in ["fire", "flame"]:
                cls = 0
            elif label == "smoke":
                cls = 1
            else:
                cls = class_id  # fallback

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        try:
            xmin = max(0, int(float(bbox.find("xmin").text)))
            ymin = max(0, int(float(bbox.find("ymin").text)))
            xmax = min(img_w, int(float(bbox.find("xmax").text)))
            ymax = min(img_h, int(float(bbox.find("ymax").text)))
        except (ValueError, AttributeError) as e:
            print(f"⚠️  Bbox parse error in {xml_path}: {e}")
            continue

        if xmax <= xmin or ymax <= ymin:
            print(f"⚠️  Invalid bbox skipped in {xml_path}")
            continue

        # Convert to YOLO normalized format
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width    = (xmax - xmin) / img_w
        height   = (ymax - ymin) / img_h

        # Clamp values to [0, 1]
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width    = min(max(width, 0.0), 1.0)
        height   = min(max(height, 0.0), 1.0)

        yolo_labels.append(
            f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return yolo_labels


def convert_bowfire_dataset(
    bowfire_images_dir: str,
    bowfire_annotations_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
):
    """
    Batch-convert entire BowFire dataset to YOLO format.
    Copies images + writes .txt label files.
    """
    img_dir   = Path(bowfire_images_dir)
    ann_dir   = Path(bowfire_annotations_dir)
    out_imgs  = Path(output_images_dir)
    out_lbls  = Path(output_labels_dir)
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in img_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"⚠️  No images found in {bowfire_images_dir}")
        return 0

    converted, skipped = 0, 0

    for img_file in tqdm(image_files, desc="Converting BowFire"):
        # Find matching XML (same stem, .xml extension)
        xml_file = ann_dir / (img_file.stem + ".xml")

        if not xml_file.exists():
            # Try alternate: image name with annotation subfolder
            xml_file = ann_dir / img_file.name.replace(img_file.suffix, ".xml")

        if not xml_file.exists():
            print(f"⚠️  No annotation for {img_file.name}, skipping.")
            skipped += 1
            continue

        try:
            img_w, img_h = get_image_size(str(img_file))
        except FileNotFoundError:
            skipped += 1
            continue

        yolo_labels = convert_xml_to_yolo(str(xml_file), img_w, img_h)

        # Copy image
        dest_img = out_imgs / img_file.name
        shutil.copy2(str(img_file), str(dest_img))

        # Write label file (empty .txt if no annotations = hard negative)
        label_file = out_lbls / (img_file.stem + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_labels))

        converted += 1

    print(f"✅ BowFire converted: {converted} images | Skipped: {skipped}")
    return converted


# ─────────────────────────────────────────────
# MULTI-SCALE DATASET: TRAIN/VAL SPLIT
# ─────────────────────────────────────────────

def split_multiscale_dataset(
    images_dir: str,
    labels_dir: str,
    output_base: str,
    train_ratio: float = 0.80,
    seed: int = 42,
):
    """
    Split Multi-Scale dataset into train/val sets.
    Copies images + labels maintaining YOLO structure.
    """
    random.seed(seed)
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [f for f in img_dir.iterdir() if f.suffix.lower() in image_extensions]

    # Only keep images that have matching label files
    valid_pairs = []
    for img in all_images:
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            valid_pairs.append((img, lbl))
        else:
            print(f"⚠️  No label for {img.name}, skipping.")

    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs   = valid_pairs[split_idx:]

    def copy_pairs(pairs, split_name):
        img_out = Path(output_base) / split_name / "images"
        lbl_out = Path(output_base) / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, lbl in tqdm(pairs, desc=f"Copying {split_name}"):
            shutil.copy2(str(img), str(img_out / img.name))
            shutil.copy2(str(lbl), str(lbl_out / lbl.name))

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs,   "val")

    print(f"✅ Multi-Scale split → Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    return len(train_pairs), len(val_pairs)


# ─────────────────────────────────────────────
# COPY BOWFIRE → TEST SET
# ─────────────────────────────────────────────

def setup_test_set(bowfire_converted_images: str, bowfire_converted_labels: str, output_base: str):
    """Copy converted BowFire images/labels to test/ folder."""
    test_img = Path(output_base) / "test" / "images"
    test_lbl = Path(output_base) / "test" / "labels"
    test_img.mkdir(parents=True, exist_ok=True)
    test_lbl.mkdir(parents=True, exist_ok=True)

    for img in Path(bowfire_converted_images).glob("*"):
        shutil.copy2(str(img), str(test_img / img.name))
    for lbl in Path(bowfire_converted_labels).glob("*.txt"):
        shutil.copy2(str(lbl), str(test_lbl / lbl.name))

    n = len(list(test_img.glob("*")))
    print(f"✅ Test set (BowFire): {n} images copied")


# ─────────────────────────────────────────────
# GENERATE fire.yaml
# ─────────────────────────────────────────────

def generate_yaml(base_dir: str, class_names: list):
    """Generate the YOLO training config YAML file."""
    yaml_content = f"""# YOLOv8 Fire Detection — Dataset Config
# Auto-generated by step1_dataset_setup.py

path: {Path(base_dir).absolute()}
train: train/images
val:   val/images
test:  test/images

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = Path(base_dir) / "fire.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"✅ YAML config saved: {yaml_path}")
    return str(yaml_path)


# ─────────────────────────────────────────────
# DATASET STATISTICS REPORT
# ─────────────────────────────────────────────

def report_dataset_stats(base_dir: str, class_names: list):
    """Print class distribution across all splits."""
    from collections import defaultdict
    print("\n📊 Dataset Statistics:")
    print("=" * 50)

    for split in ["train", "val", "test"]:
        lbl_dir = Path(base_dir) / split / "labels"
        if not lbl_dir.exists():
            continue

        class_counts = defaultdict(int)
        total_images = 0
        empty_images = 0

        for lbl_file in lbl_dir.glob("*.txt"):
            total_images += 1
            lines = lbl_file.read_text().strip().splitlines()
            if not lines:
                empty_images += 1
            for line in lines:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_counts[cls_id] += 1

        print(f"\n  [{split.upper()}] {total_images} images ({empty_images} empty/hard-neg)")
        for cls_id, name in enumerate(class_names):
            print(f"    Class {cls_id} ({name}): {class_counts[cls_id]} instances")

    print("=" * 50)


# ─────────────────────────────────────────────
# MAIN — Run all steps
# ─────────────────────────────────────────────

def main():
    cfg = CONFIG

    print("🔥 Fire Detection — Step 1: Dataset Setup")
    print("=" * 50)

    # 1. Create folder structure
    create_folder_structure(cfg["base_dir"])

    # 2. Convert BowFire to YOLO format
    print("\n📂 Converting BowFire dataset...")
    bowfire_conv_imgs = "raw/bowfire_converted/images"
    bowfire_conv_lbls = "raw/bowfire_converted/labels"
    convert_bowfire_dataset(
        cfg["bowfire_images"],
        cfg["bowfire_annotations"],
        bowfire_conv_imgs,
        bowfire_conv_lbls,
    )

    # 3. Split Multi-Scale dataset
    print("\n📂 Splitting Multi-Scale dataset...")
    split_multiscale_dataset(
        cfg["multiscale_images"],
        cfg["multiscale_labels"],
        cfg["base_dir"],
        cfg["train_ratio"],
    )

    # 4. Setup BowFire as test set
    print("\n📂 Setting up BowFire test set...")
    setup_test_set(bowfire_conv_imgs, bowfire_conv_lbls, cfg["base_dir"])

    # 5. Generate YAML
    print("\n📝 Generating fire.yaml...")
    generate_yaml(cfg["base_dir"], cfg["class_names"])

    # 6. Report stats
    report_dataset_stats(cfg["base_dir"], cfg["class_names"])

    print("\n✅ Step 1 Complete! Run step2_augmentation.py next.")


if __name__ == "__main__":
    main()
