#All Code for Downloading Dataset, testing, and exporting to zip folder for Ping Pong YOLOv26

#import dataset and train model
from ultralytics import YOLO
from roboflow import Roboflow
import os
import random
import shutil
import glob
import yaml
from pathlib import Path
import torch
from multiprocessing import freeze_support


def collect_image_label_pairs(root):
    allowed_image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        if any(part.startswith('.') for part in Path(dirpath).parts):
            continue
        for filename in filenames:
            if Path(filename).suffix.lower() in allowed_image_exts:
                image_paths.append(Path(dirpath) / filename)
    image_paths.sort()

    pairs = []
    for image_path in image_paths:
        label_path = image_path.with_suffix('.txt')
        if not label_path.exists():
            parts = list(image_path.parts)
            lower_parts = [p.lower() for p in parts]
            if 'images' in lower_parts:
                images_index = lower_parts.index('images')
                label_parts = parts[:images_index] + ['labels'] + parts[images_index + 1:]
                label_path = Path(*label_parts).with_suffix('.txt')
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            print(f"Warning: skipping {image_path} because label file was not found")
    return pairs


def write_split_yaml(split_dir, nc, names, output_file):
    data = {
        'train': str(Path(split_dir) / 'train' / 'images').replace('\\', '/'),
        'val': str(Path(split_dir) / 'val' / 'images').replace('\\', '/'),
        'test': str(Path(split_dir) / 'test' / 'images').replace('\\', '/'),
        'nc': nc,
        'names': names,
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False)


def create_split_dataset(dataset_root, pairs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if not pairs:
        raise ValueError('No image/label pairs found to split.')
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError('Split ratios must sum to 1.0')

    random.seed(42)
    random.shuffle(pairs)
    total = len(pairs)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    if val_count == 0 and total >= 2:
        val_count = 1
        train_count = max(total - 2, 1)
        test_count = total - train_count - val_count
    if test_count == 0 and total >= 2:
        test_count = 1
        train_count = max(total - val_count - 1, 1)
        test_count = total - train_count - val_count

    split_dir = Path(dataset_root) / 'split'
    for subset in ['train', 'val', 'test']:
        for subfolder in ['images', 'labels']:
            dest = split_dir / subset / subfolder
            dest.mkdir(parents=True, exist_ok=True)

    subsets = [
        ('train', pairs[:train_count]),
        ('val', pairs[train_count:train_count + val_count]),
        ('test', pairs[train_count + val_count:]),
    ]

    for subset, batch in subsets:
        for image_path, label_path in batch:
            image_dest = split_dir / subset / 'images' / image_path.name
            label_dest = split_dir / subset / 'labels' / label_path.name
            shutil.copy2(image_path, image_dest)
            shutil.copy2(label_path, label_dest)

    return split_dir


def main():
    # Update YOUR_API_KEY with your Roboflow API key.
    rf = Roboflow(api_key="BkfVrYURZpLcMQliz2D2")
    project = rf.workspace("pingpong-1bery").project("pingpong-eguq0")
    version = project.version(9)
    dataset = version.download("yolo26")

    data_yaml = os.path.join(dataset.location, "data.yaml")
    with open(data_yaml, 'r', encoding='utf-8') as f:
        raw_data = yaml.safe_load(f)

    nc = raw_data.get('nc', 1)
    names = raw_data.get('names', [])
    if not names:
        names = ['object']

    image_root = Path(dataset.location) / 'images'
    label_root = Path(dataset.location) / 'labels'
    if not image_root.exists() or not label_root.exists():
        image_root = Path(dataset.location)
        label_root = Path(dataset.location)

    pairs = collect_image_label_pairs(image_root)
    if not pairs:
        raise RuntimeError(f'No image-label pairs found in {image_root}')

    split_dir = create_split_dataset(dataset.location, pairs, 0.7, 0.15, 0.15)
    split_data_yaml = Path(dataset.location) / 'split' / 'data_split.yaml'
    write_split_yaml(split_dir, nc, names, split_data_yaml)

    # Check for CUDA-capable GPU and select device
    if torch.cuda.is_available():
        print(f"CUDA available. GPU count: {torch.cuda.device_count()}")
        try:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        train_device = 0  # pass 0 to use the first GPU
    else:
        print("No CUDA GPU detected. Training will run on CPU.")
        train_device = 'cpu'

    model = YOLO("yolo26x.pt")
    model.train(data=str(split_data_yaml), epochs=100, name="ping_model", device=train_device, batch=4)

    # Create 'ping_modelv1' folder to store model weights and train results
    output_dir = os.path.join(os.getcwd(), "ping_modelv1")
    os.makedirs(output_dir, exist_ok=True)

    # Find latest run folder under runs/*/* (handles 'train', 'detect', etc.)
    run_candidates = glob.glob(os.path.join("runs", "*", "*"))
    latest_run = None
    if run_candidates:
        # prefer runs that contain weights or .pt files; fall back to newest mtime
        def run_score(path):
            weight_dir = os.path.join(path, "weights")
            pt_files = glob.glob(os.path.join(path, "**", "*.pt"), recursive=True)
            score = 0
            if os.path.isdir(weight_dir):
                score += 1000
            score += len(pt_files)
            # small contribution from mtime so newer runs win ties
            try:
                score += os.path.getmtime(path) * 1e-6
            except Exception:
                pass
            return score

        latest_run = max(run_candidates, key=run_score)
        dest = os.path.join(output_dir, os.path.basename(latest_run))
        try:
            shutil.copytree(latest_run, dest, dirs_exist_ok=True)
        except Exception as e:
            print(f"Warning: failed to copy training run: {e}")
    else:
        print("No runs/*/* folders found to copy.")

    # Copy any .pt weight files (from latest run, any runs subfolder, or current dir) into output_dir
    pt_files = set()
    if latest_run:
        pt_files.update(glob.glob(os.path.join(latest_run, "**", "*.pt"), recursive=True))
    # also check all runs subfolders for .pt files
    pt_files.update(glob.glob(os.path.join("runs", "**", "*.pt"), recursive=True))
    # and top-level .pt files
    pt_files.update(glob.glob("*.pt"))
    for pt in sorted(pt_files):
        try:
            shutil.copy(pt, output_dir)
        except Exception as e:
            print(f"Warning: failed to copy {pt}: {e}")

    # Zip into 'ping_modelv1.zip'
    zip_base = os.path.join(os.getcwd(), "ping_modelv1")
    shutil.make_archive(zip_base, 'zip', root_dir=output_dir)
    print(f"Created zip: {zip_base}.zip")


if __name__ == '__main__':
    freeze_support()
    main()
