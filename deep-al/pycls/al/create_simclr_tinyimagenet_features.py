import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

# --- Configuration ---
CHECKPOINT_PATH = '/cs/labs/daphna/itai.david/representations_bank/tiny-imagenet_simclr/pretext/model_seed1.pth.tar'  # Path to your SimCLR checkpoint
DATA_DIR = '/cs/labs/daphna/itai.david/py_repos/TypiClust/data/tiny-imagenet-200/tiny-imagenet-200'  # Root of your TinyImageNet dataset
BATCH_SIZE = 256
NUM_WORKERS = 4
OUTPUT_FILE = 'simclr_features.npz'  # Output file name

import torch
import torch.nn as nn
from torchvision.models import resnet18

import torch
import torch.nn as nn
from torchvision.models import resnet18

import os
import shutil


def format_val_directory(val_dir='./val'):
    """
    Restructures the TinyImageNet validation directory from a flat list of images
    to a class-wise subdirectory structure compatible with ImageFolder.
    """

    # Paths
    images_dir = os.path.join(val_dir, 'images')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')

    # Check if files exist
    if not os.path.exists(images_dir):
        print(f"Error: 'images' folder not found in {val_dir}")
        return
    if not os.path.exists(annotations_file):
        print(f"Error: 'val_annotations.txt' not found in {val_dir}")
        return

    # Open annotations file
    print("Reading annotations and moving files...")
    with open(annotations_file, 'r') as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        parts = line.strip().split('\t')
        filename = parts[0]
        class_label = parts[1]

        # Source path (where the image is now)
        src_path = os.path.join(images_dir, filename)

        # Destination directory (e.g., ./val/n01440764/)
        dest_dir = os.path.join(val_dir, class_label)

        # Destination path
        dest_path = os.path.join(dest_dir, filename)

        # 1. Create class directory if it doesn't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # 2. Move the file
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
            count += 1

    # Optional: Remove the now empty 'images' folder
    # if len(os.listdir(images_dir)) == 0:
    #     os.rmdir(images_dir)

    print(f"Successfully moved {count} images into class subdirectories.")

def load_simclr_backbone(checkpoint_path):
    # 1. Initialize Model (Modified for TinyImageNet/CIFAR)
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Common in SimCLR for small images
    model.fc = nn.Identity()

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    # 2. Clean and Rename Keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            # Remove 'backbone.' prefix
            k = k.replace('backbone.', '')

            # --- THE FIX: Rename 'shortcut' to 'downsample' ---
            if 'shortcut' in k:
                k = k.replace('shortcut', 'downsample')

            # Filter out fc layers just in case
            if not k.startswith('fc'):
                new_state_dict[k] = v

    # 3. Load Weights
    # We still use strict=False because the checkpoint might have extra keys
    # (like layer1.0.shortcut) that standard ResNet18 doesn't use.
    msg = model.load_state_dict(new_state_dict, strict=False)

    print("Weights loaded.")
    print(f"Missing keys: {len(msg.missing_keys)} (Should be 0 now)")
    if len(msg.missing_keys) > 0:
        print(f"Sample missing: {msg.missing_keys[:3]}")

    return model


def extract_features(data_dir, split='train'):
    """Extracts features for a specific split (train/val)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transformations (Standard for validation/testing: Resize + Crop + Normalize)
    transform = transforms.Compose([
        transforms.Resize(64),  # TinyImageNet is 64x64
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    split_dir = os.path.join(data_dir, split)

    # Check if split exists
    if not os.path.exists(split_dir):
        print(f"Warning: {split_dir} does not exist. Skipping.")
        return None, None

    dataset = datasets.ImageFolder(split_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Extracting features from {split_dir}...")

    # Model Setup
    model = load_simclr_backbone(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()

    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)

            # Forward pass (output is batch_size x 512)
            output = model(images)

            # Move to CPU and flatten if necessary
            features_list.append(output.cpu().numpy())
            labels_list.append(targets.numpy())

    # Concatenate all batches
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    return all_features, all_labels


if __name__ == '__main__':
    # 1. Extract Train Features
    # train_feats, train_labels = extract_features(DATA_DIR, split='train')
    # if train_feats is not None:
    #     print(f"Train Shape: {train_feats.shape}")  # Should be (100000, 512)

    # 2. Extract Val Features
    # format_val_directory(val_dir=os.path.join(DATA_DIR, 'val'))
    val_feats, val_labels = extract_features(DATA_DIR, split='val')
    if val_feats is not None:
        print(f"Val Shape: {val_feats.shape}")  # Should be (10000, 512)

    # 3. Save to disk
    print(f"Saving to {OUTPUT_FILE}...")
    save_dict = {}
    if train_feats is not None:
        save_dict['train_features'] = train_feats
        save_dict['train_labels'] = train_labels
    if val_feats is not None:
        save_dict['val_features'] = val_feats
        save_dict['val_labels'] = val_labels

    np.savez_compressed(OUTPUT_FILE, **save_dict)
    print("Done!")