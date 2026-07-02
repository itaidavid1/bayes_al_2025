import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel
from tqdm import tqdm


BATCH_SIZE = 256
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# HuggingFace model names: dinov2-small, dinov2-base, dinov2-large, dinov2-giant
DINOV2_MODEL = "facebook/dinov2-base"
DATA_DIR = "./data"
OUTPUT_DIR = "./embeddings"

# DINOv2 expects at least 224x224; CIFAR images are 32x32
RESIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transform():
    return transforms.Compose([
        transforms.Resize((RESIZE, RESIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_datasets(transform):
    cifar10_train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)
    cifar100_train = datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=transform)
    cifar100_test = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=transform)

    return {
        "cifar10_train": cifar10_train,
        "cifar10_test": cifar10_test,
        "cifar100_train": cifar100_train,
        "cifar100_test": cifar100_test,
    }


HF_CACHE_DIR = "/cs/labs/daphna/itai.david/.cache/huggingface"


def load_dinov2(model_name: str) -> nn.Module:
    model = AutoModel.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model.eval()
    model.to(DEVICE)
    return model


@torch.inference_mode()
def extract_embeddings(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(DEVICE)
        outputs = model(pixel_values=images)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def save_embeddings(embeddings: np.ndarray, labels: np.ndarray, name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.npz")
    np.savez_compressed(path, embeddings=embeddings, labels=labels)
    print(f"Saved {name}: embeddings={embeddings.shape}, labels={labels.shape} -> {path}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Model:  {DINOV2_MODEL}\n")

    transform = get_transform()
    datasets_dict = load_datasets(transform)

    print("Loading DINOv2 model...")
    model = load_dinov2(DINOV2_MODEL)
    embed_dim = model.config.hidden_size
    print(f"Embed dim: {embed_dim}\n")

    for split_name, dataset in datasets_dict.items():
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=DEVICE == "cuda",
        )
        print(f"Extracting embeddings for {split_name} ({len(dataset)} samples)...")
        embeddings, labels = extract_embeddings(model, loader)
        model_tag = DINOV2_MODEL.replace("/", "_")
        save_embeddings(embeddings, labels, f"{model_tag}_{split_name}")

    print("\nDone. All embeddings saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()