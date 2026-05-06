from pathlib import Path
import random

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


class PairedTextureDataset(Dataset):
    """Loads texture folders that contain albedo.png and roughness.png."""

    def __init__(
        self,
        root_dir,
        image_size=(256, 256),
        augment=False,
        random_horizontal_flip=False,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment
        self.random_horizontal_flip = random_horizontal_flip

        self.samples = []
        for sample_dir in sorted(self.root_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            albedo_path = sample_dir / "albedo.png"
            roughness_path = sample_dir / "roughness.png"
            if albedo_path.exists() and roughness_path.exists():
                self.samples.append(
                    {
                        "name": sample_dir.name,
                        "albedo_path": albedo_path,
                        "roughness_path": roughness_path,
                    }
                )

        if not self.samples:
            raise ValueError(f"No paired texture folders found in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        albedo = Image.open(sample["albedo_path"]).convert("RGB")
        roughness = Image.open(sample["roughness_path"])

        albedo, roughness = self._transform_pair(albedo, roughness)

        albedo_tensor = TF.to_tensor(albedo)
        roughness_tensor = self._roughness_to_tensor(roughness)

        return albedo_tensor, roughness_tensor

    def _roughness_to_tensor(self, roughness):
        roughness_array = np.array(roughness)

        if roughness_array.ndim == 3:
            # Some downloaded roughness maps are stored as RGB/RGBA even though the
            # channels represent one grayscale map. Use one channel to keep [1,H,W].
            roughness_array = roughness_array[..., 0]

        roughness_array = roughness_array.astype(np.float32)
        roughness_tensor = torch.from_numpy(roughness_array).unsqueeze(0)

        # Preserve 16-bit grayscale maps instead of forcing them through 8-bit PIL conversion.
        if roughness_array.max() > 255:
            roughness_tensor = roughness_tensor / 65535.0
        else:
            roughness_tensor = roughness_tensor / 255.0

        return TF.resize(
            roughness_tensor,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

    def _transform_pair(self, albedo, roughness):
        if self.augment:
            albedo, roughness = self._random_resized_crop_pair(albedo, roughness)

            if random.random() < 0.5:
                albedo = TF.hflip(albedo)
                roughness = TF.hflip(roughness)
            if random.random() < 0.5:
                albedo = TF.vflip(albedo)
                roughness = TF.vflip(roughness)

            rotations = random.randint(0, 3)
            if rotations:
                angle = rotations * 90
                albedo = TF.rotate(albedo, angle)
                roughness = TF.rotate(roughness, angle)

            albedo = TF.adjust_brightness(albedo, random.uniform(0.9, 1.1))
            albedo = TF.adjust_contrast(albedo, random.uniform(0.9, 1.1))
            albedo = TF.adjust_saturation(albedo, random.uniform(0.95, 1.05))
        else:
            albedo = TF.resize(
                albedo,
                self.image_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

        if self.random_horizontal_flip and random.random() < 0.5:
            albedo = TF.hflip(albedo)
            roughness = TF.hflip(roughness)

        return albedo, roughness

    def _random_resized_crop_pair(self, albedo, roughness):
        width, height = albedo.size
        scale = random.uniform(0.65, 1.0)
        crop_width = max(1, int(width * scale))
        crop_height = max(1, int(height * scale))
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)

        albedo = TF.resized_crop(
            albedo,
            top,
            left,
            crop_height,
            crop_width,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        roughness = TF.resized_crop(
            roughness,
            top,
            left,
            crop_height,
            crop_width,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        return albedo, roughness

    def sample_name(self, index):
        return self.samples[index]["name"]


# def show_texture_pair(dataset, index=0):
#     albedo_tensor, roughness_tensor = dataset[index]

#     fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#     axes[0].imshow(albedo_tensor.permute(1, 2, 0))
#     axes[0].set_title(f"Albedo: {dataset.sample_name(index)}")
#     axes[0].axis("off")

#     axes[1].imshow(roughness_tensor.squeeze(0), cmap="gray", vmin=0.0, vmax=1.0)
#     axes[1].set_title("Roughness")
#     axes[1].axis("off")

#     plt.tight_layout()

def show_texture_pair(dataset, index=0):
    albedo_tensor, roughness_tensor = dataset[index]
    roughness_image = roughness_tensor.squeeze(0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(albedo_tensor.permute(1, 2, 0))
    axes[0].set_title(f"Albedo: {dataset.sample_name(index)}")
    axes[0].axis("off")

    axes[1].imshow(
        roughness_image,
        cmap="gray",
        vmin=roughness_image.min().item(),
        vmax=roughness_image.max().item(),
    )
    axes[1].set_title(
        f"Roughness\nmin={roughness_image.min().item():.3f}, max={roughness_image.max().item():.3f}"
    )
    axes[1].axis("off")

    plt.tight_layout()


def debug_texture_pair(dataset, index=0):
    sample = dataset.samples[index]

    albedo_image = Image.open(sample["albedo_path"])
    roughness_image = Image.open(sample["roughness_path"])

    albedo_array = np.array(albedo_image)
    roughness_array = np.array(roughness_image)

    albedo_tensor, roughness_tensor = dataset[index]

    print(f"Sample: {sample['name']}")
    print()
    print("Raw files")
    print(
        f"  albedo    mode={albedo_image.mode} dtype={albedo_array.dtype} "
        f"shape={albedo_array.shape} min={albedo_array.min()} max={albedo_array.max()}"
    )
    print(
        f"  roughness mode={roughness_image.mode} dtype={roughness_array.dtype} "
        f"shape={roughness_array.shape} min={roughness_array.min()} max={roughness_array.max()}"
    )
    print()
    print("Dataset tensors")
    print(
        f"  albedo    shape={tuple(albedo_tensor.shape)} dtype={albedo_tensor.dtype} "
        f"min={albedo_tensor.min().item():.6f} max={albedo_tensor.max().item():.6f}"
    )
    print(
        f"  roughness shape={tuple(roughness_tensor.shape)} dtype={roughness_tensor.dtype} "
        f"min={roughness_tensor.min().item():.6f} max={roughness_tensor.max().item():.6f}"
    )
