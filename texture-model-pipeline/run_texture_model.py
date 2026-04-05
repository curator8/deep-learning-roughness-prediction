from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from texture_dataset import PairedTextureDataset
from texture_model import (
    RoughnessAutoencoder,
    RoughnessUNet,
    fit_model,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_roughness_png(roughness_tensor: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = roughness_tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0).numpy()
    image_uint16 = (image * 65535.0).round().astype(np.uint16)
    Image.fromarray(image_uint16, mode="I;16").save(output_path)


def save_albedo_png(albedo_tensor: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = albedo_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    image_uint8 = (image * 255.0).round().astype(np.uint8)
    Image.fromarray(image_uint8, mode="RGB").save(output_path)


@torch.no_grad()
def export_prediction_artifacts(
    model: torch.nn.Module,
    dataset: PairedTextureDataset,
    sample_index: int,
    device: str,
    albedo_output_path: Path,
    target_output_path: Path,
    prediction_output_path: Path,
    preview_output_path: Path,
) -> None:
    model.eval()

    albedo_tensor, target_roughness = dataset[sample_index]
    predicted_roughness = model(albedo_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()

    save_albedo_png(albedo_tensor, albedo_output_path)
    save_roughness_png(target_roughness, target_output_path)
    save_roughness_png(predicted_roughness, prediction_output_path)

    preview_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(albedo_tensor.permute(1, 2, 0))
    axes[0].set_title(f"Albedo: {dataset.sample_name(sample_index)}")
    axes[0].axis("off")

    axes[1].imshow(target_roughness.squeeze(0), cmap="gray")
    axes[1].set_title("Target Roughness")
    axes[1].axis("off")

    axes[2].imshow(predicted_roughness.squeeze(0), cmap="gray")
    axes[2].set_title("Predicted Roughness")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(preview_output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    model_name: str,
    model_class,
    dataloader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = fit_model(model, dataloader, optimizer, device, epochs=epochs)

    print(f"\nRunning experiment: {model_name}")
    for row in history:
        print(
            f"Epoch {row['epoch']}/{epochs} - "
            f"train_mae={row['train_mae']:.6f}, "
            f"mae={row['mae']:.6f}, "
            f"rmse={row['rmse']:.6f}, "
            f"cosine_similarity={row['cosine_similarity']:.6f}"
        )

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train roughness prediction models, run ablations, and export a predicted map."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--sample-name", type=str, default="dirt_01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / "train"
    outputs_dir = project_root / "outputs"
    mesh_prediction_dir = (
        project_root
        / "simple-textured-mesh"
        / "static"
        / "predictions"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = PairedTextureDataset(
        root_dir=data_dir,
        image_size=(args.image_size, args.image_size),
        random_horizontal_flip=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    print(f"Dataset size: {len(dataset)} texture pairs")
    print("Note: metrics below are computed on the training set because no validation split exists yet.")

    if args.sample_index is not None:
        sample_index = args.sample_index
    else:
        sample_names = [sample["name"] for sample in dataset.samples]
        if args.sample_name not in sample_names:
            raise ValueError(
                f"Sample name '{args.sample_name}' not found. Available samples: {sample_names}"
            )
        sample_index = sample_names.index(args.sample_name)

    print(f"Export sample: {dataset.sample_name(sample_index)}")

    experiments = {
        "unet_skip_connections": RoughnessUNet,
        "autoencoder_no_skips": RoughnessAutoencoder,
    }

    ablation_results: dict[str, dict[str, float]] = {}
    trained_models: dict[str, torch.nn.Module] = {}

    for model_name, model_class in experiments.items():
        model, history = run_experiment(
            model_name=model_name,
            model_class=model_class,
            dataloader=dataloader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        trained_models[model_name] = model
        ablation_results[model_name] = history[-1]

    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = outputs_dir / "ablation_results.json"
    metrics_path.write_text(json.dumps(ablation_results, indent=2))

    print("\nAblation summary:")
    for model_name, metrics in ablation_results.items():
        print(
            f"{model_name}: "
            f"train_mae={metrics['train_mae']:.6f}, "
            f"mae={metrics['mae']:.6f}, "
            f"rmse={metrics['rmse']:.6f}, "
            f"cosine_similarity={metrics['cosine_similarity']:.6f}"
        )

    best_model_name = min(ablation_results, key=lambda name: ablation_results[name]["mae"])
    best_model = trained_models[best_model_name]
    print(f"\nBest model by MAE: {best_model_name}")

    checkpoint_path = outputs_dir / f"{best_model_name}.pth"
    torch.save(best_model.state_dict(), checkpoint_path)

    albedo_png_path = mesh_prediction_dir / "albedo_input.png"
    target_png_path = mesh_prediction_dir / "roughness_original.png"
    prediction_png_path = mesh_prediction_dir / "roughness_pred.png"
    preview_png_path = outputs_dir / "prediction_preview.png"
    export_prediction_artifacts(
        model=best_model,
        dataset=dataset,
        sample_index=sample_index,
        device=device,
        albedo_output_path=albedo_png_path,
        target_output_path=target_png_path,
        prediction_output_path=prediction_png_path,
        preview_output_path=preview_png_path,
    )

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved best model checkpoint to: {checkpoint_path}")
    print(f"Saved exported albedo map to: {albedo_png_path}")
    print(f"Saved original roughness map to: {target_png_path}")
    print(f"Saved predicted roughness map to: {prediction_png_path}")
    print(f"Saved preview image to: {preview_png_path}")


if __name__ == "__main__":
    main()
