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
    predict_seamless_roughness,
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
    predicted_roughness = predict_seamless_roughness(
        model,
        albedo_tensor.unsqueeze(0).to(device),
    ).squeeze(0).cpu()

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
    train_dataloader: DataLoader,
    train_eval_dataloader: DataLoader,
    test_dataloader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    model = model_class().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )
    print(f"\nRunning experiment: {model_name}", flush=True)

    def print_epoch(row: dict[str, float]) -> None:
        print(
            f"Epoch {row['epoch']}/{epochs} - "
            f"train_loss={row['train_loss']:.6f}, "
            f"train_mae={row['train_mae']:.6f}, "
            f"test_mae={row['test_mae']:.6f}, "
            f"test_rmse={row['test_rmse']:.6f}, "
            f"test_cosine_similarity={row['test_cosine_similarity']:.6f}, "
            f"lr={row['learning_rate']:.2e}",
            flush=True,
        )

    history = fit_model(
        model=model,
        train_dataloader=train_dataloader,
        train_eval_dataloader=train_eval_dataloader,
        eval_dataloader=test_dataloader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        scheduler=scheduler,
        epoch_callback=print_epoch,
    )

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train roughness prediction models, run ablations, and export a predicted map."
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--sample-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    train_data_dir = project_root / "data" / "train"
    test_data_dir = project_root / "data" / "test"
    outputs_dir = project_root / "outputs"
    mesh_prediction_dir = (
        project_root
        / "simple-textured-mesh"
        / "static"
        / "predictions"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = PairedTextureDataset(
        root_dir=train_data_dir,
        image_size=(args.image_size, args.image_size),
        augment=not args.no_augment,
    )
    train_eval_dataset = PairedTextureDataset(
        root_dir=train_data_dir,
        image_size=(args.image_size, args.image_size),
        augment=False,
    )
    test_dataset = PairedTextureDataset(
        root_dir=test_data_dir,
        image_size=(args.image_size, args.image_size),
        augment=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    train_eval_dataloader = DataLoader(
        train_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train dataset size: {len(train_dataset)} texture pairs")
    print(f"Test dataset size: {len(test_dataset)} texture pairs")
    print(
        "Augmentation: "
        f"{'disabled' if args.no_augment else 'random crop, flips, 90-degree rotations, light albedo color jitter'}"
    )

    if args.sample_index is not None:
        sample_index = args.sample_index
        export_dataset = test_dataset
    elif args.sample_name is not None:
        test_sample_names = [sample["name"] for sample in test_dataset.samples]
        train_sample_names = [sample["name"] for sample in train_eval_dataset.samples]
        if args.sample_name in test_sample_names:
            export_dataset = test_dataset
            sample_index = test_sample_names.index(args.sample_name)
        elif args.sample_name in train_sample_names:
            export_dataset = train_eval_dataset
            sample_index = train_sample_names.index(args.sample_name)
        else:
            available_samples = sorted(test_sample_names + train_sample_names)
            raise ValueError(
                f"Sample name '{args.sample_name}' not found. Available samples: {available_samples}"
            )
    else:
        export_dataset = test_dataset
        sample_index = 0

    print(f"Export sample: {export_dataset.sample_name(sample_index)}")

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
            train_dataloader=train_dataloader,
            train_eval_dataloader=train_eval_dataloader,
            test_dataloader=test_dataloader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        trained_models[model_name] = model
        ablation_results[model_name] = min(history, key=lambda row: row["test_mae"])

    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = outputs_dir / "ablation_results.json"
    metrics_path.write_text(json.dumps(ablation_results, indent=2))

    checkpoint_paths = {}
    for model_name, trained_model in trained_models.items():
        checkpoint_paths[model_name] = outputs_dir / f"{model_name}.pth"
        torch.save(trained_model.state_dict(), checkpoint_paths[model_name])

    print("\nAblation summary:")
    for model_name, metrics in ablation_results.items():
        print(
            f"{model_name}: "
            f"train_mae={metrics['train_mae']:.6f}, "
            f"test_mae={metrics['test_mae']:.6f}, "
            f"test_rmse={metrics['test_rmse']:.6f}, "
            f"test_cosine_similarity={metrics['test_cosine_similarity']:.6f}, "
            f"best_epoch={metrics['epoch']}"
        )

    best_model_name = min(ablation_results, key=lambda name: ablation_results[name]["test_mae"])
    best_model = trained_models[best_model_name]
    print(f"\nBest model by test MAE: {best_model_name}")

    checkpoint_path = checkpoint_paths[best_model_name]

    albedo_png_path = mesh_prediction_dir / "albedo_input.png"
    target_png_path = mesh_prediction_dir / "roughness_original.png"
    prediction_png_path = mesh_prediction_dir / "roughness_pred.png"
    preview_png_path = outputs_dir / "prediction_preview.png"
    export_prediction_artifacts(
        model=best_model,
        dataset=export_dataset,
        sample_index=sample_index,
        device=device,
        albedo_output_path=albedo_png_path,
        target_output_path=target_png_path,
        prediction_output_path=prediction_png_path,
        preview_output_path=preview_png_path,
    )

    print(f"Saved metrics to: {metrics_path}")
    for model_name, saved_checkpoint_path in checkpoint_paths.items():
        print(f"Saved {model_name} checkpoint to: {saved_checkpoint_path}")
    print(f"Best model checkpoint: {checkpoint_path}")
    print(f"Saved exported albedo map to: {albedo_png_path}")
    print(f"Saved original roughness map to: {target_png_path}")
    print(f"Saved predicted roughness map to: {prediction_png_path}")
    print(f"Saved preview image to: {preview_png_path}")


if __name__ == "__main__":
    main()
