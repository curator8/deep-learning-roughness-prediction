import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RoughnessUNet(nn.Module):
    """Basic encoder-decoder for albedo-to-roughness prediction."""

    def __init__(self, in_channels=3, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()

        self.encoder1 = EncoderBlock(in_channels, features[0])
        self.encoder2 = EncoderBlock(features[0], features[1])
        self.encoder3 = EncoderBlock(features[1], features[2])
        self.encoder4 = EncoderBlock(features[2], features[3])

        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        self.decoder4 = DecoderBlock(features[3] * 2, features[3], features[3])
        self.decoder3 = DecoderBlock(features[3], features[2], features[2])
        self.decoder2 = DecoderBlock(features[2], features[1], features[1])
        self.decoder1 = DecoderBlock(features[1], features[0], features[0])

        self.output_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        # Roughness is naturally bounded to [0, 1].
        return torch.sigmoid(self.output_layer(x))


class RoughnessAutoencoder(nn.Module):
    """Simpler encoder-decoder baseline without skip connections."""

    def __init__(self, in_channels=3, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.encoder1 = EncoderBlock(in_channels, features[0])
        self.encoder2 = EncoderBlock(features[0], features[1])
        self.encoder3 = EncoderBlock(features[1], features[2])
        self.encoder4 = EncoderBlock(features[2], features[3])

        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(features[3], features[3])
        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(features[2], features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(features[1], features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(features[0], features[0])

        self.output_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        _, x = self.encoder1(x)
        _, x = self.encoder2(x)
        _, x = self.encoder3(x)
        _, x = self.encoder4(x)

        x = self.bottleneck(x)
        x = self.dec4(self.up4(x))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))

        return torch.sigmoid(self.output_layer(x))


def predict_tiled_center(model, input_tensor, tile_repeat=3):
    """Predict with wrapped context and crop the center tile.

    This reduces visible seams when the output roughness map is repeated as a
    texture because the model sees opposite image edges next to each other.
    """
    if tile_repeat < 3 or tile_repeat % 2 == 0:
        raise ValueError("tile_repeat must be an odd integer greater than or equal to 3")

    _, _, height, width = input_tensor.shape
    center_index = tile_repeat // 2
    tiled_input = input_tensor.repeat(1, 1, tile_repeat, tile_repeat)
    tiled_prediction = model(tiled_input)

    top = center_index * height
    left = center_index * width
    return tiled_prediction[:, :, top:top + height, left:left + width]


def blend_seams(roughness_tensor, feather=24):
    """Blend opposite texture edges so repeated roughness maps tile cleanly."""
    _, _, height, width = roughness_tensor.shape
    feather = min(feather, height // 2, width // 2)
    if feather <= 0:
        return roughness_tensor

    blended = roughness_tensor.clone()
    weights = torch.linspace(
        0.0,
        1.0,
        feather,
        device=roughness_tensor.device,
        dtype=roughness_tensor.dtype,
    ).view(1, 1, 1, feather)

    left_band = roughness_tensor[:, :, :, :feather]
    right_band = torch.flip(roughness_tensor[:, :, :, -feather:], dims=[3])
    seam_band = left_band * weights + right_band * (1.0 - weights)
    blended[:, :, :, :feather] = seam_band
    blended[:, :, :, -feather:] = torch.flip(seam_band, dims=[3])

    weights = weights.transpose(2, 3)
    top_band = blended[:, :, :feather, :]
    bottom_band = torch.flip(blended[:, :, -feather:, :], dims=[2])
    seam_band = top_band * weights + bottom_band * (1.0 - weights)
    blended[:, :, :feather, :] = seam_band
    blended[:, :, -feather:, :] = torch.flip(seam_band, dims=[2])

    return blended


def predict_seamless_roughness(model, input_tensor, tile_repeat=3, feather=24):
    prediction = predict_tiled_center(model, input_tensor, tile_repeat=tile_repeat)
    return blend_seams(prediction, feather=feather)


def train_one_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    sample_count = 0

    for albedo_batch, roughness_batch in dataloader:
        albedo_batch = albedo_batch.to(device)
        roughness_batch = roughness_batch.to(device)

        predicted_roughness = model(albedo_batch)
        loss = torch.nn.functional.smooth_l1_loss(predicted_roughness, roughness_batch)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = albedo_batch.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / sample_count


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()

    total_mae = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    total_samples = 0

    for albedo_batch, roughness_batch in dataloader:
        albedo_batch = albedo_batch.to(device)
        roughness_batch = roughness_batch.to(device)

        predicted_roughness = model(albedo_batch)

        mae = torch.mean(torch.abs(predicted_roughness - roughness_batch))
        mse = torch.mean((predicted_roughness - roughness_batch) ** 2)

        pred_flat = predicted_roughness.flatten(start_dim=1)
        target_flat = roughness_batch.flatten(start_dim=1)
        cosine = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1).mean()

        batch_size = albedo_batch.size(0)
        total_mae += mae.item() * batch_size
        total_mse += mse.item() * batch_size
        total_cosine += cosine.item() * batch_size
        total_samples += batch_size

    average_mae = total_mae / total_samples
    average_mse = total_mse / total_samples

    return {
        "mae": average_mae,
        "mse": average_mse,
        "rmse": average_mse ** 0.5,
        "cosine_similarity": total_cosine / total_samples,
    }


def fit_model(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    device,
    epochs,
    scheduler=None,
    train_eval_dataloader=None,
    epoch_callback=None,
):
    history = []
    best_state = None
    best_mae = float("inf")
    train_eval_dataloader = train_eval_dataloader or train_dataloader

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        train_metrics = evaluate_model(model, train_eval_dataloader, device)
        eval_metrics = evaluate_model(model, eval_dataloader, device)

        if scheduler is not None:
            scheduler.step(eval_metrics["mae"])

        if eval_metrics["mae"] < best_mae:
            best_mae = eval_metrics["mae"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        metrics = {
            "train_loss": train_loss,
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_cosine_similarity": train_metrics["cosine_similarity"],
            "test_mae": eval_metrics["mae"],
            "test_rmse": eval_metrics["rmse"],
            "test_cosine_similarity": eval_metrics["cosine_similarity"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        if epoch_callback is not None:
            epoch_callback(metrics)

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def show_prediction(model, dataset, index=0, device="cpu"):
    import matplotlib.pyplot as plt

    model.eval()

    albedo_tensor, target_roughness = dataset[index]
    input_tensor = albedo_tensor.unsqueeze(0).to(device)
    predicted_roughness = model(input_tensor).squeeze(0).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(albedo_tensor.permute(1, 2, 0))
    axes[0].set_title(f"Albedo: {dataset.sample_name(index)}")
    axes[0].axis("off")

    axes[1].imshow(target_roughness.squeeze(0), cmap="gray")
    axes[1].set_title("Target Roughness")
    axes[1].axis("off")

    axes[2].imshow(predicted_roughness.squeeze(0), cmap="gray")
    axes[2].set_title("Predicted Roughness")
    axes[2].axis("off")

    plt.tight_layout()
