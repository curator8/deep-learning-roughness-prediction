import torch
from torch import nn
import matplotlib.pyplot as plt


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


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0

    for albedo_batch, roughness_batch in dataloader:
        albedo_batch = albedo_batch.to(device)
        roughness_batch = roughness_batch.to(device)

        predicted_roughness = model(albedo_batch)
        loss = torch.nn.functional.l1_loss(predicted_roughness, roughness_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()

    total_mae = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    total_batches = 0

    for albedo_batch, roughness_batch in dataloader:
        albedo_batch = albedo_batch.to(device)
        roughness_batch = roughness_batch.to(device)

        predicted_roughness = model(albedo_batch)

        mae = torch.mean(torch.abs(predicted_roughness - roughness_batch))
        mse = torch.mean((predicted_roughness - roughness_batch) ** 2)

        pred_flat = predicted_roughness.flatten(start_dim=1)
        target_flat = roughness_batch.flatten(start_dim=1)
        cosine = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1).mean()

        total_mae += mae.item()
        total_mse += mse.item()
        total_cosine += cosine.item()
        total_batches += 1

    average_mae = total_mae / total_batches
    average_mse = total_mse / total_batches

    return {
        "mae": average_mae,
        "mse": average_mse,
        "rmse": average_mse ** 0.5,
        "cosine_similarity": total_cosine / total_batches,
    }


def fit_model(model, dataloader, optimizer, device, epochs):
    history = []

    for epoch in range(epochs):
        train_mae = train_one_epoch(model, dataloader, optimizer, device)
        metrics = evaluate_model(model, dataloader, device)
        metrics["train_mae"] = train_mae
        metrics["epoch"] = epoch + 1
        history.append(metrics)

    return history


@torch.no_grad()
def show_prediction(model, dataset, index=0, device="cpu"):
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
