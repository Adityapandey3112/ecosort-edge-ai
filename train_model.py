import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Class definitions for the EcoSort Dataset
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block:
    1. Depthwise Conv: Applies spatial filtering per input channel independently.
    2. Pointwise Conv: Combines features across channels using 1x1 convolutions.
    Keeps total parameter count very low while maintaining feature quality.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise step
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise step
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EcoSortClassifier(nn.Module):
    """
    Lightweight CNN designed under the 350,000 parameter limit (~346.8k params).
    Input: [B, 3, 96, 96] -> Output: [B, 6]
    """
    def __init__(self, num_classes=6):
        super().__init__()
        # Initial Conv Stem (96x96 -> 48x48)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Feature Extraction Backbone
        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),   # 48x48 -> 24x24
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),  # 24x24 -> 12x12
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 320, stride=2),  # 12x12 -> 6x6
            DepthwiseSeparableConv(320, 320, stride=1),
        )
        
        # Global Pooling & Linear Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(320, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_to_onnx(model, output_path="model.onnx"):
    """
    Exports the trained PyTorch model to ONNX format.
    Ensures input tensor is [1, 3, 96, 96] and output is [1, 6].
    """
    model.eval()
    dummy_input = torch.randn(1, 3, 96, 96)
    print(f"\nExporting model to '{output_path}'...")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX exported successfully! File Size: {size_mb:.2f} MB")


def main(args=None):
    parser = argparse.ArgumentParser(description="Train EcoSort Edge AI Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output path for ONNX file")
    parsed_args = parser.parse_args(args)

    # Device Setup (CUDA GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Instantiation & Parameter Verification
    model = EcoSortClassifier(num_classes=len(CLASSES)).to(device)
    params = count_parameters(model)
    print(f"Model Parameters: {params:,} / 350,000 limit")

    # Image Transforms (Resizing to 96x96 and ImageNet Normalization)
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loading Datasets
    train_path = os.path.join(parsed_args.data_dir, "train")
    val_path = os.path.join(parsed_args.data_dir, "val")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Ensure '{parsed_args.data_dir}' contains both 'train' and 'val' subfolders.")

    train_loader = DataLoader(
        datasets.ImageFolder(train_path, transform=train_transform),
        batch_size=parsed_args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        datasets.ImageFolder(val_path, transform=val_transform),
        batch_size=parsed_args.batch_size, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=parsed_args.lr, weight_decay=1e-4)

    # Training Loop
    print("\nStarting Training...")
    for epoch in range(1, parsed_args.epochs + 1):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        # Validation Loop
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total += labels.size(0)

        train_acc = 100 * correct / total if total > 0 else 0
        val_acc = 100 * v_correct / v_total if v_total > 0 else 0
        print(f"Epoch [{epoch:02d}/{parsed_args.epochs:02d}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Export Trained Model to ONNX
    export_to_onnx(model.cpu(), output_path=parsed_args.output)


if __name__ == "__main__":
    # If running inside a Jupyter/Colab notebook with no sys.argv parameters,
    # default to the extracted dataset path.
    import sys
    if len(sys.argv) <= 1 or "colab_kernel_launcher" in sys.argv[0]:
        main(["--data_dir", "/content/ecosort_dataset_public"])
    else:
        main()
