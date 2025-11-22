"""task2_cnn_mnist.py

Deep learning model for image classification on MNIST using PyTorch.

- Downloads MNIST dataset
- Trains a small CNN
- Visualizes training loss curve

Run:
    python task2_cnn_mnist.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 3
LR = 1e-3


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def get_dataloaders(batch_size=BATCH_SIZE):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        test_acc = evaluate(model, test_loader)
        print(
            f"[Epoch {epoch+1}/{EPOCHS}] "
            f"Train loss: {avg_loss:.4f} | Test accuracy: {test_acc:.4f}"
        )

    return train_losses


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def plot_training_curve(losses, out_path):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve (MNIST CNN)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    print("[Task 2] Training CNN on MNIST...")
    train_loader, test_loader = get_dataloaders()
    model = SimpleCNN().to(DEVICE)

    losses = train_model(model, train_loader, test_loader)

    os.makedirs("outputs", exist_ok=True)
    plot_path = os.path.join("outputs", "training_loss_curve.png")
    plot_training_curve(losses, plot_path)
    print(f"[Task 2] Saved training loss curve to: {plot_path}")


if __name__ == "__main__":
    main()
