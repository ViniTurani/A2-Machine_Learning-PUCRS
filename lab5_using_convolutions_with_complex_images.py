# %%
# Install dependencies (uncomment if running for the first time)
# !pip install -r ./requirements.txt

# %%
# Imports

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# %%
# Configs

SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Speed boost for fixed input sizes

# %%
# Data transformations
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# %%
# Dataset paths
splits = {
    "train": "hf://datasets/myvision/gender-classification/data/train-00000-of-00001.parquet",
    "test": "hf://datasets/myvision/gender-classification/data/test-00000-of-00001.parquet",
    "eval": "hf://datasets/myvision/gender-classification/data/eval-00000-of-00001.parquet",
}

# %%
# Load datasets
train_df = pd.read_parquet(splits["train"])
test_df = pd.read_parquet(splits["test"])
eval_df = pd.read_parquet(splits["eval"])

num_classes = train_df["label"].nunique()

print(f"Detected {num_classes} classes.")

# %%
# Custom Dataset


class GenderDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_bytes = self.dataframe.iloc[idx]["image"]["bytes"]
        label = self.dataframe.iloc[idx]["label"]

        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# %%
# Dataloaders

train_dataset = GenderDataset(train_df, transform=transform)
test_dataset = GenderDataset(test_df, transform=transform)
eval_dataset = GenderDataset(eval_df, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

# %%
# Visualize


def imshow(img):
    img = img * 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()


dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images[:8]))
print("Labels:", labels.tolist())

# %%
# Model


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


model = SimpleCNN().to(device)

# %%
# Loss, Optimizer, Scheduler

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1,
)

# %%
# Training Loop

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        images, labels = images.to(device), labels.to(device)

        # === BACKPROP + GRADIENT DESCENT ===
        optimizer.zero_grad()  # 🔥 Zero previous gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # 🔥 Backpropagation: compute gradients
        optimizer.step()  # 🔥 Gradient Descent: update weights
        # ====================================

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    scheduler.step()

    # Eval on validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Validation Accuracy={val_acc:.2f}%")

print("Finished Training")

# %%
# Test Accuracy

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# %%
# Save Model

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": EPOCHS,
    },
    "checkpoint.pth",
)

print("Model saved as checkpoint.pth")

# %%
