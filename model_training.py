import os
import pathlib
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

# %%
# Configs

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001

LOAD_CHECKPOINT = True
CHECKPOINT_PATH = "checkpoint.pth"

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A small speed‑up when input sizes are fixed
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# %%
# Data-augmentation pipeline
def add_gaussian_noise(std: float = 0.05):
    """Return a transform that adds zero-mean Gaussian noise in tensor space."""
    return transforms.Lambda(lambda x: x + torch.randn_like(x) * std)


transform_train = transforms.Compose(
    [
        # 1) Geometric distortions (operate on PIL images)
        transforms.RandomApply([transforms.RandomRotation(45)], p=0.5),
        transforms.RandomApply(
            [transforms.RandomAffine(0, translate=(0.10, 0.10))], p=0.5
        ),
        transforms.RandomApply([transforms.RandomAffine(0, scale=(0.9, 1.1))], p=0.5),
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    0,
                    shear=15,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=128,
                )
            ],
            p=0.5,
        ),
        # random horizontal/vertical flips
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        # photometric distortions (still on PIL images)
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04  # ±30 %
                )
            ]
        ),
        # convert to tensor
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # tensor-space augmentations
        transforms.RandomApply([add_gaussian_noise(std=0.05)], p=0.5),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
        # normalisation
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

# Validation / test use the deterministic pipeline only
transform_eval = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
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

print("Train set class distribution:\n", train_df["label"].value_counts())
print("Test set class distribution:\n", test_df["label"].value_counts())
print("Eval set class distribution:\n", eval_df["label"].value_counts())
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
train_loader = DataLoader(
    GenderDataset(train_df, transform=transform_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

test_loader = DataLoader(
    GenderDataset(test_df, transform=transform_eval),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
eval_loader = DataLoader(
    GenderDataset(eval_df, transform=transform_eval),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


# %%
# Visualize a mini‑batch (optional)
def imshow(img):
    img = img * 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()


# %%
# Model definition


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 64×64  -> 32×32
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 32×32  -> 16×16
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 16×16  ->  8×8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #  8×8  ->  4×4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # #  4×4  ->  2×2
            # nn.Conv2d(128, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))


model = SimpleCNN().to(device)

# ──────────────────────────────────────────────────────────────────────────────
# Hooks to capture intermediate feature maps
# ──────────────────────────────────────────────────────────────────────────────
from typing import Dict

feature_maps: Dict[str, torch.Tensor] = {}


def _hook(_, __, output):
    # store the latest activation tensor on every forward pass
    feature_maps["latest"] = output.detach()


# register the hook on the whole conv stack
hook_handle = model.conv_layers.register_forward_hook(_hook)


def save_feature_maps(
    feat_tensor: torch.Tensor,
    img_tensor: torch.Tensor,
    labels: torch.Tensor,
    tag: str = "train",
) -> None:
    """
    Visualise the *first* occurrence of each class (0 = woman, 1 = man) in a batch.

    • Two columns per row:
        left  - original RGB image (unnormalised)
        right - mean-over-channels feature map from the CNN

    • At most one row per class → max 2 rows total.
    • Writes   feature_maps_<tag>.png   in the working directory.

    Parameters
    ----------
    feat_tensor : torch.Tensor  (B, C, H, W)  CNN activations from a forward hook
    img_tensor  : torch.Tensor  (B, 3, H, W)  input images *after* transform
    labels      : torch.Tensor  (B,)          integer class labels
    tag         : str                          identifier used in the filename
    """

    # indices of the first example of each class
    idx_0 = (labels == 0).nonzero(as_tuple=True)[0]
    idx_1 = (labels == 1).nonzero(as_tuple=True)[0]
    selected = []
    if len(idx_0) > 0:
        selected.append(idx_0[0].item())
    if len(idx_1) > 0:
        selected.append(idx_1[0].item())
    if not selected:
        return  # nothing to plot in this batch

    n_rows = len(selected)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(6, 3 * n_rows), squeeze=False, facecolor="white"
    )

    for row, idx in enumerate(selected):
        # ---- Original image (undo normalisation) ----
        img = img_tensor[idx].cpu() * 0.5 + 0.5  # revert [-1,1] → [0,1]
        np_img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        axes[row, 0].imshow(np_img)
        axes[row, 0].set_title(f"Label {labels[idx].item()} – original")
        axes[row, 0].axis("off")

        # ---- Feature map (mean over channels) ----
        feat_img = feat_tensor[idx].mean(0).cpu()  # (H, W)
        axes[row, 1].imshow(feat_img, cmap="viridis")
        axes[row, 1].set_title("Feature map")
        axes[row, 1].axis("off")

    plt.suptitle(f"{tag}: first examples of each class\n(0 = woman, 1 = man)")
    plt.tight_layout()
    plt.savefig(f"feature_maps_{tag}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


# %%
# Optional: load pretrained weights
SKIP_TRAINING = False
if LOAD_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"✅ Loaded weights from {CHECKPOINT_PATH}")
    model.eval()
    SKIP_TRAINING = True
else:
    print("🛈 No checkpoint found – training from scratch.")

# %%
# Loss, Optimizer, Scheduler (define regardless so they are available later)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# If we loaded a checkpoint, restore optimizer/scheduler to resume training later
if LOAD_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

# %%
# Training loop (skipped if SKIP_TRAINING is True)
if not SKIP_TRAINING:
    epoch_losses, epoch_val_acc = [], []
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if epoch == 0 and batch_idx == 0:
                save_feature_maps(
                    feat_tensor=feature_maps["latest"],  # captured by your forward hook
                    img_tensor=images,  # current batch of inputs
                    labels=labels,  # ground-truth labels
                    tag="train",
                )
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        epoch_val_acc.append(val_acc)
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.2f}%")

    if epoch_losses and epoch_val_acc:
        fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
        # Loss curve
        ax_loss.plot(epoch_losses, label="Loss")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # Accuracy curve
        ax_acc.plot(epoch_val_acc, label="Val Acc", color="tab:orange")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title("Validation Accuracy")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("loss_accuracy.png")
        plt.close()

    # Save the checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": EPOCHS,
        },
        CHECKPOINT_PATH,
    )
    print(f"💾 Model saved to {CHECKPOINT_PATH}")
else:
    print("➡️  Skipping training (checkpoint mode)")

# %%
# Test Accuracy (always run to verify performance)
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")


# %%
# Prediction helper
def predict_image(image_path, model, transform, class_names=None, device=device):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
    pred_idx = predicted.item()
    if class_names:
        return class_names[pred_idx], conf.item()
    return pred_idx, conf.item()


# %%
def dataset_mean_std(loader):
    n, mean, M2 = 0, torch.zeros(3), torch.zeros(3)
    for x, _ in loader:
        bs = x.size(0)
        x = x.view(bs, 3, -1)
        mean += x.mean(2).sum(0)
        M2 += x.var(2, unbiased=False).sum(0)
        n += bs
    mean /= n
    std = (M2 / n).sqrt()
    return mean, std


# remove hook to free resources
hook_handle.remove()


if __name__ == "__main__":
    train_mean, train_std = dataset_mean_std(train_loader)
    print("\n==== DATASET STATISTICS ====")
    print("TRAIN  mean:", train_mean, "\n       std :", train_std)

    cam_dir = pathlib.Path("cam_tensors")
    cam_files = list(cam_dir.glob("*.pt"))
    if cam_files:
        cam_stack = torch.cat([torch.load(p) for p in cam_files])
        cam_mean = cam_stack.mean([0, 2, 3])
        cam_std = cam_stack.std([0, 2, 3])
        print("\n==== WEBCAM STATISTICS ====")
        print("CAMERA mean:", cam_mean, "\n       std :", cam_std)
        del cam_stack
    else:
        print(
            "\n(no cam_tensors/*.pt found – run camera.py and press SPACE a few times)"
        )
