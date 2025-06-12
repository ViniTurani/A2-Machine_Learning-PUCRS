from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from PIL import Image


def plot_confusion_matrix(
    model,
    loader,
    class_names,
    device,
    normalize: bool = False,
    filename: str = "apresentacao/confusion_matrix.png",
):
    """
    Evaluate *model* on *loader* and write a labelled confusion‑matrix PNG.

    If `normalize=True`, each row (true class) is converted to percentages.
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # ── Plot ────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="white")
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def imshow(img):
    img = img * 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()


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
    plt.savefig(
        f"apresentacao/feature_maps_{tag}.png", facecolor="white", bbox_inches="tight"
    )
    plt.close(fig)


def predict_image(
    image_path,
    model,
    transform,
    device,
    class_names=None,
):
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
