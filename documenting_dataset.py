import random
from io import BytesIO

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


splits = {
    "train": "hf://datasets/myvision/gender-classification/data/train-00000-of-00001.parquet",
    "test": "hf://datasets/myvision/gender-classification/data/test-00000-of-00001.parquet",
    "eval": "hf://datasets/myvision/gender-classification/data/eval-00000-of-00001.parquet",
}

train_df = pd.read_parquet(splits["train"])


sample_row = train_df.iloc[random.randint(0, len(train_df) - 1)]
img_bytes = sample_row["image"]
if isinstance(img_bytes, dict):
    img_bytes = img_bytes.get("bytes") or img_bytes.get("data")
img = Image.open(BytesIO(img_bytes)).convert("RGB")  # type: ignore[call-arg]
img_np = np.array(img)


transform_dict = {
    "Rotate": A.Compose([A.Rotate(limit=30, p=1.0)]),
    "ShiftScale": A.Compose(
        [A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=1.0)]
    ),
    "Flip": A.Compose([A.HorizontalFlip(p=1.0)]),
    "GaussNoise": A.Compose([A.GaussNoise(var_limit=(10, 50), p=1.0)]),  # type: ignore[call-arg]
    "Brightness": A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)]
    ),
    "HSV": A.Compose(
        [
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=1.0
            )
        ]
    ),
    "CoarseDropout": A.Compose(
        [
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(20, 20),
                hole_width_range=(20, 20),
                fill_value=128,
                p=1.0,
            )  # type: ignore
        ]
    ),
    "Blur": A.Compose([A.Blur(blur_limit=3, p=1.0)]),
}


def show_augmented(original, augmented, title):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(original)
    axs[0].set_title("Original")
    axs[0].axis("off")
    axs[1].imshow(augmented)
    axs[1].set_title(title)
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()


for name, tf in transform_dict.items():
    aug = tf(image=img_np)["image"]
    show_augmented(img_np, aug, name)
