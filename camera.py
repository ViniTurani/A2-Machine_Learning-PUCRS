"""
Webcam snapshot → gender prediction

Run with:
    python camera.py
Press SPACE to take a picture, ESC to quit.
"""

import cv2
from PIL import Image
import torch
from torchvision import transforms

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Model skeleton + weights
# ──────────────────────────────────────────────────────────────────────────────
from model_training import SimpleCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoint.pth"  # path to the .pth you saved after training
CLASS_NAMES = ["woman", "man"]  # same order as during training
CAM_INDEX = 0  # change if you have multiple webcams
SNAPSHOT_KEY = 32  # space bar
EXIT_KEY = 27  # esc key

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# Instantiate architecture and load weights
model = SimpleCNN().to(DEVICE)
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state["model_state_dict"])
model.eval()  # important: turn off dropout / batch-norm update


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Helper: classify a single OpenCV frame
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_from_frame(frame_bgr) -> str:
    """Return 'man' or 'woman' for an OpenCV BGR frame."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    img_t = transform(pil_img).unsqueeze(0).to(DEVICE)

    logits = model(img_t)
    _, pred = torch.max(logits, 1)
    return CLASS_NAMES[pred.item()]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Webcam main loop
# ──────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise IOError(f"❌  Cannot open webcam {CAM_INDEX}")

print("🔴  Live feed – press SPACE to classify, ESC to quit")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == EXIT_KEY:  # ESC
            break

        if key == SNAPSHOT_KEY:  # SPACE
            label = predict_from_frame(frame)
            print("Prediction:", label)

            # Overlay the result and show for 1 second
            show = frame.copy()
            cv2.putText(
                show,
                f"Prediction: {label}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Webcam", show)
            cv2.waitKey(1000)
finally:
    print("🔴  Quitting...")
    cap.release()
    cv2.destroyAllWindows()
