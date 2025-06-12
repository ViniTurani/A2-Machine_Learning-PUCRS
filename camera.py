import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

from model_training import SimpleCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoint.pth"
CLASS_NAMES = ["woman", "man"]

CAM_INDEX = 0
SNAPSHOT_KEY = 32
EXIT_KEY = 27
ZOOM_FACTOR = 3

# transform igual ao de validacao
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

model = SimpleCNN().to(DEVICE)
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state["model_state_dict"])
model.eval()  # desliga dropout e batch-norm em inferência

# ----------------------------------------------------------------------
demo_feature_saved = False
feature_maps_demo = {}


def _hook_cam(_, __, output):
    feature_maps_demo["latest"] = output.detach()


hook_handle_cam = model.conv_layers.register_forward_hook(_hook_cam)

# buffer para armazenar os dois primeiros frames e suas feature maps
demo_buffer: list[tuple[np.ndarray, torch.Tensor, str]] = []  # (orig RGB, feat tensor)


def save_demo_feature_maps(
    frame_bgr: np.ndarray, feat_tensor: torch.Tensor, prediction: str
) -> None:
    """
    Coleta os dois primeiros frames da webcam e salva feature_maps_demo.png.

    • Cada linha (máximo 2 linhas):
        esquerda - frame RGB original após zoom
        direita  - mapa de características médio (sobre canais) da pilha de convolução

    Frames subsequentes são ignorados após o gráfico ser salvo.
    """
    global demo_buffer

    rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    demo_buffer.append((rgb_img, feat_tensor.squeeze(0).cpu(), prediction))
    if len(demo_buffer) < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(6, 6), squeeze=False, facecolor="white")
    for row, (rgb, feat, pred) in enumerate(demo_buffer[:2]):
        # original frame
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"Frame {row+1} - pred: {pred}")
        axes[row, 0].axis("off")

        # mean feature map
        axes[row, 1].imshow(feat.mean(0), cmap="viridis")
        axes[row, 1].set_title(f"Feature map - pred: {pred}")
        axes[row, 1].axis("off")

    plt.suptitle("Demo: first two frames")
    plt.tight_layout()
    plt.savefig(
        "apresentacao/feature_maps_demo.png", facecolor="white", bbox_inches="tight"
    )
    plt.close(fig)

    demo_buffer = demo_buffer[:2]


def digital_zoom(frame_bgr, factor: float = 1.0):
    """
    Simula zoom óptico recortando o centro do frame e redimensionando.

    factor = 1.0 → sem zoom • 2.0 → 2x (corta 50 % do frame)
    """
    if factor <= 1.0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    new_h, new_w = int(h / factor), int(w / factor)
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    crop = frame_bgr[y1 : y1 + new_h, x1 : x1 + new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------
@torch.no_grad()
def predict_from_frame(frame_bgr, threshold: float = 0.50) -> str:
    """
    Retorna 'man', 'woman' ou 'inconclusive' para um frame BGR.

    Se a probabilidade máxima < threshold, devolve 'inconclusive'.
    """
    # aplica zoom antes de qualquer outra etapa
    frame_bgr = digital_zoom(frame_bgr, factor=ZOOM_FACTOR)

    # BGR → PIL RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_t = transform(pil_img).unsqueeze(0).to(DEVICE)  # type: ignore

    logits = model(img_t)

    probs = torch.softmax(logits, dim=1)
    max_prob, pred = torch.max(probs, 1)

    # save feature map for the first demo image only once
    if "latest" in feature_maps_demo:
        save_demo_feature_maps(frame_bgr, feature_maps_demo["latest"], CLASS_NAMES[pred.item()])  # type: ignore

    print(f"Probabilities: {probs.cpu().numpy()}")
    print(
        f"Max probability: {max_prob.item():.2f} for class '{CLASS_NAMES[pred.item()]}'"  # type: ignore
    )
    if max_prob.item() < threshold:
        return "inconclusive"
    return CLASS_NAMES[pred.item()]  # type: ignore


# ----------------------------------------------------------------------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # resolução “base”
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise IOError(f"❌ Cannot open webcam {CAM_INDEX}")

print("🔴 Live feed - press SPACE to classify, ESC to quit")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # aplica zoom apenas para visualização em tempo real
        show_frame = digital_zoom(frame, factor=ZOOM_FACTOR)
        cv2.imshow("Webcam", show_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == EXIT_KEY:  # ESC
            break
        if key == SNAPSHOT_KEY:  # SPACE
            label = predict_from_frame(frame)
            print("Prediction:", label)

            # exibe resultado sobreposto por 3 s
            annotated = show_frame.copy()
            cv2.putText(
                annotated,
                f"Prediction: {label}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Webcam", annotated)
            cv2.waitKey(3000)
finally:
    print("🔴  Quitting...")
    hook_handle_cam.remove()
    cap.release()
    cv2.destroyAllWindows()
