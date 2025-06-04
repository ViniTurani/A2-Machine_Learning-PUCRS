import fastapi
import uvicorn
import torch

from model_training import predict_image
from model_training import SimpleCNN, transform
import os
from fastapi import HTTPException

app = fastapi.FastAPI()


class_names = ["woman", "man"]


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict():
    return predict_image(
        image_path="path/to/image.jpg",
        model=load_model(),
        transform=transform,
        class_names=class_names,
    )


def load_model() -> SimpleCNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("checkpoint.pth"):
        raise HTTPException(status_code=404, detail="Model checkpoint not found")

    model: SimpleCNN = torch.load("checkpoint.pth", map_location=device)
    return model


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
