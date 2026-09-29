import io
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

from cnn_project.predict import DEFAULT_MODEL_PATH, predict_top_k

state = {}


@asynccontextmanager
async def lifespan(app):
    # بارگذاری مدل فقط یک بار هنگام شروع سرور
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    state["model"] = keras.models.load_model(model_path)
    yield
    state.clear()


app = FastAPI(title="CIFAR-10 CNN API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()], top_k: int = 3):
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    return {"predictions": predict_top_k(state["model"], image, top_k)}
