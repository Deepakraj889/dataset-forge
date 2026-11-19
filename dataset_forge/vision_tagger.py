import os
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import onnxruntime as ort
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn

HF_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
MODEL_URL = f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/model.onnx"
TAGS_URL = f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/selected_tags.csv"

def get_cache_dir() -> Path:
    cache = Path.home() / ".cache" / "dataset_forge" / "models" / "wd-v1-4-convnext-tagger-v2"
    cache.mkdir(parents=True, exist_ok=True)
    return cache

def download_file_with_progress(url: str, dest_path: Path, description: str):
    """Downloads a file with an interactive Rich progress bar."""
    req = urllib.request.Request(url, headers={"User-Agent": "DatasetForge/0.1.0"})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress:
            task = progress.add_task(description, total=total_size)
            chunk_size = 1024 * 1024  # 1MB
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

def ensure_model_downloaded() -> Tuple[Path, Path]:
    """Ensures model.onnx and selected_tags.csv are cached locally; downloads if missing."""
    cache = get_cache_dir()
    model_path = cache / "model.onnx"
    tags_path = cache / "selected_tags.csv"

    if not tags_path.exists():
        download_file_with_progress(TAGS_URL, tags_path, "Downloading WD14 Tag Dictionary...")

    if not model_path.exists():
        download_file_with_progress(MODEL_URL, model_path, "Downloading WD14 Vision Tagger Model (~370MB)...")

    return model_path, tags_path

class WD14Tagger:
    def __init__(self, model_path: Optional[Path] = None, tags_path: Optional[Path] = None):
        if model_path is None or tags_path is None:
            m_path, t_path = ensure_model_downloaded()
            model_path = model_path or m_path
            tags_path = tags_path or t_path

        # Setup ONNX inference session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available_providers = ort.get_available_providers()
        active_providers = [p for p in providers if p in available_providers]

        self.session = ort.InferenceSession(str(model_path), providers=active_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.target_size = self.input_shape[1] if len(self.input_shape) >= 2 and isinstance(self.input_shape[1], int) else 448

        # Load and parse tags dataframe
        df = pd.read_csv(tags_path)
        self.tag_names = df["name"].tolist()
        # category 0 = general, category 4 = character, category 9 = rating
        self.categories = df["category"].tolist() if "category" in df.columns else [0] * len(self.tag_names)

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """Loads and pre-processes image with aspect-ratio preserving white padding to target_size."""
        with Image.open(image_path) as img:
            img = img.convert("RGBA")
            # Create white canvas for padding
            canvas = Image.new("RGBA", img.size, (255, 255, 255))
            canvas.alpha_composite(img)
            img = canvas.convert("RGB")

            # Square pad while preserving aspect ratio
            w, h = img.size
            max_dim = max(w, h)
            padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            padded.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))

            # Resize to model input size (usually 448x448)
            resized = padded.resize((self.target_size, self.target_size), Image.BICUBIC)

            # Convert to BGR array as required by ConvNeXt WD14 taggers
            arr = np.array(resized, dtype=np.float32)
            arr = arr[:, :, ::-1]  # RGB to BGR
            arr = np.expand_dims(arr, axis=0)
            return arr

    def predict(
        self,
        image_path: Path,
        general_threshold: float = 0.35,
        character_threshold: float = 0.80
    ) -> List[str]:
        """Runs visual inference on an image and returns predicted Danbooru tags."""
        try:
            tensor = self.preprocess_image(image_path)
            outputs = self.session.run(None, {self.input_name: tensor})
            probs = outputs[0][0]

            # If model outputs raw logits instead of probabilities, apply sigmoid
            if np.min(probs) < 0.0 or np.max(probs) > 1.0:
                probs = 1.0 / (1.0 + np.exp(-probs))

            predicted_tags: List[str] = []
            for name, cat, prob in zip(self.tag_names, self.categories, probs):
                if cat == 9: # Skip rating tags (handled by profile)
                    continue
                thresh = character_threshold if cat == 4 else general_threshold
                if prob >= thresh:
                    predicted_tags.append(name)

            return predicted_tags
        except Exception:
            return []
