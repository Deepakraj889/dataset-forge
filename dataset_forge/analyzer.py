from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image

from .models import ImageQualityReport
from .profiles import ProfileConfig, get_profile
from .bucketer import find_best_bucket

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def analyze_single_image(image_path: Path, profile: Optional[ProfileConfig] = None) -> Optional[ImageQualityReport]:
    """Analyzes a single image for resolution, blur score, and brightness metrics."""
    if profile is None:
        profile = get_profile()

    try:
        # Load image with OpenCV (handling Unicode paths safely)
        with open(image_path, "rb") as f:
            bytes_data = bytearray(f.read())
            np_arr = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return None

        h, w = img.shape[:2]
        ar = w / max(h, 1)

        # Convert to grayscale for Laplacian blur and brightness analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute Laplacian variance (sharpness metric)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = laplacian_var < profile.blur_threshold

        # Brightness metrics
        mean_brightness = float(np.mean(gray))
        is_dark = mean_brightness < 25.0
        is_blown_out = mean_brightness > 230.0

        # Bucket assignment
        assigned_bucket = find_best_bucket(w, h, profile.buckets)

        return ImageQualityReport(
            file_path=image_path,
            width=w,
            height=h,
            aspect_ratio=ar,
            blur_score=laplacian_var,
            is_blurry=is_blurry,
            mean_brightness=mean_brightness,
            is_dark=is_dark,
            is_blown_out=is_blown_out,
            assigned_bucket=assigned_bucket
        )
    except Exception:
        return None

def analyze_dataset(directory: Path, profile: Optional[ProfileConfig] = None) -> List[ImageQualityReport]:
    """Scans and analyzes all images in a directory."""
    if profile is None:
        profile = get_profile()

    reports: List[ImageQualityReport] = []
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            rep = analyze_single_image(p, profile)
            if rep is not None:
                reports.append(rep)

    return reports
