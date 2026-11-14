from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image
from .models import ResolutionBucket

# Load standard OpenCV Haar face cascade
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = None

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return _face_cascade

def find_subject_focus_point(img_cv: np.ndarray) -> Tuple[float, float]:
    """
    Finds normalized (x, y) center of interest (0.0 to 1.0).
    If a face is detected, focuses slightly above face center (for head/hair balance).
    Otherwise defaults to image optical center (0.5, 0.45).
    """
    cascade = get_face_cascade()
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    h, w = img_cv.shape[:2]
    if len(faces) > 0:
        # Pick largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        # Focus on upper third of face to preserve full head/hair
        center_x = (fx + fw / 2.0) / w
        center_y = (fy + fh * 0.4) / h
        return center_x, center_y

    return 0.5, 0.5

def smart_crop_and_resize(
    image_path: Path,
    target_bucket: ResolutionBucket,
    output_path: Path,
    use_face_detection: bool = True
) -> bool:
    """Crops and resizes an image to the exact target bucket resolution with subject awareness."""
    try:
        with open(image_path, "rb") as f:
            bytes_data = bytearray(f.read())
            np_arr = np.asarray(bytes_data, dtype=np.uint8)
            img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_cv is None:
            return False

        h, w = img_cv.shape[:2]
        target_w = target_bucket.width
        target_h = target_bucket.height
        target_ar = target_w / target_h
        current_ar = w / h

        if use_face_detection:
            focus_x, focus_y = find_subject_focus_point(img_cv)
        else:
            focus_x, focus_y = 0.5, 0.5

        if current_ar > target_ar:
            # Current image is wider than target: crop left/right
            new_w = int(round(h * target_ar))
            crop_center_x = int(round(focus_x * w))
            x1 = max(0, crop_center_x - new_w // 2)
            if x1 + new_w > w:
                x1 = w - new_w
            y1 = 0
            crop_w, crop_h = new_w, h
        else:
            # Current image is taller than target: crop top/bottom
            new_h = int(round(w / target_ar))
            crop_center_y = int(round(focus_y * h))
            y1 = max(0, crop_center_y - new_h // 2)
            if y1 + new_h > h:
                y1 = h - new_h
            x1 = 0
            crop_w, crop_h = w, new_h

        cropped = img_cv[y1:y1 + crop_h, x1:x1 + crop_w]

        # Resize to exact bucket dimensions with high quality interpolation
        resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Encode and save safely
        ext = output_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            _, buf = cv2.imencode(ext, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        else:
            _, buf = cv2.imencode(".png", resized)

        with open(output_path, "wb") as f:
            f.write(buf)

        return True
    except Exception:
        return False
