import numpy as np
import cv2
from PIL import Image
from dataset_forge.watermark import (
    detect_solid_borders,
    generate_corner_mask,
    inpaint_image,
    remove_watermark
)

def test_detect_solid_borders():
    # Create image with 20px black bar on bottom
    img = np.ones((200, 200, 3), dtype=np.uint8) * 180
    img[180:, :] = 0  # 20px black bar at bottom

    top, bottom, left, right = detect_solid_borders(img)
    assert bottom >= 18
    assert top == 0

def test_generate_corner_mask():
    # Create blank image with white text in bottom-right corner
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(img, "@ArtistSignature", (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    mask = generate_corner_mask(img, region="bottom-right", margin_ratio=0.35)
    assert np.any(mask[200:, 180:])
    # Should not mark top-left corner
    assert not np.any(mask[:100, :100])

def test_remove_watermark_end_to_end(tmp_path):
    img_path = tmp_path / "watermarked.png"
    out_path = tmp_path / "cleaned.png"

    # Create image with texture and watermark text
    img = np.ones((250, 250, 3), dtype=np.uint8) * 120
    cv2.putText(img, "SAMPLE WATERMARK", (150, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
    Image.fromarray(img).save(img_path)

    success = remove_watermark(img_path, out_path, region="bottom-right")
    assert success
    assert out_path.exists()

    # Verify output can be read and dimensions match
    with Image.open(out_path) as res:
        assert res.size == (250, 250)
