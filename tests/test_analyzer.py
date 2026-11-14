import numpy as np
from PIL import Image
from dataset_forge.analyzer import analyze_single_image

def test_analyze_single_image(tmp_path):
    img_path = tmp_path / "test.png"
    # Create sharp checkerboard image
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[::2, ::2] = 255
    arr[1::2, 1::2] = 255
    Image.fromarray(arr).save(img_path)

    report = analyze_single_image(img_path)
    assert report is not None
    assert report.width == 200
    assert report.height == 200
    assert report.blur_score > 0
    assert not report.is_blurry
