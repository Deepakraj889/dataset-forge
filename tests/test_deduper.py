import numpy as np
from PIL import Image
from dataset_forge.deduper import find_duplicates

def test_find_duplicates(tmp_path):
    # Create two identical test images
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"
    img3_path = tmp_path / "img3.png"

    arr1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    Image.fromarray(arr1).save(img1_path)
    Image.fromarray(arr1).save(img2_path)

    # Different noise image
    arr3 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr3).save(img3_path)

    dupes = find_duplicates([img1_path, img2_path, img3_path], max_distance=1)
    assert len(dupes) == 1
    assert dupes[0].distance == 0
