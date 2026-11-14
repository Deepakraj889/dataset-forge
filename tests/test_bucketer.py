from dataset_forge.bucketer import find_best_bucket, generate_sdxl_buckets
from dataset_forge.profiles import ILLUSTRIOUS_SDXL_BUCKETS

def test_find_best_bucket_square():
    bucket = find_best_bucket(1000, 1000, ILLUSTRIOUS_SDXL_BUCKETS)
    assert bucket.width == 1024
    assert bucket.height == 1024

def test_find_best_bucket_portrait():
    # 768 x 1344 is 9:16 portrait
    bucket = find_best_bucket(600, 1050, ILLUSTRIOUS_SDXL_BUCKETS)
    assert bucket.aspect_ratio < 1.0

def test_find_best_bucket_landscape():
    bucket = find_best_bucket(1920, 1080, ILLUSTRIOUS_SDXL_BUCKETS)
    assert bucket.aspect_ratio > 1.0

def test_generate_sdxl_buckets():
    buckets = generate_sdxl_buckets()
    assert len(buckets) > 5
    for b in buckets:
        assert b.width % 64 == 0
        assert b.height % 64 == 0
