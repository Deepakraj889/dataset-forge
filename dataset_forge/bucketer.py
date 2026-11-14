import math
from typing import List, Tuple, Optional
from .models import ResolutionBucket
from .profiles import ILLUSTRIOUS_SDXL_BUCKETS

def find_best_bucket(width: int, height: int, buckets: Optional[List[ResolutionBucket]] = None) -> ResolutionBucket:
    """Finds the optimal resolution bucket for an image to minimize cropping distortion."""
    if buckets is None:
        buckets = ILLUSTRIOUS_SDXL_BUCKETS

    img_ar = width / max(height, 1)
    
    best_bucket = buckets[0]
    min_diff = float("inf")

    for bucket in buckets:
        diff = abs(bucket.aspect_ratio - img_ar)
        if diff < min_diff:
            min_diff = diff
            best_bucket = bucket

    return best_bucket

def generate_sdxl_buckets(target_area: int = 1024 * 1024, step: int = 64, min_dim: int = 512, max_dim: int = 1536) -> List[ResolutionBucket]:
    """Generates a complete mathematically consistent set of SDXL buckets divisible by 64."""
    buckets = []
    seen_sizes = set()

    # Generate heights from min_dim to max_dim stepping by 64
    for h in range(min_dim, max_dim + 1, step):
        w = round((target_area / h) / step) * step
        if min_dim <= w <= max_dim and (w, h) not in seen_sizes:
            seen_sizes.add((w, h))
            ar = w / h
            name = f"{w}x{h} (AR {ar:.2f})"
            buckets.append(ResolutionBucket(width=w, height=h, name=name))

    # Sort buckets by aspect ratio ascending (tallest portrait to widest landscape)
    buckets.sort(key=lambda b: b.aspect_ratio)
    return buckets
