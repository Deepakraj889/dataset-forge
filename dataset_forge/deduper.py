from pathlib import Path
from typing import List, Dict, Tuple, Set
from PIL import Image
import imagehash
from .models import DuplicateMatch

def compute_hashes(image_paths: List[Path]) -> Dict[Path, imagehash.ImageHash]:
    """Computes perceptual hashes (pHash) for a list of images."""
    hashes: Dict[Path, imagehash.ImageHash] = {}
    for p in image_paths:
        try:
            with Image.open(p) as img:
                h = imagehash.phash(img)
                hashes[p] = h
        except Exception:
            continue
    return hashes

def find_duplicates(image_paths: List[Path], max_distance: int = 5) -> List[DuplicateMatch]:
    """Finds all image pairs with a Hamming distance <= max_distance."""
    hashes = compute_hashes(image_paths)
    paths = list(hashes.keys())
    matches: List[DuplicateMatch] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    for i in range(len(paths)):
        p1 = paths[i]
        h1 = hashes[p1]
        for j in range(i + 1, len(paths)):
            p2 = paths[j]
            h2 = hashes[p2]
            dist = h1 - h2 # Hamming distance
            if dist <= max_distance:
                key = (str(p1), str(p2))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    matches.append(DuplicateMatch(original=p1, duplicate=p2, distance=dist))

    return matches
