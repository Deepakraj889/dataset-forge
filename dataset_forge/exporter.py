import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from .models import ImageQualityReport
from .analyzer import IMAGE_EXTENSIONS

def export_kohya_structure(
    source_dir: Path,
    output_dir: Path,
    repeats: int,
    concept_name: str
) -> Path:
    """Exports dataset into canonical Kohya-ss folder naming: {repeats}_{concept_name}."""
    target_folder = output_dir / f"{repeats}_{concept_name}"
    target_folder.mkdir(parents=True, exist_ok=True)

    for p in source_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            # Copy image
            shutil.copy2(p, target_folder / p.name)
            # Copy paired caption txt if present
            txt_pair = p.with_suffix(".txt")
            if txt_pair.exists():
                shutil.copy2(txt_pair, target_folder / txt_pair.name)

    return target_folder

def export_metadata_jsonl(
    source_dir: Path,
    output_file: Path
) -> int:
    """Exports image-caption pairs into HuggingFace/Kohya metadata.jsonl format."""
    records = []
    for p in source_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            txt_pair = p.with_suffix(".txt")
            caption = txt_pair.read_text(encoding="utf-8").strip() if txt_pair.exists() else ""
            record = {
                "file_name": p.name,
                "caption": caption
            }
            records.append(record)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return len(records)
