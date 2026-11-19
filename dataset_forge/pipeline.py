import os
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from PIL import Image

from .models import ModelDialect
from .profiles import get_profile, ILLUSTRIOUS_SDXL_BUCKETS
from .analyzer import IMAGE_EXTENSIONS
from .deduper import find_duplicates
from .cropper import crop_worker
from .bucketer import find_best_bucket
from .tagger import process_tags, parse_caption_text
from .exporter import export_lora_structure, export_metadata_jsonl
from .watermark import clean_watermark_worker
from .vision_tagger import WD14Tagger

console = Console()

def clean_watermarks_pipeline(
    dataset_dir: Path,
    output_dir: Path,
    region: str = "corners",
    method: str = "telea",
    trim_borders: bool = True
) -> int:
    """Cleans watermarks, signatures, and letterbox bars from dataset using multi-threaded parallelism."""
    paths = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        console.print("[yellow]No images found in source folder.[/yellow]")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(p, output_dir / p.name, region, trim_borders, method, None) for p in paths]
    workers = min(len(tasks), os.cpu_count() or 4)
    success_count = 0

    with Progress(console=console) as progress:
        task = progress.add_task(f"Cleaning watermarks [Parallel: {workers} threads] ({region})...", total=len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(clean_watermark_worker, t) for t in tasks]
            for f in as_completed(futures):
                try:
                    if f.result():
                        success_count += 1
                except Exception as e:
                    console.print(f"[red]Error on image:[/red] {e}")
                progress.update(task, advance=1)

    return success_count

def crop_pipeline(
    dataset_dir: Path,
    output_dir: Path,
    face_detect: bool = True
) -> int:
    """Smart-crops dataset to closest SDXL 1024 aspect-ratio buckets."""
    paths = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        console.print("[yellow]No images found to crop.[/yellow]")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_tasks = []
    for p in paths:
        with Image.open(p) as img:
            w, h = img.size
        bucket = find_best_bucket(w, h, ILLUSTRIOUS_SDXL_BUCKETS)
        out_file = output_dir / p.name
        crop_tasks.append((p, bucket, out_file, face_detect))

    workers = min(len(crop_tasks), os.cpu_count() or 4)
    success_count = 0

    with Progress(console=console) as progress:
        task = progress.add_task(f"Smart cropping [Parallel: {workers} threads]...", total=len(crop_tasks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(crop_worker, t) for t in crop_tasks]
            for f in as_completed(futures):
                try:
                    if f.result():
                        success_count += 1
                except Exception as e:
                    console.print(f"[red]Error on crop:[/red] {e}")
                progress.update(task, advance=1)

    return success_count

def autotag_pipeline(
    dataset_dir: Path,
    trigger_word: Optional[str] = None,
    character_tags: Optional[str] = None,
    threshold: float = 0.35,
    quality: bool = True
) -> int:
    """Auto-generates Danbooru caption tags for images using the WD14 Vision Tagger."""
    paths = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        console.print("[yellow]No images found to tag.[/yellow]")
        return 0

    console.print(f"Loading WD14 Vision Tagger for [cyan]{len(paths)}[/cyan] images...")
    tagger = WD14Tagger()
    profile = get_profile(ModelDialect.ILLUSTRIOUS)
    char_list = [c.strip() for c in character_tags.split(",")] if character_tags else []
    q_tags = profile.quality_tags if quality else []

    with Progress(console=console) as progress:
        task = progress.add_task("Vision auto-tagging images...", total=len(paths))
        for p in paths:
            pred_tags = tagger.predict(p, general_threshold=threshold)
            res = process_tags(
                tags=pred_tags,
                trigger_word=trigger_word,
                character_tags=char_list,
                quality_tags=q_tags,
                banned_tags=profile.banned_tags
            )
            out_txt = p.with_suffix(".txt")
            out_txt.write_text(res.final_caption, encoding="utf-8")
            progress.update(task, advance=1)

    return len(paths)

def prepare_pipeline(
    dataset_dir: Path,
    output_dir: Path,
    concept: str = "concept",
    repeats: int = 10,
    trigger_word: Optional[str] = None,
    dedupe_check: bool = True,
    watermark_clean: bool = False,
    autotag: bool = True,
    tag_threshold: float = 0.35
) -> Path:
    """All-in-one preparation pipeline: Dedupe -> Watermark Inpaint -> Smart Crop -> Vision Tag / Clean -> Kohya Export."""
    temp_crop_dir = output_dir / "_temp_cropped"
    temp_crop_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if dedupe_check:
        console.print(f"Checking {len(paths)} images for duplicates...")
        dupes = find_duplicates(paths, max_distance=4)
        if dupes:
            console.print(f"[yellow]Found {len(dupes)} duplicates. Skipping duplicates in pipeline...[/yellow]")
            dupe_paths = {d.duplicate for d in dupes}
            paths = [p for p in paths if p not in dupe_paths]

    working_paths = paths
    workers = min(len(paths), os.cpu_count() or 4)

    if watermark_clean:
        wm_tasks = [(p, temp_crop_dir / f"_wm_{p.name}", "corners", True, "telea", None) for p in paths]
        with Progress(console=console) as progress:
            task = progress.add_task(f"Cleaning watermarks [Parallel: {workers} threads]...", total=len(wm_tasks))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(clean_watermark_worker, t) for t in wm_tasks]
                for f in as_completed(futures):
                    progress.update(task, advance=1)
        working_paths = [temp_crop_dir / f"_wm_{p.name}" for p in paths]

    crop_tasks = []
    for orig_p, work_p in zip(paths, working_paths):
        with Image.open(work_p) as img:
            w, h = img.size
        bucket = find_best_bucket(w, h, ILLUSTRIOUS_SDXL_BUCKETS)
        out_file = temp_crop_dir / orig_p.name
        crop_tasks.append((work_p, bucket, out_file, True))

    with Progress(console=console) as progress:
        task = progress.add_task(f"Smart cropping [Parallel: {workers} threads]...", total=len(crop_tasks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(crop_worker, t) for t in crop_tasks]
            for f in as_completed(futures):
                progress.update(task, advance=1)

    if watermark_clean:
        for p in working_paths:
            p.unlink(missing_ok=True)

    profile = get_profile(ModelDialect.ILLUSTRIOUS)
    txt_files = list(temp_crop_dir.glob("*.txt"))

    if not txt_files and autotag:
        console.print("[bold cyan]No existing captions found. Automatically generating tags with WD14 Vision Tagger...[/bold cyan]")
        tagger = WD14Tagger()
        cropped_imgs = [p for p in temp_crop_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

        with Progress(console=console) as progress:
            task = progress.add_task("Vision auto-tagging images...", total=len(cropped_imgs))
            for img_p in cropped_imgs:
                pred_tags = tagger.predict(img_p, general_threshold=tag_threshold)
                res = process_tags(
                    tags=pred_tags,
                    trigger_word=trigger_word or concept,
                    quality_tags=profile.quality_tags,
                    banned_tags=profile.banned_tags
                )
                img_p.with_suffix(".txt").write_text(res.final_caption, encoding="utf-8")
                progress.update(task, advance=1)
    elif not txt_files and not autotag:
        cropped_imgs = [p for p in temp_crop_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        baseline_caption = ", ".join([trigger_word or concept] + profile.quality_tags)
        for img_p in cropped_imgs:
            img_p.with_suffix(".txt").write_text(baseline_caption, encoding="utf-8")
    else:
        console.print("Normalizing existing tags with Illustrious standard quality tags...")
        for txt_file in txt_files:
            content = txt_file.read_text(encoding="utf-8")
            tags = parse_caption_text(content)
            res = process_tags(
                tags=tags,
                trigger_word=trigger_word or concept,
                quality_tags=profile.quality_tags,
                banned_tags=profile.banned_tags
            )
            txt_file.write_text(res.final_caption, encoding="utf-8")

    final_dir = export_lora_structure(temp_crop_dir, output_dir, repeats, concept)
    export_metadata_jsonl(final_dir, final_dir / "metadata.jsonl")

    shutil.rmtree(temp_crop_dir, ignore_errors=True)
    return final_dir
