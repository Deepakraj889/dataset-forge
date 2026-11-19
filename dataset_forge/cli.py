from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .models import ModelDialect
from .profiles import get_profile
from .analyzer import analyze_dataset, IMAGE_EXTENSIONS
from .deduper import find_duplicates
from .tagger import process_tags, parse_caption_text
from .pipeline import (
    clean_watermarks_pipeline,
    crop_pipeline,
    autotag_pipeline,
    prepare_pipeline,
)

app = typer.Typer(
    name="dataset-forge",
    help="⚡ DatasetForge: High-performance dataset curation, smart bucketing & tag engineering for SDXL & Illustrious.",
    add_completion=False,
    invoke_without_command=True
)
console = Console()

BANNER = r"""[bold cyan]
  ___       _                 _   ___                     
 |   \ __ _| |_ __ _ ___ ___| |_| __|__ _ _ __ _ ___ 
 | |) / _` |  _/ _` (_-</ -_)  _| _/ _ \ '_/ _` / -_)
 |___/\__,_|\__\__,_/__/\___|\__|_| \___/_| \__, \___|
                                            |___/     
[/bold cyan][dim]Dataset Curation & Tag Engineering for Illustrious (SDXL)[/dim]
"""

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Launch interactive shell by default if no subcommand is passed."""
    if ctx.invoked_subcommand is None:
        from .repl import run_interactive_shell
        run_interactive_shell()

@app.command(name="shell")
def shell_cmd():
    """Launch the interactive DatasetForge shell."""
    from .repl import run_interactive_shell
    run_interactive_shell()

@app.command()
def analyze(
    dataset_dir: Path = typer.Argument(..., help="Path to image dataset folder"),
    dialect: ModelDialect = typer.Option(ModelDialect.ILLUSTRIOUS, "--dialect", "-d", help="Model dialect profile"),
    blur_threshold: Optional[float] = typer.Option(None, "--blur-threshold", "-b", help="Custom blur variance threshold")
):
    """Analyze dataset quality, aspect ratios, blur score, and bucket assignments."""
    console.print(BANNER)
    if not dataset_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Directory '{dataset_dir}' does not exist.")
        raise typer.Exit(1)

    profile = get_profile(dialect)
    if blur_threshold is not None:
        profile.blur_threshold = blur_threshold

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(description=f"Analyzing dataset images in {dataset_dir.name}...", total=None)
        reports = analyze_dataset(dataset_dir, profile)

    if not reports:
        console.print("[yellow]No supported images found.[/yellow]")
        return

    table = Table(title=f"Dataset Quality Report ({len(reports)} images)", show_header=True, header_style="bold magenta")
    table.add_column("File", style="dim", max_width=30)
    table.add_column("Resolution", justify="right")
    table.add_column("Aspect Ratio", justify="right")
    table.add_column("Target Bucket", style="cyan")
    table.add_column("Blur Score", justify="right")
    table.add_column("Status", justify="center")

    blurry_count = 0
    bucket_counts = {}

    for r in reports[:25]:
        status = "[green]Pass[/green]"
        if r.is_blurry:
            status = "[bold red]Blurry[/bold red]"
            blurry_count += 1
        elif r.is_dark:
            status = "[yellow]Dark[/yellow]"
        elif r.is_blown_out:
            status = "[yellow]Blown Out[/yellow]"

        b_name = r.assigned_bucket.name if r.assigned_bucket else "N/A"
        bucket_counts[b_name] = bucket_counts.get(b_name, 0) + 1

        table.add_row(
            r.file_path.name,
            f"{r.width}x{r.height}",
            f"{r.aspect_ratio:.2f}",
            b_name,
            f"{r.blur_score:.1f}",
            status
        )

    console.print(table)
    if len(reports) > 25:
        console.print(f"[dim]... and {len(reports) - 25} more images.[/dim]")

    console.print(Panel(
        f"[bold]Total Images:[/bold] {len(reports)}\n"
        f"[bold]Flagged Blurry (< {profile.blur_threshold:.1f}):[/bold] {blurry_count}\n"
        f"[bold]Profile Dialect:[/bold] {dialect.value.upper()}",
        title="Summary",
        border_style="green"
    ))

@app.command()
def dedupe(
    dataset_dir: Path = typer.Argument(..., help="Path to image dataset folder"),
    max_distance: int = typer.Option(5, "--threshold", "-t", help="Hamming distance threshold (0=exact, 1-6=near duplicate)"),
    quarantine_dir: Optional[Path] = typer.Option(None, "--quarantine", "-q", help="Move duplicates to this directory instead of deleting")
):
    """Scan and detect duplicate or near-duplicate images using perceptual hashing."""
    console.print(BANNER)
    paths = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    if not paths:
        console.print("[yellow]No images found to deduplicate.[/yellow]")
        return

    console.print(f"Scanning [cyan]{len(paths)}[/cyan] images with pHash (Threshold: {max_distance})...")
    duplicates = find_duplicates(paths, max_distance=max_distance)

    if not duplicates:
        console.print("[bold green]✓ No duplicate images detected![/bold green]")
        return

    table = Table(title=f"Detected Duplicates ({len(duplicates)} pairs)", show_header=True, header_style="bold red")
    table.add_column("Original Image", style="cyan")
    table.add_column("Duplicate Image", style="yellow")
    table.add_column("Hamming Dist", justify="center")

    for d in duplicates:
        table.add_row(d.original.name, d.duplicate.name, str(d.distance))
        if quarantine_dir:
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            target = quarantine_dir / d.duplicate.name
            d.duplicate.rename(target)

    console.print(table)
    if quarantine_dir:
        console.print(f"[bold green]Moved {len(duplicates)} duplicate files to '{quarantine_dir}'.[/bold green]")

@app.command()
def tag(
    dataset_dir: Path = typer.Argument(..., help="Path to dataset folder containing .txt captions"),
    trigger_word: Optional[str] = typer.Option(None, "--trigger", "-t", help="Trigger word to prepend (e.g. 'hinata')"),
    character_tags: Optional[str] = typer.Option(None, "--character", "-c", help="Comma-separated character identity tags"),
    quality_preset: bool = typer.Option(True, "--quality/--no-quality", help="Append Illustrious quality tags")
):
    """Batch clean, normalize, and engineer Danbooru tags for Illustrious."""
    console.print(BANNER)
    profile = get_profile(ModelDialect.ILLUSTRIOUS)
    q_tags = profile.quality_tags if quality_preset else []
    char_list = [c.strip() for c in character_tags.split(",")] if character_tags else []

    txt_files = list(dataset_dir.rglob("*.txt"))
    if not txt_files:
        console.print("[yellow]No .txt caption files found.[/yellow]")
        return

    updated = 0
    with Progress(console=console) as progress:
        task = progress.add_task("Cleaning tags...", total=len(txt_files))
        for p in txt_files:
            content = p.read_text(encoding="utf-8")
            tags = parse_caption_text(content)
            res = process_tags(
                tags=tags,
                trigger_word=trigger_word,
                character_tags=char_list,
                quality_tags=q_tags,
                banned_tags=profile.banned_tags
            )
            p.write_text(res.final_caption, encoding="utf-8")
            updated += 1
            progress.update(task, advance=1)

    console.print(f"[bold green]✓ Successfully normalized and updated {updated} caption files![/bold green]")

@app.command(name="autotag")
def autotag_cmd(
    dataset_dir: Path = typer.Argument(..., help="Path to image dataset folder"),
    trigger_word: Optional[str] = typer.Option(None, "--trigger", "-t", help="Trigger word to prepend"),
    character_tags: Optional[str] = typer.Option(None, "--character", "-c", help="Character identity tags"),
    threshold: float = typer.Option(0.35, "--threshold", "-th", help="Tag confidence threshold"),
    quality: bool = typer.Option(True, "--quality/--no-quality", help="Append Illustrious quality tags")
):
    """Automatically generate Danbooru caption tags for images using the WD14 Vision Tagger."""
    console.print(BANNER)
    count = autotag_pipeline(
        dataset_dir=dataset_dir,
        trigger_word=trigger_word,
        character_tags=character_tags,
        threshold=threshold,
        quality=quality
    )
    console.print(f"[bold green]✓ Successfully generated captions for {count} images![/bold green]")

@app.command()
def crop(
    dataset_dir: Path = typer.Argument(..., help="Path to image dataset folder"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Target output folder for cropped images"),
    face_detect: bool = typer.Option(True, "--face-detect/--center-crop", help="Use face-aware focal point crop")
):
    """Smart-crop images to their closest SDXL / Illustrious aspect ratio bucket."""
    console.print(BANNER)
    count = crop_pipeline(dataset_dir=dataset_dir, output_dir=output_dir, face_detect=face_detect)
    console.print(f"[bold green]✓ Cropped {count} images into '{output_dir}'.[/bold green]")

@app.command(name="clean-watermarks")
def clean_watermarks_cmd(
    dataset_dir: Path = typer.Argument(..., help="Path to image dataset folder"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Target output folder"),
    region: str = typer.Option("corners", "--region", "-r", help="Region to scan: corners, bottom-right, bottom, all-margins"),
    method: str = typer.Option("telea", "--method", "-m", help="Inpainting algorithm: telea or ns"),
    trim_borders: bool = typer.Option(True, "--trim-borders/--no-trim-borders", help="Auto-trim solid letterbox/banner bars")
):
    """Scan and remove watermarks, signatures, logos, and letterbox bars using all CPU cores."""
    console.print(BANNER)
    count = clean_watermarks_pipeline(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        region=region,
        method=method,
        trim_borders=trim_borders
    )
    console.print(f"[bold green]✓ Cleaned {count} images into '{output_dir}'.[/bold green]")

@app.command()
def prepare(
    dataset_dir: Path = typer.Argument(..., help="Raw dataset folder"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Final destination folder"),
    concept: str = typer.Option(..., "--concept", "-c", help="Concept / Character name (e.g. 'makima')"),
    repeats: int = typer.Option(10, "--repeats", "-r", help="Training folder repeats count (default: 10)"),
    trigger_word: Optional[str] = typer.Option(None, "--trigger", "-t", help="Trigger word to prepend"),
    dedupe_check: bool = typer.Option(True, "--dedupe/--no-dedupe", help="Filter duplicates before processing"),
    watermark_clean: bool = typer.Option(False, "--watermark-clean/--no-watermark-clean", help="Auto-inpaint corner watermarks & signatures"),
    autotag: bool = typer.Option(True, "--autotag/--no-autotag", help="Auto-generate Danbooru tags using WD14 vision model if captions missing"),
    tag_threshold: float = typer.Option(0.35, "--threshold", "-th", help="Vision tagger confidence threshold")
):
    """End-to-End Pipeline: Analyze -> Dedupe -> Watermark Inpaint -> Smart Crop -> Vision Tag / Clean -> Export standard LoRA structure."""
    console.print(BANNER)
    console.print(f"[bold cyan]Starting End-to-End Preparation for '{concept}' ({repeats} repeats)...[/bold cyan]")
    final_dir = prepare_pipeline(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        concept=concept,
        repeats=repeats,
        trigger_word=trigger_word,
        dedupe_check=dedupe_check,
        watermark_clean=watermark_clean,
        autotag=autotag,
        tag_threshold=tag_threshold
    )
    console.print(Panel(
        f"[bold green]✓ Pipeline Complete![/bold green]\n"
        f"[bold]Output Directory:[/bold] {final_dir}\n"
        f"[bold]Format:[/bold] Standard LoRA ({repeats}_{concept}) + metadata.jsonl",
        title="Success",
        border_style="green"
    ))

if __name__ == "__main__":
    app()
