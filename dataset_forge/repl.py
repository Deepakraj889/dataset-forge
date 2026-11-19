import os
import shlex
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from .models import ModelDialect
from .analyzer import analyze_dataset, IMAGE_EXTENSIONS
from .deduper import find_duplicates
from .profiles import get_profile
from .pipeline import (
    clean_watermarks_pipeline,
    crop_pipeline,
    autotag_pipeline,
    prepare_pipeline,
)

console = Console()

SLASH_COMMANDS = [
    "/load",
    "/cd",
    "/output",
    "/out",
    "/status",
    "/concept",
    "/trigger",
    "/analyze",
    "/clean",
    "/caption",
    "/autotag",
    "/crop",
    "/dedupe",
    "/prepare",
    "/build",
    "/help",
    "/clear",
    "/exit",
    "/quit",
]

SHELL_STYLE = Style.from_dict({
    "prompt": "#00d7d7 bold",
    "at": "#888888",
    "dir": "#5fafff",
})

class DatasetForgeSession:
    def __init__(self):
        self.dataset_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.concept: str = "concept"
        self.trigger: Optional[str] = None
        self.repeats: int = 10
        self.watermark_clean: bool = True
        self.dialect: ModelDialect = ModelDialect.ILLUSTRIOUS

    @property
    def images(self):
        if not self.dataset_dir or not self.dataset_dir.exists():
            return []
        return [p for p in self.dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

def clean_input_path(raw_path: str) -> Path:
    """Cleans quotation marks, trailing slashes, and whitespace from paths."""
    p_str = raw_path.strip().strip("'\"").strip()
    return Path(p_str)

def resolve_output_dir(state: DatasetForgeSession, default_suffix: str) -> Path:
    """Returns state.output_dir if already set, or interactively prompts user with a sensible default."""
    if state.output_dir:
        state.output_dir.mkdir(parents=True, exist_ok=True)
        return state.output_dir

    default_p = state.dataset_dir.parent / f"{state.dataset_dir.name}_{default_suffix}"
    try:
        user_choice = console.input(f"[cyan]Output folder[/cyan] [dim](press Enter for '{default_p}')[/dim]: ").strip()
    except (KeyboardInterrupt, EOFError):
        return default_p

    chosen = clean_input_path(user_choice) if user_choice else default_p
    chosen.mkdir(parents=True, exist_ok=True)
    return chosen

def print_help_table():
    table = Table(title="⚡ DatasetForge Interactive Commands", show_header=True, header_style="bold magenta")
    table.add_column("Command", style="bold cyan", width=18)
    table.add_column("Arguments", style="yellow", width=22)
    table.add_column("Description", style="white")

    table.add_row("/load (or cd / paste)", "<folder_path>", "Set active image dataset (drag & drop supported)")
    table.add_row("/output (or /out)", "<folder_path>", "Explicitly set output destination folder")
    table.add_row("/status", "", "Display current loaded dataset & settings")
    table.add_row("/concept", "<name>", "Set character or concept name (e.g. leafa)")
    table.add_row("/trigger", "<word>", "Set trigger word prepended to captions")
    table.add_row("/clean", "[region]", "Inpaint watermarks & signatures across all cores")
    table.add_row("/caption", "[threshold]", "Run WD14 Vision Tagger to auto-generate tags")
    table.add_row("/crop", "", "Smart-crop to SDXL 1024 buckets (face-aware)")
    table.add_row("/analyze", "[blur_threshold]", "Scan resolutions, blur scores & SDXL buckets")
    table.add_row("/dedupe", "[threshold]", "Scan for identical / burst-shot duplicates")
    table.add_row("/prepare", "", "Run ALL-IN-ONE pipeline (Clean -> Crop -> Tag -> Export)")
    table.add_row("/clear", "", "Clear terminal screen")
    table.add_row("/exit", "", "Exit interactive shell")
    console.print(table)

def run_interactive_shell():
    state = DatasetForgeSession()
    completer = WordCompleter(SLASH_COMMANDS, ignore_case=True, sentence=True)
    history = InMemoryHistory()
    session = PromptSession(history=history, completer=completer, style=SHELL_STYLE)

    console.print(r"""[bold cyan]
  ___       _                 _   ___                     
 |   \ __ _| |_ __ _ ___ ___| |_| __|__ _ _ __ _ ___ 
 | |) / _` |  _/ _` (_-</ -_)  _| _/ _ \ '_/ _` / -_)
 |___/\__,_|\__\__,_/__/\___|\__|_| \___/_| \__, \___|
                                            |___/     
[/bold cyan][dim]Interactive Dataset Curation Shell — Type [bold cyan]/help[/bold cyan] for commands, [bold cyan]/exit[/bold cyan] to quit.[/dim]
""")

    while True:
        try:
            dir_name = state.dataset_dir.name if state.dataset_dir else "no folder"
            img_count = len(state.images)
            prompt_text = [
                ("class:prompt", "forge"),
                ("class:at", " ["),
                ("class:dir", f"{dir_name} ({img_count} imgs)"),
                ("class:at", "]> "),
            ]

            user_input = session.prompt(prompt_text).strip()
            if not user_input:
                continue

            path_candidate = clean_input_path(user_input)
            if path_candidate.is_dir():
                state.dataset_dir = path_candidate
                console.print(f"[bold green]✓ Loaded dataset:[/bold green] [cyan]{path_candidate}[/cyan] ({len(state.images)} images)")
                continue

            parts = shlex.split(user_input, posix=False)
            raw_cmd = parts[0]
            cmd = raw_cmd.lower().lstrip("/")
            args = parts[1:]

            if cmd in {"cd", "load", "dir"}:
                if not args:
                    console.print("[yellow]Usage: /load <folder_path>[/yellow]")
                    continue
                path = clean_input_path(" ".join(args))
                if path.is_dir():
                    state.dataset_dir = path
                    console.print(f"[bold green]✓ Loaded dataset:[/bold green] [cyan]{path}[/cyan] ({len(state.images)} images)")
                else:
                    console.print(f"[bold red]Error:[/bold red] Folder '{path}' does not exist.")
                continue

            elif cmd in {"out", "output"}:
                if not args:
                    console.print(f"Current output folder: [cyan]{state.output_dir or 'Not set (will prompt on run)'}[/cyan]. Usage: /output <path>")
                    continue
                out_path = clean_input_path(" ".join(args))
                out_path.mkdir(parents=True, exist_ok=True)
                state.output_dir = out_path
                console.print(f"[bold green]✓ Output folder set to:[/bold green] [cyan]{state.output_dir}[/cyan]")

            elif cmd in {"exit", "quit"}:
                console.print("[dim]Goodbye![/dim]")
                break

            elif cmd == "help":
                print_help_table()

            elif cmd == "clear":
                console.clear()

            elif cmd == "status":
                dir_str = str(state.dataset_dir) if state.dataset_dir else "[dim]None (paste path or use /load <folder>)[/dim]"
                out_str = str(state.output_dir) if state.output_dir else "[dim]Not set (prompts dynamically)[/dim]"
                console.print(Panel(
                    f"[bold]Active Dataset:[/bold] {dir_str}\n"
                    f"[bold]Total Images:[/bold] {len(state.images)}\n"
                    f"[bold]Concept Name:[/bold] {state.concept}\n"
                    f"[bold]Trigger Word:[/bold] {state.trigger or state.concept}\n"
                    f"[bold]Output Directory:[/bold] {out_str}\n"
                    f"[bold]Model Dialect:[/bold] {state.dialect.value.upper()}",
                    title="Session Status",
                    border_style="cyan"
                ))

            elif cmd == "concept":
                if not args:
                    console.print(f"Current concept: [cyan]{state.concept}[/cyan]. Usage: /concept <name>")
                    continue
                state.concept = args[0].strip().strip("'\"")
                console.print(f"[bold green]✓ Concept set to:[/bold green] [cyan]{state.concept}[/cyan]")

            elif cmd == "trigger":
                if not args:
                    console.print(f"Current trigger: [cyan]{state.trigger or state.concept}[/cyan]. Usage: /trigger <word>")
                    continue
                state.trigger = args[0].strip().strip("'\"")
                console.print(f"[bold green]✓ Trigger word set to:[/bold green] [cyan]{state.trigger}[/cyan]")

            elif cmd == "analyze":
                if not state.dataset_dir:
                    console.print("[yellow]No dataset loaded yet. Please paste a folder path or type /load <folder> first.[/yellow]")
                    continue
                b_thresh = float(args[0]) if args else None
                from .cli import analyze
                analyze(dataset_dir=state.dataset_dir, dialect=state.dialect, blur_threshold=b_thresh)

            elif cmd in {"clean", "watermark"}:
                if not state.dataset_dir:
                    console.print("[yellow]No dataset loaded yet. Please paste a folder path or type /load <folder> first.[/yellow]")
                    continue
                target_out = resolve_output_dir(state, "cleaned")
                region = args[0] if args else "corners"
                count = clean_watermarks_pipeline(
                    dataset_dir=state.dataset_dir,
                    output_dir=target_out,
                    region=region,
                    method="telea",
                    trim_borders=True
                )
                console.print(f"[bold green]✓ Cleaned {count}/{len(state.images)} images into '{target_out}'.[/bold green]")

            elif cmd in {"caption", "autotag"}:
                if not state.dataset_dir:
                    console.print("[yellow]No dataset loaded yet. Please paste a folder path or type /load <folder> first.[/yellow]")
                    continue
                thresh = float(args[0]) if args else 0.35
                count = autotag_pipeline(
                    dataset_dir=state.dataset_dir,
                    trigger_word=state.trigger or state.concept,
                    threshold=thresh
                )
                console.print(f"[bold green]✓ Generated captions for {count} images in '{state.dataset_dir}'![/bold green]")

            elif cmd == "crop":
                if not state.dataset_dir:
                    console.print("[yellow]No dataset loaded yet. Please paste a folder path or type /load <folder> first.[/yellow]")
                    continue
                target_out = resolve_output_dir(state, "cropped")
                count = crop_pipeline(
                    dataset_dir=state.dataset_dir,
                    output_dir=target_out,
                    face_detect=True
                )
                console.print(f"[bold green]✓ Cropped {count}/{len(state.images)} images into '{target_out}'.[/bold green]")

            elif cmd == "dedupe":
                if not state.dataset_dir:
                    console.print("[yellow]No dataset loaded yet. Please paste a folder path or type /load <folder> first.[/yellow]")
                    continue
                from .cli import dedupe
                thresh = int(args[0]) if args else 5
                dedupe(dataset_dir=state.dataset_dir, max_distance=thresh)

            elif cmd in {"prepare", "build"}:
                if not state.dataset_dir:
                    console.print("[yellow]No dataset loaded yet. Please paste a folder path or type /load <folder> first.[/yellow]")
                    continue
                target_out = resolve_output_dir(state, "lora_ready")
                final_dir = prepare_pipeline(
                    dataset_dir=state.dataset_dir,
                    output_dir=target_out,
                    concept=state.concept,
                    repeats=state.repeats,
                    trigger_word=state.trigger or state.concept,
                    watermark_clean=state.watermark_clean
                )
                console.print(Panel(
                    f"[bold green]✓ Pipeline Complete![/bold green]\n"
                    f"[bold]Output Directory:[/bold] {final_dir}\n"
                    f"[bold]Format:[/bold] Standard LoRA ({state.repeats}_{state.concept}) + metadata.jsonl",
                    title="Success",
                    border_style="green"
                ))

            else:
                console.print(f"[red]Unknown command:[/red] [yellow]{user_input}[/yellow]. Type [bold cyan]/help[/bold cyan] for commands, or paste a folder path to load it.")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session closed.[/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
