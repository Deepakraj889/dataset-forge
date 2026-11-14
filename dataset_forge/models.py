from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class ModelDialect(str, Enum):
    ILLUSTRIOUS = "illustrious"
    PONY = "pony"
    SDXL = "sdxl"
    FLUX = "flux"

@dataclass
class ResolutionBucket:
    width: int
    height: int
    name: str = ""

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    def __repr__(self) -> str:
        return f"{self.width}x{self.height} (AR {self.aspect_ratio:.2f})"

@dataclass
class ImageQualityReport:
    file_path: Path
    width: int
    height: int
    aspect_ratio: float
    blur_score: float
    is_blurry: bool
    mean_brightness: float
    is_dark: bool
    is_blown_out: bool
    assigned_bucket: Optional[ResolutionBucket] = None

@dataclass
class DuplicateMatch:
    original: Path
    duplicate: Path
    distance: int

@dataclass
class TagProcessingResult:
    original_tags: List[str]
    cleaned_tags: List[str]
    injected_trigger: Optional[str]
    character_tags: List[str]
    quality_tags: List[str]

    @property
    def final_caption(self) -> str:
        return ", ".join(self.cleaned_tags)
