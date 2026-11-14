import re
from typing import List, Set, Dict, Optional
from pathlib import Path
from .models import TagProcessingResult
from .profiles import ProfileConfig, get_profile, DEFAULT_NOISE_TAGS

def normalize_danbooru_tag(tag: str) -> str:
    """Normalizes Danbooru tags: cleans underscores, unescapes characters, and strips whitespace."""
    t = tag.strip().lower()
    # Unescape escaped brackets
    t = t.replace(r"\(", "(").replace(r"\)", ")")
    # Replace underscores with spaces (standard SD prompt format)
    t = t.replace("_", " ")
    # Clean multiple spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse_caption_text(text: str) -> List[str]:
    """Parses comma-separated caption string into a list of normalized tags."""
    if not text:
        return []
    raw_tags = [t.strip() for t in text.split(",") if t.strip()]
    return [normalize_danbooru_tag(t) for t in raw_tags]

def process_tags(
    tags: List[str],
    trigger_word: Optional[str] = None,
    character_tags: Optional[List[str]] = None,
    quality_tags: Optional[List[str]] = None,
    banned_tags: Optional[Set[str]] = None,
    custom_replacements: Optional[Dict[str, str]] = None,
    keep_rating_tag: bool = True
) -> TagProcessingResult:
    """
    Cleans, normalizes, deduplicates, and re-orders tags according to Illustrious/SDXL best practices.
    Standard order:
    [Trigger Word] -> [Character Tags] -> [General Tags] -> [Quality/Rating Tags]
    """
    if banned_tags is None:
        banned_tags = DEFAULT_NOISE_TAGS
    if custom_replacements is None:
        custom_replacements = {}

    cleaned_general: List[str] = []
    seen: Set[str] = set()

    # Normalize trigger word
    norm_trigger = normalize_danbooru_tag(trigger_word) if trigger_word else None
    if norm_trigger:
        seen.add(norm_trigger)

    # Normalize character tags
    norm_char_tags: List[str] = []
    if character_tags:
        for ct in character_tags:
            nct = normalize_danbooru_tag(ct)
            if nct not in seen:
                seen.add(nct)
                norm_char_tags.append(nct)

    # Process input tags
    for raw in tags:
        tag = normalize_danbooru_tag(raw)
        if not tag:
            continue

        # Apply custom replacement map
        if tag in custom_replacements:
            tag = normalize_danbooru_tag(custom_replacements[tag])

        # Skip banned/noise tags
        if tag in banned_tags:
            continue

        # Avoid duplicates
        if tag in seen:
            continue

        seen.add(tag)
        cleaned_general.append(tag)

    # Process quality tags
    norm_quality: List[str] = []
    if quality_tags:
        for qt in quality_tags:
            nqt = normalize_danbooru_tag(qt)
            if nqt not in seen:
                seen.add(nqt)
                norm_quality.append(nqt)

    # Assemble final ordered tag list
    final_tags: List[str] = []
    if norm_trigger:
        final_tags.append(norm_trigger)
    final_tags.extend(norm_char_tags)
    final_tags.extend(cleaned_general)
    final_tags.extend(norm_quality)

    return TagProcessingResult(
        original_tags=tags,
        cleaned_tags=final_tags,
        injected_trigger=norm_trigger,
        character_tags=norm_char_tags,
        quality_tags=norm_quality
    )

def batch_update_caption_files(
    directory: Path,
    trigger_word: Optional[str] = None,
    character_tags: Optional[List[str]] = None,
    quality_tags: Optional[List[str]] = None,
    banned_tags: Optional[Set[str]] = None,
    custom_replacements: Optional[Dict[str, str]] = None
) -> int:
    """Updates all .txt caption files in a directory."""
    updated_count = 0
    for txt_path in directory.rglob("*.txt"):
        try:
            content = txt_path.read_text(encoding="utf-8")
            tags = parse_caption_text(content)
            res = process_tags(
                tags=tags,
                trigger_word=trigger_word,
                character_tags=character_tags,
                quality_tags=quality_tags,
                banned_tags=banned_tags,
                custom_replacements=custom_replacements
            )
            txt_path.write_text(res.final_caption, encoding="utf-8")
            updated_count += 1
        except Exception:
            continue
    return updated_count
