from dataset_forge.tagger import (
    normalize_danbooru_tag,
    parse_caption_text,
    process_tags
)

def test_normalize_danbooru_tag():
    assert normalize_danbooru_tag("black_hair") == "black hair"
    assert normalize_danbooru_tag(r"\(medium\)") == "(medium)"
    assert normalize_danbooru_tag("  SOLO  ") == "solo"

def test_parse_caption_text():
    raw = "1girl, solo, black_hair, red_eyes"
    parsed = parse_caption_text(raw)
    assert parsed == ["1girl", "solo", "black hair", "red eyes"]

def test_process_tags_ordering_and_dedupe():
    input_tags = ["solo", "black_hair", "watermark", "solo", "bad quality"]
    res = process_tags(
        tags=input_tags,
        trigger_word="hinata",
        character_tags=["hyuuga hinata"],
        quality_tags=["masterpiece", "best quality"]
    )
    # Check trigger word is first
    assert res.cleaned_tags[0] == "hinata"
    # Check character tag is second
    assert res.cleaned_tags[1] == "hyuuga hinata"
    # Check noise tags removed
    assert "watermark" not in res.cleaned_tags
    assert "bad quality" not in res.cleaned_tags
    # Check quality tags are at the end
    assert "masterpiece" in res.cleaned_tags
    assert "best quality" in res.cleaned_tags
    # Check deduplication
    assert res.cleaned_tags.count("solo") == 1
