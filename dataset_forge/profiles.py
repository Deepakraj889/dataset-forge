from typing import List, Dict, Set
from .models import ModelDialect, ResolutionBucket

ILLUSTRIOUS_SDXL_BUCKETS: List[ResolutionBucket] = [
    ResolutionBucket(1024, 1024, "1:1 Square"),
    ResolutionBucket(1152, 896, "9:7 Landscape"),
    ResolutionBucket(896, 1152, "7:9 Portrait"),
    ResolutionBucket(1216, 832, "3:2 Landscape"),
    ResolutionBucket(832, 1216, "2:3 Portrait"),
    ResolutionBucket(1344, 768, "16:9 Landscape"),
    ResolutionBucket(768, 1344, "9:16 Portrait"),
    ResolutionBucket(1536, 640, "21:9 Ultra-Wide"),
    ResolutionBucket(640, 1536, "9:21 Tall Banner"),
]

PONY_BUCKETS: List[ResolutionBucket] = ILLUSTRIOUS_SDXL_BUCKETS

DEFAULT_NOISE_TAGS: Set[str] = {
    "watermark", "signature", "username", "artist name", "sample",
    "lowres", "bad quality", "worst quality", "jpeg artifacts",
    "cropped", "censored", "bar censor", "mosaic censoring",
    "white border", "black border"
}

ILLUSTRIOUS_QUALITY_TAGS: List[str] = [
    "masterpiece", "best quality", "very aesthetic", "absurdres", "newest"
]

PONY_QUALITY_TAGS: List[str] = [
    "score_9", "score_8_up", "score_7_up", "score_6_up", "score_5_up", "score_4_up"
]

class ProfileConfig:
    def __init__(self, dialect: ModelDialect):
        self.dialect = dialect
        if dialect == ModelDialect.ILLUSTRIOUS:
            self.base_resolution = 1024
            self.buckets = ILLUSTRIOUS_SDXL_BUCKETS
            self.blur_threshold = 85.0
            self.quality_tags = ILLUSTRIOUS_QUALITY_TAGS
            self.banned_tags = DEFAULT_NOISE_TAGS
            self.rating_tags = {"safe", "general", "sensitive", "questionable", "explicit"}
        elif dialect == ModelDialect.PONY:
            self.base_resolution = 1024
            self.buckets = PONY_BUCKETS
            self.blur_threshold = 85.0
            self.quality_tags = PONY_QUALITY_TAGS
            self.banned_tags = DEFAULT_NOISE_TAGS
            self.rating_tags = {"rating_safe", "rating_questionable", "rating_explicit"}
        else: # Standard SDXL
            self.base_resolution = 1024
            self.buckets = ILLUSTRIOUS_SDXL_BUCKETS
            self.blur_threshold = 95.0
            self.quality_tags = ["masterpiece", "best quality", "ultra high res"]
            self.banned_tags = DEFAULT_NOISE_TAGS
            self.rating_tags = set()

def get_profile(dialect: ModelDialect = ModelDialect.ILLUSTRIOUS) -> ProfileConfig:
    return ProfileConfig(dialect)
