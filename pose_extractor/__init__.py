"""Extract one photo of each classic bodybuilding pose held in a video."""

from .models import Config, ExtractedPose
from .pipeline import extract_poses

__all__ = ["Config", "ExtractedPose", "extract_poses"]
__version__ = "0.3.0"
