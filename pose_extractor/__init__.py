"""Extract one photo per distinct pose held in a bodybuilding video."""

from .models import Config, ExtractedPose
from .pipeline import extract_poses

__all__ = ["Config", "ExtractedPose", "extract_poses"]
__version__ = "0.1.0"
