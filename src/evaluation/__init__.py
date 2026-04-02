"""Evaluation engine module."""
try:
	from .similarity import SimilarityEngine
except Exception:
	SimilarityEngine = None

from .grader import Grader
from .feedback import FeedbackGenerator

__all__ = ["Grader", "FeedbackGenerator"]
if SimilarityEngine is not None:
	__all__.append("SimilarityEngine")
