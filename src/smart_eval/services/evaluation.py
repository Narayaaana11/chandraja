from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from smart_eval.config import (
    BEST_MODEL_FILE,
    LEGACY_MODEL_FILE,
    LEGACY_MODEL_METADATA_FILE,
)
from smart_eval.ml.features import extract_feature_metrics, parse_keywords
from smart_eval.ml.predict import TrainedScorePredictor

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class SemanticScorer:
    def __init__(self) -> None:
        self.mode = "tfidf"
        self.model = None

        if SentenceTransformer is None:
            logger.warning("sentence-transformers import failed. Falling back to TF-IDF similarity.")
            return

        try:
            logger.info("Loading sentence-transformers model...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.mode = "transformer"
            logger.info("Sentence-transformers model loaded successfully.")
        except Exception as exc:
            logger.warning(
                "Failed to load sentence-transformers model (%s). Falling back to TF-IDF similarity.",
                exc,
            )

    @staticmethod
    def _clip(val: float) -> float:
        return max(0.0, min(float(val), 1.0))

    def similarity(self, text_a: str, text_b: str) -> float:
        if self.mode == "transformer" and self.model is not None:
            emb = self.model.encode([text_a, text_b])
            return self._clip(cosine_similarity([emb[0]], [emb[1]])[0][0])

        try:
            vect = TfidfVectorizer(stop_words="english")
            mat = vect.fit_transform([text_a, text_b])
            return self._clip(cosine_similarity(mat[0], mat[1])[0][0])
        except ValueError:
            return 0.0

    def pairwise_similarity(self, texts: List[str]) -> np.ndarray:
        if len(texts) < 2:
            return np.zeros((len(texts), len(texts)))

        if self.mode == "transformer" and self.model is not None:
            emb = self.model.encode(texts)
            return cosine_similarity(emb)

        try:
            vect = TfidfVectorizer(stop_words="english")
            mat = vect.fit_transform(texts)
            return cosine_similarity(mat)
        except ValueError:
            return np.zeros((len(texts), len(texts)))


class ContentScorer:
    def __init__(self, semantic_scorer: SemanticScorer) -> None:
        self.semantic_scorer = semantic_scorer
        self.predictor = TrainedScorePredictor.from_file(BEST_MODEL_FILE)
        if self.predictor is None:
            self.predictor = TrainedScorePredictor.from_file(LEGACY_MODEL_FILE)
        self.model_info = self._load_model_info(LEGACY_MODEL_METADATA_FILE)

    @staticmethod
    def _load_model_info(meta_path: Path) -> Dict[str, object]:
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get_content_score(
        self,
        student_text: str,
        reference_text: str,
        keywords: List[str],
        max_marks: int = 10,
    ) -> Dict[str, object]:
        metrics = extract_feature_metrics(
            student_text=student_text,
            reference_text=reference_text,
            keywords=keywords,
            semantic_fn=self.semantic_scorer.similarity,
        )

        heuristic_ratio = (
            0.6 * float(metrics["semantic_score"])
            + 0.25 * float(metrics["keyword_score"])
            + 0.15 * float(metrics["length_score"])
        )

        if self.predictor is not None:
            ratio = self.predictor.predict_ratio(
                semantic_score=float(metrics["semantic_score"]),
                keyword_score=float(metrics["keyword_score"]),
                length_score=float(metrics["length_score"]),
            )
            score_mode = "trained_model"
        else:
            ratio = float(max(0.0, min(heuristic_ratio, 1.0)))
            score_mode = "heuristic"

        marks = round(ratio * max_marks, 2)

        return {
            "semantic_score": round(float(metrics["semantic_score"]), 2),
            "keyword_score": round(float(metrics["keyword_score"]), 2),
            "length_score": round(float(metrics["length_score"]), 2),
            "marks": marks,
            "found_keywords": metrics["found_keywords"],
            "missing_keywords": metrics["missing_keywords"],
            "content_mode": score_mode,
            "model_info": self.model_info,
        }


def preprocess_and_ocr(image_path: str) -> tuple[str, np.ndarray, np.ndarray]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read uploaded image. Please upload a valid PNG/JPG file.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    try:
        text = pytesseract.image_to_string(thresh)
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR is not installed or not in PATH. Install it and restart the server."
        )

    return text, image, gray


def get_presentation_score(image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    clarity_score = min(laplacian_var / 1000, 1)

    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    ink_pixels = np.sum(binary > 0)
    total_pixels = binary.shape[0] * binary.shape[1]
    ink_density = ink_pixels / total_pixels
    density_score = min(ink_density * 2, 1)

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    alignment_score = 0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        alignment_score = max(0, min(1 - np.std(angles), 1))

    presentation_score = (clarity_score * 0.4 + density_score * 0.3 + alignment_score * 0.3) * 2

    return {
        "clarity": round(float(clarity_score), 2),
        "ink_density": round(float(density_score), 2),
        "alignment": round(float(alignment_score), 2),
        "total": round(float(presentation_score), 2),
    }


def detect_plagiarism(texts: List[str], names: List[str], semantic_scorer: SemanticScorer) -> List[Dict[str, object]]:
    if len(texts) < 2:
        return []

    sim_matrix = semantic_scorer.pairwise_similarity(texts)
    results = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = float(sim_matrix[i][j])
            results.append(
                {
                    "student_a": names[i],
                    "student_b": names[j],
                    "similarity": round(sim, 2),
                    "flagged": sim > 0.9,
                }
            )
    return results


def parse_keywords_from_form(raw_keywords: str) -> List[str]:
    return parse_keywords(raw_keywords)
