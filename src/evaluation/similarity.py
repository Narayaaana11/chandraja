"""Similarity computation engine using SentenceTransformers."""

import logging
import re
from typing import Dict, List, Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception as e:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False
    SENTENCE_TRANSFORMERS_ERROR = str(e)
else:
    SENTENCE_TRANSFORMERS_ERROR = ""

logger = logging.getLogger(__name__)


class SimilarityEngine:
    """Compute semantic similarity between texts using pre-trained models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize similarity engine.
        
        Args:
            config: Dictionary with evaluation configuration including:
                - model: Model name (e.g., "all-MiniLM-L6-v2")
                - similarity_threshold: Minimum similarity score (0.0-1.0)
        """
        self.config = config
        self.model_name = config.get('model', 'all-MiniLM-L6-v2')
        self.threshold = config.get('similarity_threshold', 0.5)
        self.use_semantic_model = False
        self.model = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.use_semantic_model = True
                logger.info("SentenceTransformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load semantic model, using lexical fallback: {e}")
        else:
            logger.warning(
                "sentence-transformers unavailable, using lexical fallback similarity"
                + (f": {SENTENCE_TRANSFORMERS_ERROR}" if SENTENCE_TRANSFORMERS_ERROR else "")
            )
    
    def compute(self, student_text: str, reference_text: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            student_text: Student's answer text
            reference_text: Reference/model answer text
        
        Returns:
            Similarity score between 0.0 (no similarity) and 1.0 (identical)
        """
        if not student_text or not reference_text:
            logger.warning("Empty text provided")
            return 0.0
        
        if not self.use_semantic_model or self.model is None:
            return self._compute_lexical_similarity(student_text, reference_text)

        try:
            # Encode both texts
            student_embedding = self.model.encode(student_text, convert_to_tensor=True)
            reference_embedding = self.model.encode(reference_text, convert_to_tensor=True)
            
            # Compute cosine similarity
            from sentence_transformers.util import pytorch_cos_sim
            similarity = pytorch_cos_sim(student_embedding, reference_embedding)[0][0]

            # Convert to float between 0 and 1 and guard non-finite model outputs.
            raw_score = float(similarity.numpy())
            if not np.isfinite(raw_score):
                raise ValueError("Non-finite semantic similarity score")
            similarity_score = float(np.clip(raw_score, 0.0, 1.0))
            
            logger.debug(f"Similarity computed: {similarity_score:.4f}")
            return similarity_score
        
        except Exception as e:
            logger.warning(f"Error computing semantic similarity, using lexical fallback: {e}")
            return self._compute_lexical_similarity(student_text, reference_text)

    def _compute_lexical_similarity(self, student_text: str, reference_text: str) -> float:
        """Compute a lightweight similarity score without external ML dependencies."""
        try:
            student_tokens = self._tokenize(student_text)
            reference_tokens = self._tokenize(reference_text)

            if not student_tokens or not reference_tokens:
                return 0.0

            student_set = set(student_tokens)
            reference_set = set(reference_tokens)

            intersection = len(student_set & reference_set)
            union = len(student_set | reference_set)
            jaccard = intersection / union if union else 0.0

            coverage = intersection / len(reference_set) if reference_set else 0.0

            score = 0.6 * jaccard + 0.4 * coverage
            if not np.isfinite(score):
                return 0.0
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.error(f"Lexical similarity failed: {e}")
            return 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize into alphanumeric words, dropping short/common stopwords."""
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        common = {
            'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with',
            'from', 'as', 'if', 'but', 'that', 'this', 'which', 'who',
            'what', 'when', 'where', 'why', 'how'
        }
        return [t for t in tokens if len(t) > 2 and t not in common]
    
    def compute_batch(self, student_questions: Dict[str, str],
                     reference_questions: Dict[str, str]) -> Dict[str, float]:
        """
        Compute similarity for multiple question pairs.
        
        Args:
            student_questions: Dictionary of {question_id: answer_text}
            reference_questions: Dictionary of {question_id: model_answer_text}
        
        Returns:
            Dictionary of {question_id: similarity_score}
        """
        scores = {}
        
        for q_id in student_questions.keys():
            if q_id in reference_questions:
                student_ans = student_questions[q_id]
                reference_ans = reference_questions[q_id]
                score = self.compute(student_ans, reference_ans)
                if not np.isfinite(score):
                    score = 0.0
                scores[q_id] = score
            else:
                logger.warning(f"No reference answer found for {q_id}")
                scores[q_id] = 0.0
        
        logger.info(f"Batch similarity computed for {len(scores)} questions")
        return scores
    
    def extract_missing_keywords(self, student_text: str,
                                reference_text: str) -> List[str]:
        """
        Extract important keywords from reference that are missing in student answer.
        
        Args:
            student_text: Student's answer
            reference_text: Reference answer
        
        Returns:
            List of missing important keywords
        """
        if not student_text or not reference_text:
            return []
        
        try:
            # Simple tokenization - split by whitespace and remove common words
            def tokenize(text):
                # Convert to lowercase and split
                tokens = text.lower().split()
                # Remove short tokens and common words
                common = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were',
                         'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                         'did', 'will', 'would', 'could', 'should', 'may', 'might',
                         'can', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with',
                         'from', 'as', 'if', 'but', 'that', 'this', 'which', 'who',
                         'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'when',
                         'where', 'why', 'how'}
                return [t for t in tokens if len(t) > 2 and t not in common]
            
            student_tokens = set(tokenize(student_text))
            reference_tokens = set(tokenize(reference_text))
            
            # Find missing tokens (in reference but not in student answer)
            missing = reference_tokens - student_tokens
            
            # Sort by frequency in reference (more important keywords first)
            missing_sorted = sorted(
                missing,
                key=lambda x: reference_text.lower().count(x),
                reverse=True
            )
            
            logger.debug(f"Missing keywords: {missing_sorted}")
            return missing_sorted[:10]  # Return top 10 missing keywords
        
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
