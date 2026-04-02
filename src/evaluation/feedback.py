"""Feedback generation for answered questions."""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class FeedbackGenerator:
    """Generate detailed feedback for evaluated questions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feedback generator.
        
        Args:
            config: Dictionary with configuration
        """
        self.config = config
    
    def generate(self, question_id: str, similarity_score: float,
                missing_keywords: List[str], marks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feedback for a single question.
        
        Args:
            question_id: Question identifier (e.g., "Q1")
            similarity_score: Similarity score (0.0-1.0)
            missing_keywords: List of keywords missing from student answer
            marks: Grading result dict with marks_awarded, max_marks, percentage
        
        Returns:
            Dictionary with structure:
            {
                "question": "Q1",
                "similarity_percentage": 82,
                "marks_awarded": 8,
                "max_marks": 10,
                "percentage": 80,
                "remark": "Good answer. Covers most key points.",
                "missing_keywords": ["entropy", "information gain"],
                "weak_areas": ["Definition lacks precision"]
            }
        """
        similarity_pct = int(similarity_score * 100)
        remark = self._generate_remark(similarity_score)
        weak_areas = self._generate_weak_areas(similarity_score, missing_keywords)
        
        return {
            "question": question_id,
            "similarity_percentage": similarity_pct,
            "marks_awarded": marks["marks_awarded"],
            "max_marks": marks["max_marks"],
            "percentage": marks.get(
                "percentage",
                int((marks.get("marks_awarded", 0) / marks.get("max_marks", 1)) * 100) if marks.get("max_marks", 0) else 0
            ),
            "remark": remark,
            "missing_keywords": missing_keywords,
            "weak_areas": weak_areas
        }
    
    def _generate_remark(self, similarity_score: float) -> str:
        """
        Generate textual remark based on similarity score.
        
        Args:
            similarity_score: Similarity score (0.0-1.0)
        
        Returns:
            Remark string
        """
        if similarity_score >= 0.80:
            return "Good answer. Covers most key points."
        elif similarity_score >= 0.60:
            return "Partial answer. Some key concepts missing."
        elif similarity_score >= 0.40:
            return "Incomplete. Review the topic thoroughly."
        else:
            return "Answer does not match reference. Major revision needed."
    
    def _generate_weak_areas(self, similarity_score: float,
                            missing_keywords: List[str]) -> List[str]:
        """
        Generate list of weak areas based on similarity and missing keywords.
        
        Args:
            similarity_score: Similarity score
            missing_keywords: Missing keywords from reference
        
        Returns:
            List of weak area descriptions
        """
        weak_areas = []
        
        # Add general weak areas based on score
        if similarity_score < 0.40:
            weak_areas.append("Fundamental misunderstanding of concepts")
        elif similarity_score < 0.60:
            weak_areas.append("Incomplete understanding of topic")
            weak_areas.append("Missing important details or examples")
        elif similarity_score < 0.80:
            weak_areas.append("Minor conceptual gaps")
        
        # Add specific areas based on missing keywords
        if missing_keywords:
            if len(missing_keywords) >= 3:
                weak_areas.append(f"Missing key terms: {', '.join(missing_keywords[:3])}")
            else:
                weak_areas.append(f"Missing: {', '.join(missing_keywords)}")
        
        return weak_areas
    
    def generate_all(self, questions_data: Any) -> List[Dict[str, Any]]:
        """
        Generate feedback for all questions.
        
        Args:
            questions_data: Dictionary with structure:
            {
                "Q1": {
                    "similarity": 0.82,
                    "marks": {...},
                    "missing_keywords": [...]
                },
                ...
            }
        
        Returns:
            List of feedback dictionaries
        """
        feedback_list = []

        if isinstance(questions_data, dict):
            iterable = [
                (q_id, data.get("similarity", 0.0), data.get("missing_keywords", []), data.get("marks", {}))
                for q_id, data in questions_data.items()
            ]
        else:
            iterable = questions_data or []

        for item in iterable:
            if isinstance(item, tuple) and len(item) == 4:
                q_id, similarity, missing_keywords, marks = item
            elif isinstance(item, dict):
                q_id = item.get("question_id", item.get("question", ""))
                similarity = item.get("similarity", 0.0)
                missing_keywords = item.get("missing_keywords", item.get("keywords", []))
                marks = item.get("marks", {})
            else:
                continue

            feedback = self.generate(q_id, similarity, missing_keywords, marks)
            feedback_list.append(feedback)
        
        logger.info(f"Feedback generated for {len(feedback_list)} questions")
        return feedback_list
