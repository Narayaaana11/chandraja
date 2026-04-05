"""Grading engine with partial scoring support."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Grader:
    """Grade answers based on similarity scores with partial scoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize grader.
        
        Args:
            config: Dictionary with grading configuration including:
                - total_marks: Total marks for all questions
                - per_question_marks: Marks per question
                - partial_scoring: Whether to apply partial scoring
                - partial_scoring_levels: List of threshold/percentage pairs
        """
        self.config = config
        self.total_marks = config.get('total_marks', 100)
        self.per_question_marks = config.get('per_question_marks', 10)
        self.partial_scoring = config.get('partial_scoring', True)
        
        self.scoring_levels = config.get('partial_scoring_levels', [
            {'threshold': 0.80, 'percentage': 100},
            {'threshold': 0.60, 'percentage': 75},
            {'threshold': 0.40, 'percentage': 50},
            {'threshold': 0.20, 'percentage': 25},
            {'threshold': 0.0, 'percentage': 0}
        ])
        
        # Sort levels by threshold (descending)
        self.scoring_levels = sorted(
            self.scoring_levels,
            key=lambda x: x['threshold'],
            reverse=True
        )
    
    def grade_question(self, similarity_score: float, max_marks: int) -> Dict[str, Any]:
        """
        Grade a single question based on similarity score.
        
        Args:
            similarity_score: Similarity score between 0.0 and 1.0
            max_marks: Maximum marks for this question
        
        Returns:
            Dictionary with structure:
            {
                "marks_awarded": 8,
                "max_marks": 10,
                "percentage": 80,
                "similarity": 0.82
            }
        """
        if not self.partial_scoring:
            # Binary grading: full marks if above threshold, zero otherwise
            marks = max_marks if similarity_score >= 0.5 else 0
            percentage = 100 if similarity_score >= 0.5 else 0
        else:
            # Partial grading based on similarity levels
            if max_marks >= 100:
                # Preserve intuitive semantics when max marks itself is percentage-like.
                percentage = round(similarity_score * 100, 2)
                marks = round(max_marks * similarity_score, 2)
            else:
                percentage = self._get_percentage_for_score(similarity_score)
                marks = round(max_marks * percentage / 100, 2)
        
        return {
            "marks_awarded": marks,
            "max_marks": max_marks,
            "percentage": percentage,
            "similarity": round(similarity_score, 4)
        }
    
    def _get_percentage_for_score(self, similarity_score: float) -> int:
        """
        Get percentage marks for a similarity score based on thresholds.
        
        Args:
            similarity_score: Score between 0.0 and 1.0
        
        Returns:
            Percentage score (0-100)
        """
        for level in self.scoring_levels:
            if similarity_score >= level['threshold']:
                return level['percentage']
        
        return 0
    
    def grade_all(self, similarity_scores: Dict[str, float],
                 per_question_marks: int = None) -> Dict[str, Any]:
        """
        Grade all questions and calculate total score.
        
        Args:
            similarity_scores: Dictionary of {question_id: similarity_score}
            per_question_marks: Marks per question (uses config if None)
        
        Returns:
            Dictionary with structure:
            {
                "questions": {
                    "Q1": {"marks_awarded": 8, "max_marks": 10, ...},
                    ...
                },
                "total": 78,
                "out_of": 100,
                "percentage": 78.0,
                "grade": "B+"
            }
        """
        if per_question_marks is None:
            per_question_marks = self.per_question_marks
        
        question_results = {}
        total_marks = 0
        total_max_marks = 0
        
        for q_id, similarity in similarity_scores.items():
            result = self.grade_question(similarity, per_question_marks)
            question_results[q_id] = result
            total_marks += result['marks_awarded']
            total_max_marks += result['max_marks']

        # Guard against tiny rounding drift that can exceed configured totals.
        total_marks = min(total_marks, total_max_marks)
        
        percentage = (total_marks / total_max_marks * 100) if total_max_marks > 0 else 0
        percentage = min(percentage, 100.0)
        grade = self._calculate_grade(percentage)
        
        logger.info(f"Grading complete: {total_marks}/{total_max_marks} ({percentage:.1f}%)")
        
        return {
            "questions": question_results,
            "total": total_marks,
            "out_of": total_max_marks,
            "percentage": round(percentage, 2),
            "grade": grade
        }
    
    def _calculate_grade(self, percentage: float) -> str:
        """
        Calculate letter grade from percentage.
        
        Args:
            percentage: Percentage score
        
        Returns:
            Letter grade (A+, A, B+, B, C+, C, D, F)
        """
        if percentage >= 95:
            return "A+"
        elif percentage >= 90:
            return "A"
        elif percentage >= 85:
            return "B+"
        elif percentage >= 80:
            return "B"
        elif percentage >= 75:
            return "C+"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"
