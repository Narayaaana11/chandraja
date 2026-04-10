"""Unit tests for similarity engine and grading modules."""

import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.similarity import SimilarityEngine
from src.evaluation.grader import Grader


class TestSimilarityEngine:
    """Test cases for SimilarityEngine class."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return {
            'model': 'all-MiniLM-L6-v2',
            'similarity_threshold': 0.5,
            'keyword_matching': True
        }
    
    @pytest.fixture
    def similarity_engine(self, config):
        """Create a SimilarityEngine instance."""
        try:
            return SimilarityEngine(config)
        except Exception as e:
            pytest.skip(f"Cannot initialize SimilarityEngine: {str(e)}")
    
    def test_initialization(self, config):
        """Test SimilarityEngine initializes."""
        try:
            engine = SimilarityEngine(config)
            assert engine is not None
        except Exception as e:
            pytest.skip(f"SentenceTransformer not available: {str(e)}")
    
    def test_identical_texts_return_high_score(self, similarity_engine):
        """Test identical texts return high similarity score."""
        text = "Machine learning is a subset of artificial intelligence"
        score = similarity_engine.compute(text, text)
        
        assert isinstance(score, float)
        assert score >= 0.99  # Identical texts should be very similar
    
    def test_completely_different_texts_return_low_score(self, similarity_engine):
        """Test completely different texts return low similarity score."""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "H2O is water"
        
        score = similarity_engine.compute(text1, text2)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score < 0.5  # Very different texts should have low similarity
    
    def test_similar_texts_return_medium_score(self, similarity_engine):
        """Test similar texts return medium similarity score."""
        text1 = "Machine learning is a field of artificial intelligence"
        text2 = "Machine learning is a subset of artificial intelligence"
        
        score = similarity_engine.compute(text1, text2)
        
        assert isinstance(score, float)
        assert 0.5 < score < 0.99  # Similar but not identical
    
    def test_compute_batch_returns_dict(self, similarity_engine):
        """Test batch computation returns dictionary."""
        student_questions = {'Q1': 'Answer A', 'Q2': 'Answer B'}
        reference_questions = {'Q1': 'Answer A', 'Q2': 'Answer B'}
        
        result = similarity_engine.compute_batch(student_questions, reference_questions)
        
        assert isinstance(result, dict)
        assert 'Q1' in result
        assert 'Q2' in result
    
    def test_compute_batch_handles_missing_questions(self, similarity_engine):
        """Test batch computation handles missing questions gracefully."""
        student_questions = {'Q1': 'Answer A'}
        reference_questions = {'Q1': 'Answer A', 'Q2': 'Answer B'}
        
        result = similarity_engine.compute_batch(student_questions, reference_questions)
        
        assert isinstance(result, dict)

    def test_compute_batch_sanitizes_non_finite_scores(self, similarity_engine, monkeypatch):
        """Test batch computation converts NaN/inf scores to 0.0."""
        monkeypatch.setattr(similarity_engine, 'compute', lambda *_: float('nan'))

        result = similarity_engine.compute_batch({'Q1': 'A'}, {'Q1': 'B'})

        assert result['Q1'] == 0.0
    
    def test_extract_missing_keywords_basic(self, similarity_engine):
        """Test keyword extraction."""
        student_text = "Machine learning uses algorithms"
        reference_text = "Machine learning uses algorithms and neural networks"
        
        keywords = similarity_engine.extract_missing_keywords(student_text, reference_text)
        
        assert isinstance(keywords, list)
        # Should identify missing words
        assert any('neural' in kw.lower() or 'network' in kw.lower() for kw in keywords)
    
    def test_extract_keywords_returns_empty_for_identical(self, similarity_engine):
        """Test keyword extraction returns empty for identical texts."""
        text = "Machine learning algorithms"
        keywords = similarity_engine.extract_missing_keywords(text, text)
        
        assert isinstance(keywords, list)
        assert len(keywords) == 0
    
    def test_similarity_score_range(self, similarity_engine):
        """Test similarity scores are always between 0 and 1."""
        texts = [
            ("hello world", "hello world"),
            ("hello world", "goodbye world"),
            ("hello", "xyz"),
            ("a", "b"),
        ]
        
        for text1, text2 in texts:
            score = similarity_engine.compute(text1, text2)
            assert 0 <= score <= 1


class TestGrader:
    """Test cases for Grader class."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return {
            'total_marks': 100,
            'per_question_marks': 10,
            'partial_scoring': True,
            'partial_scoring_levels': [
                {'threshold': 0.8, 'percentage': 100},
                {'threshold': 0.6, 'percentage': 75},
                {'threshold': 0.4, 'percentage': 50},
                {'threshold': 0.2, 'percentage': 25},
                {'threshold': 0.0, 'percentage': 0}
            ]
        }
    
    @pytest.fixture
    def grader(self, config):
        """Create a Grader instance."""
        return Grader(config)
    
    def test_initialization(self, config):
        """Test Grader initializes."""
        grader = Grader(config)
        assert grader is not None
    
    def test_grade_perfect_score(self, grader):
        """Test grading a perfect answer (similarity 1.0)."""
        result = grader.grade_question(1.0, 10)
        
        assert result['marks_awarded'] == 10
        assert result['percentage'] == 100
        assert result['max_marks'] == 10
    
    def test_grade_zero_score(self, grader):
        """Test grading a completely wrong answer (similarity 0.0)."""
        result = grader.grade_question(0.0, 10)
        
        assert result['marks_awarded'] == 0
        assert result['percentage'] == 0
        assert result['max_marks'] == 10
    
    def test_grade_partial_score_80_percent(self, grader):
        """Test grading with 80% similarity threshold."""
        result = grader.grade_question(0.85, 10)
        
        # Should get 100% of marks at 80%+ similarity
        assert result['marks_awarded'] == 10
        assert result['percentage'] == 100
    
    def test_grade_partial_score_60_percent(self, grader):
        """Test grading with 60% similarity threshold."""
        result = grader.grade_question(0.65, 10)
        
        # Should get 75% of marks at 60-79% similarity
        assert result['marks_awarded'] == 7.5
        assert result['percentage'] == 75
    
    def test_grade_partial_score_40_percent(self, grader):
        """Test grading with 40% similarity threshold."""
        result = grader.grade_question(0.45, 10)
        
        # Should get 50% of marks at 40-59% similarity
        assert result['marks_awarded'] == 5.0
        assert result['percentage'] == 50
    
    def test_calculate_grade_a_plus(self, grader):
        """Test letter grade calculation for A+ (95-100%)."""
        result = grader.grade_question(1.0, 100)  # 100%
        grade = grader._calculate_grade(result['percentage'])
        
        assert grade == 'A+'
    
    def test_calculate_grade_a(self, grader):
        """Test letter grade calculation for A (90-94%)."""
        result = grader.grade_question(0.95, 100)  # 95%
        grade = grader._calculate_grade(result['percentage'])
        
        assert grade in ['A', 'A+']
    
    def test_calculate_grade_b(self, grader):
        """Test letter grade calculation for B (80-89%)."""
        result = grader.grade_question(0.85, 100)  # 85%
        grade = grader._calculate_grade(result['percentage'])
        
        assert grade in ['B', 'B+']
    
    def test_calculate_grade_f(self, grader):
        """Test letter grade calculation for F (<60%)."""
        result = grader.grade_question(0.3, 100)  # 30%
        grade = grader._calculate_grade(result['percentage'])
        
        assert grade == 'F'
    
    def test_grade_all_multiple_questions(self, grader):
        """Test grading multiple questions."""
        similarity_scores = {
            'Q1': 0.9,
            'Q2': 0.7,
            'Q3': 0.4,
            'Q4': 0.2
        }
        per_question_marks = 10
        
        result = grader.grade_all(similarity_scores, per_question_marks)
        
        assert 'total' in result
        assert 'out_of' in result
        assert 'percentage' in result
        assert 'grade' in result
        assert 'questions' in result
        assert len(result['questions']) == 4
    
    def test_grade_all_total_calculation(self, grader):
        """Test total marks calculation."""
        similarity_scores = {
            'Q1': 1.0,  # 10 marks
            'Q2': 0.5   # 5 marks
        }
        per_question_marks = 10
        
        result = grader.grade_all(similarity_scores, per_question_marks)
        
        # 10 + 5 = 15 out of 20
        assert result['total'] == 15
        assert result['out_of'] == 20
        assert result['percentage'] == 75.0
    
    def test_grade_all_returns_correct_structure(self, grader):
        """Test grade_all returns correct JSON structure."""
        similarity_scores = {'Q1': 0.8, 'Q2': 0.6}
        per_question_marks = 10
        
        result = grader.grade_all(similarity_scores, per_question_marks)
        
        assert isinstance(result, dict)
        assert 'questions' in result
        assert isinstance(result['questions'], dict)
        assert 'total' in result
        assert isinstance(result['total'], (int, float))
        assert 'percentage' in result
        assert isinstance(result['percentage'], (int, float))
        assert 'grade' in result
        assert isinstance(result['grade'], str)


class TestGraderThresholds:
    """Test grader threshold configurations."""
    
    def test_custom_thresholds(self):
        """Test custom scoring thresholds."""
        config = {
            'total_marks': 100,
            'per_question_marks': 20,
            'partial_scoring': True,
            'partial_scoring_levels': [
                {'threshold': 0.9, 'percentage': 100},
                {'threshold': 0.7, 'percentage': 80},
                {'threshold': 0.5, 'percentage': 60},
                {'threshold': 0.0, 'percentage': 0}
            ]
        }
        
        grader = Grader(config)
        result = grader.grade_question(0.75, 20)
        
        # 75% similarity should match 70% threshold -> 80% of marks
        assert result['percentage'] == 80
        assert result['marks_awarded'] == 16  # 20 * 80%
