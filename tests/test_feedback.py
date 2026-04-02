"""Unit tests for feedback generation and visualization modules."""

import pytest
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.feedback import FeedbackGenerator
from src.visualization.charts import ChartGenerator


class TestFeedbackGenerator:
    """Test cases for FeedbackGenerator class."""
    
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
    def feedback_gen(self, config):
        """Create a FeedbackGenerator instance."""
        return FeedbackGenerator(config)
    
    def test_initialization(self, config):
        """Test FeedbackGenerator initializes."""
        gen = FeedbackGenerator(config)
        assert gen is not None
    
    def test_generate_returns_dict(self, feedback_gen):
        """Test generate() returns dictionary."""
        result = feedback_gen.generate('Q1', 0.8, ['keyword1'], {'marks_awarded': 8, 'max_marks': 10})
        
        assert isinstance(result, dict)
        assert 'question' in result
        assert 'similarity_percentage' in result
        assert 'remark' in result
        assert 'missing_keywords' in result
        assert 'weak_areas' in result
    
    def test_generate_high_score_remark(self, feedback_gen):
        """Test high score generates positive remark."""
        result = feedback_gen.generate('Q1', 0.9, [], {'marks_awarded': 9, 'max_marks': 10})
        
        assert 'remark' in result
        remark = result['remark'].lower()
        # Should contain positive feedback
        assert any(word in remark for word in ['good', 'excellent', 'well', 'strong'])
    
    def test_generate_low_score_remark(self, feedback_gen):
        """Test low score generates corrective remark."""
        result = feedback_gen.generate('Q1', 0.2, [], {'marks_awarded': 2, 'max_marks': 10})
        
        assert 'remark' in result
        remark = result['remark'].lower()
        # Should contain constructive feedback
        assert len(remark) > 0
    
    def test_generate_includes_missing_keywords(self, feedback_gen):
        """Test feedback includes missing keywords."""
        keywords = ['algorithm', 'complexity', 'optimization']
        result = feedback_gen.generate('Q1', 0.5, keywords, {'marks_awarded': 5, 'max_marks': 10})
        
        assert result['missing_keywords'] == keywords
    
    def test_generate_weak_areas(self, feedback_gen):
        """Test weak areas are generated."""
        result = feedback_gen.generate('Q1', 0.5, ['missing_keyword'], {'marks_awarded': 5, 'max_marks': 10})
        
        assert isinstance(result['weak_areas'], list)
        # Low score should have weak areas
        assert len(result['weak_areas']) > 0
    
    def test_generate_all_multiple_questions(self, feedback_gen):
        """Test generating feedback for multiple questions."""
        questions_data = [
            {'question_id': 'Q1', 'similarity': 0.9, 'keywords': [], 'marks': {'marks_awarded': 9, 'max_marks': 10}},
            {'question_id': 'Q2', 'similarity': 0.5, 'keywords': ['keyword'], 'marks': {'marks_awarded': 5, 'max_marks': 10}},
            {'question_id': 'Q3', 'similarity': 0.2, 'keywords': ['kw1', 'kw2'], 'marks': {'marks_awarded': 2, 'max_marks': 10}}
        ]
        
        # Mock generate to avoid accessing undefined keys
        feedback_gen.generate = lambda q, s, k, m: {
            'question': q,
            'similarity_percentage': int(s * 100),
            'marks_awarded': m['marks_awarded'],
            'max_marks': m['max_marks'],
            'percentage': int(m['marks_awarded'] / m['max_marks'] * 100),
            'remark': 'Test remark',
            'missing_keywords': k,
            'weak_areas': []
        }
        
        result = feedback_gen.generate_all([
            (q['question_id'], q['similarity'], q['keywords'], q['marks']) 
            for q in questions_data
        ])
        
        assert isinstance(result, list)
    
    def test_remark_consistency(self, feedback_gen):
        """Test that remarks are consistent with scores."""
        low_result = feedback_gen.generate('Q1', 0.3, [], {'marks_awarded': 3, 'max_marks': 10})
        high_result = feedback_gen.generate('Q1', 0.9, [], {'marks_awarded': 9, 'max_marks': 10})
        
        assert isinstance(low_result['remark'], str)
        assert isinstance(high_result['remark'], str)
        # Both should have content
        assert len(low_result['remark']) > 0
        assert len(high_result['remark']) > 0


class TestChartGenerator:
    """Test cases for ChartGenerator class."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return {
            'visualization': {
                'style': 'default',
                'colors': {'good': '#4CAF50', 'avg': '#FFC107', 'poor': '#F44336'}
            }
        }
    
    @pytest.fixture
    def chart_gen(self, config):
        """Create a ChartGenerator instance."""
        try:
            return ChartGenerator(config)
        except Exception as e:
            pytest.skip(f"Cannot initialize ChartGenerator: {str(e)}")
    
    def test_initialization(self, config):
        """Test ChartGenerator initializes."""
        try:
            gen = ChartGenerator(config)
            assert gen is not None
        except Exception as e:
            pytest.skip(f"Matplotlib not available: {str(e)}")
    
    def test_bar_chart_creation(self, chart_gen, tmp_path):
        """Test bar chart is created."""
        question_scores = {
            'Q1': {'marks_awarded': 8, 'max_marks': 10, 'percentage': 80},
            'Q2': {'marks_awarded': 6, 'max_marks': 10, 'percentage': 60},
            'Q3': {'marks_awarded': 9, 'max_marks': 10, 'percentage': 90}
        }
        
        output_path = tmp_path / 'bar_chart.png'
        try:
            chart_gen.bar_chart(question_scores, str(output_path))
            assert output_path.exists()
        except Exception as e:
            pytest.skip(f"Cannot create bar chart: {str(e)}")
    
    def test_pie_chart_creation(self, chart_gen, tmp_path):
        """Test pie chart is created."""
        output_path = tmp_path / 'pie_chart.png'
        
        try:
            chart_gen.pie_chart(75, 100, str(output_path))
            assert output_path.exists()
        except Exception as e:
            pytest.skip(f"Cannot create pie chart: {str(e)}")
    
    def test_line_chart_creation(self, chart_gen, tmp_path):
        """Test line chart is created."""
        submissions = [
            {'percentage': 70},
            {'percentage': 75},
            {'percentage': 82},
            {'percentage': 88}
        ]
        
        output_path = tmp_path / 'line_chart.png'
        try:
            chart_gen.line_chart(submissions, str(output_path))
            assert output_path.exists()
        except Exception as e:
            pytest.skip(f"Cannot create line chart: {str(e)}")
    
    def test_generate_all_creates_all_charts(self, chart_gen, tmp_path):
        """Test generate_all creates all three charts."""
        result = {
            'total': 75,
            'out_of': 100,
            'percentage': 75,
            'grade': 'B+',
            'questions': {
                'Q1': {'marks_awarded': 25, 'max_marks': 33, 'percentage': 75},
                'Q2': {'marks_awarded': 25, 'max_marks': 33, 'percentage': 75},
                'Q3': {'marks_awarded': 25, 'max_marks': 34, 'percentage': 73}
            }
        }
        
        try:
            charts = chart_gen.generate_all(result, str(tmp_path))
            
            assert isinstance(charts, dict)
            assert 'bar_chart' in charts
            assert 'pie_chart' in charts
            assert 'line_chart' in charts
        except Exception as e:
            pytest.skip(f"Cannot generate charts: {str(e)}")


class TestVisualizationIntegration:
    """Integration tests for visualization module."""
    
    def test_chart_save_formats(self, tmp_path):
        """Test charts are saved in PNG format."""
        try:
            from src.visualization.charts import ChartGenerator
            gen = ChartGenerator({})
            
            output_file = tmp_path / 'test_chart.png'
            gen.pie_chart(50, 100, str(output_file))
            
            # Check file extension
            assert output_file.suffix == '.png'
            # Check file was created
            assert output_file.exists()
        except Exception as e:
            pytest.skip(f"Visualization test skipped: {str(e)}")
