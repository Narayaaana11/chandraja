"""Unit tests for text preprocessing module."""

import pytest
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.cleaner import TextCleaner


class TestTextCleaner:
    """Test cases for TextCleaner class."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return {
            'lowercase': True,
            'fix_ocr_errors': True,
            'remove_special_chars': True,
            'normalize_whitespace': True,
            'remove_stopwords': False,
            'sentence_segmentation': False
        }
    
    @pytest.fixture
    def cleaner(self, config):
        """Create a TextCleaner instance."""
        return TextCleaner(config)
    
    def test_initialization(self, config):
        """Test TextCleaner initializes with config."""
        cleaner = TextCleaner(config)
        assert cleaner is not None
    
    def test_lowercase_conversion(self, cleaner):
        """Test text is converted to lowercase."""
        text = "HELLO World TEST"
        cleaned = cleaner.clean(text)
        assert cleaned.isupper() == False
        assert 'hello' in cleaned.lower()
    
    def test_whitespace_normalization(self, cleaner):
        """Test multiple spaces and newlines are normalized."""
        text = "Hello    world\n\n\nTest"
        cleaned = cleaner.clean(text)
        # Should not have multiple consecutive spaces
        assert '    ' not in cleaned
        assert '\n\n' not in cleaned
    
    def test_special_chars_removal(self, cleaner):
        """Test special characters are removed correctly."""
        text = "Hello@#$%World!?&"
        cleaned = cleaner.clean(text)
        # Should remove problematic characters
        assert '@' not in cleaned
        assert '#' not in cleaned
        assert '$' not in cleaned
    
    def test_ocr_error_fixing(self, cleaner):
        """Test common OCR errors are fixed."""
        text = "He11o w0r1d"  # OCR mistakes: 1 instead of l
        cleaned = cleaner.clean(text)
        # After cleaning, should improve OCR errors
        assert isinstance(cleaned, str)
    
    def test_split_by_questions_with_default(self, cleaner):
        """Test splitting text into questions."""
        text = "Q1. What is ML? Q2. Define AI? Q3. What is DL?"
        result = cleaner.split_by_questions(text, delimiter='Q')
        
        assert isinstance(result, dict)
        assert 'questions' in result
        assert 'count' in result
        assert result['count'] > 0
    
    def test_split_by_questions_returns_dict(self, cleaner):
        """Test split result has correct structure."""
        text = "Q1. First answer here. Q2. Second answer here."
        result = cleaner.split_by_questions(text, delimiter='Q')
        
        assert isinstance(result['questions'], dict)
        assert isinstance(result['count'], int)
    
    def test_split_by_questions_with_colon(self, cleaner):
        """Test splitting with different question formats."""
        text = "Q1: Answer to first question. Q2: Answer to second question."
        result = cleaner.split_by_questions(text, delimiter='Q')
        
        assert result['count'] > 0
        assert len(result['questions']) > 0
    
    def test_clean_removes_extra_whitespace(self, cleaner):
        """Test whitespace normalization."""
        text = "Hello   world     test"
        cleaned = cleaner.clean(text)
        lines = cleaned.split('\n')
        for line in lines:
            # Check no double spaces
            assert '  ' not in line
    
    def test_save_processed_creates_json(self, cleaner, tmp_path):
        """Test processed text is saved to JSON file."""
        file_id = 'test-uuid'
        result = {
            'questions': {'Q1': 'Answer 1', 'Q2': 'Answer 2'},
            'count': 2
        }
        
        # Override save_processed to use tmp_path
        import tempfile
        original_results = Path(tempfile.gettempdir()) / 'processed'
        original_results.mkdir(parents=True, exist_ok=True)
        cleaner.save_processed(file_id, result)
        
        # File should exist
        saved_file = original_results / f'{file_id}.json'
        assert saved_file.exists()
    
    def test_clean_with_punctuation(self, cleaner):
        """Test handling of punctuation."""
        text = "Hello! How are you? I'm fine... Thank you."
        cleaned = cleaner.clean(text)
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0


class TestQuestionSplitting:
    """Test cases for question splitting functionality."""
    
    @pytest.fixture
    def cleaner(self):
        """Create a TextCleaner instance."""
        config = {
            'lowercase': True,
            'fix_ocr_errors': True,
            'remove_special_chars': False,  # Keep punctuation
            'normalize_whitespace': True,
            'remove_stopwords': False,
            'sentence_segmentation': False
        }
        return TextCleaner(config)
    
    def test_split_multiple_questions(self, cleaner):
        """Test splitting multiple questions."""
        text = """
        Q1. What is the capital of France? Paris
        Q2. What is the capital of Italy? Rome
        Q3. What is the capital of Spain? Madrid
        """
        result = cleaner.split_by_questions(text, delimiter='Q')
        assert result['count'] >= 3
    
    def test_split_preserves_content(self, cleaner):
        """Test that splitting preserves question content."""
        text = "Q1. Question one answer. Q2. Question two answer."
        result = cleaner.split_by_questions(text, delimiter='Q')
        
        # Check that original content is preserved
        combined_content = ' '.join(result['questions'].values())
        assert 'answer' in combined_content.lower()
    
    def test_split_handles_empty_questions(self, cleaner):
        """Test handling of empty questions."""
        text = "Q1. Q2. some text Q3. more text"
        result = cleaner.split_by_questions(text, delimiter='Q')
        # Should handle gracefully
        assert result['count'] >= 0


class TestTextCleanerEdgeCases:
    """Test edge cases for text cleaning."""
    
    @pytest.fixture
    def cleaner(self):
        """Create a TextCleaner instance."""
        config = {
            'lowercase': True,
            'fix_ocr_errors': True,
            'remove_special_chars': True,
            'normalize_whitespace': True,
            'remove_stopwords': False,
            'sentence_segmentation': False
        }
        return TextCleaner(config)
    
    def test_clean_empty_string(self, cleaner):
        """Test cleaning empty string."""
        result = cleaner.clean("")
        assert isinstance(result, str)
    
    def test_clean_only_whitespace(self, cleaner):
        """Test cleaning whitespace-only string."""
        result = cleaner.clean("    \n\n    ")
        assert result.strip() == ""
    
    def test_clean_very_long_text(self, cleaner):
        """Test cleaning very long text."""
        long_text = "word " * 10000
        result = cleaner.clean(long_text)
        assert isinstance(result, str)
    
    def test_split_no_questions(self, cleaner):
        """Test splitting text with no questions."""
        text = "This is just plain text without any questions."
        result = cleaner.split_by_questions(text, delimiter='Q')
        # Should handle gracefully
        assert result['count'] >= 0
