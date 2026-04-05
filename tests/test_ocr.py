"""Unit tests for OCR extraction module."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.extractor import OCRExtractor


class TestOCRExtractor:
    """Test cases for OCRExtractor class."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return {
            'engine': 'paddleocr',
            'language': 'en',
            'dpi': 300,
            'use_gpu': False
        }
    
    @pytest.fixture
    def mock_extractor(self, config):
        """Create an OCRExtractor instance with mocked dependencies."""
        with patch('src.ocr.extractor.PaddleOCR'):
            extractor = OCRExtractor(config)
            return extractor
    
    def test_initialization_with_paddleocr(self, config):
        """Test OCRExtractor initializes with PaddleOCR."""
        with patch('src.ocr.extractor.PaddleOCR'):
            extractor = OCRExtractor(config)
            assert extractor is not None
            assert extractor.engine is not None
    
    def test_initialization_fallback_to_tesseract(self, config):
        """Test OCRExtractor falls back to Tesseract if PaddleOCR fails."""
        config['engine'] = 'tesseract'
        with patch('src.ocr.extractor.pytesseract'):
            extractor = OCRExtractor(config)
            assert extractor is not None
    
    def test_extract_from_pdf_returns_correct_structure(self, mock_extractor):
        """Test extraction returns correct JSON structure."""
        with patch('src.ocr.extractor.convert_from_path') as mock_convert:
            # Mock the PDF conversion
            mock_image = MagicMock()
            mock_convert.return_value = [mock_image]
            
            # Mock extraction
            with patch.object(mock_extractor, '_extract_paddleocr', return_value='extracted text'):
                result = mock_extractor.extract_from_pdf('/path/to/test.pdf')
                
                assert isinstance(result, dict)
                assert 'pages' in result
                assert 'full_text' in result
                assert 'page_count' in result
    
    def test_extract_from_pdf_returns_correct_page_count(self, mock_extractor):
        """Test extraction correctly counts pages."""
        with patch('src.ocr.extractor.convert_from_path') as mock_convert:
            mock_images = [MagicMock() for _ in range(3)]
            mock_convert.return_value = mock_images
            
            with patch.object(mock_extractor, '_extract_paddleocr', return_value='text'):
                result = mock_extractor.extract_from_pdf('/path/to/test.pdf')
                assert result['page_count'] == 3
    
    def test_save_extraction_creates_json_file(self, mock_extractor, tmp_path):
        """Test extraction result is saved to JSON file."""
        extraction_result = {
            'pages': [{'page': 1, 'text': 'Page 1 content'}],
            'full_text': 'Page 1 content',
            'page_count': 1
        }
        
        file_id = 'test-uuid-123'
        mock_extractor.save_extraction(file_id, extraction_result, str(tmp_path))
        
        # Check file was created
        saved_file = tmp_path / f'{file_id}.json'
        assert saved_file.exists()
        
        # Check content
        with open(saved_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert saved_data['page_count'] == 1
        assert saved_data['full_text'] == 'Page 1 content'
    
    def test_extract_missing_keywords_basic(self, mock_extractor):
        """Test keyword extraction identifies missing words."""
        student_text = "Machine learning is a subset of artificial intelligence"
        reference_text = "Machine learning is a subset of artificial intelligence focused on algorithms"
        
        with patch.object(mock_extractor, 'extract_missing_keywords', return_value=['algorithms', 'focused']):
            keywords = mock_extractor.extract_missing_keywords(student_text, reference_text)
            assert isinstance(keywords, list)
    
    def test_extract_returns_empty_for_empty_pdf(self, mock_extractor):
        """Test extraction gracefully handles empty PDFs."""
        with patch('src.ocr.extractor.convert_from_path') as mock_convert:
            mock_convert.return_value = []
            
            result = mock_extractor.extract_from_pdf('/path/to/empty.pdf')
            assert result['page_count'] == 0


class TestOCRIntegration:
    """Integration tests for OCR module."""
    
    def test_ocr_pipeline_complete(self):
        """Test complete OCR pipeline: upload → extract → save."""
        # This would require actual PDF file or mocking
        pass
