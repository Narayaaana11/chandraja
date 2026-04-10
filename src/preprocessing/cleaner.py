"""Text preprocessing and cleaning module."""

import re
import logging
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    import nltk
    from nltk.corpus import stopwords
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and preprocess text."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize text cleaner.
        
        Args:
            config: Dictionary with preprocessing configuration including:
                - lowercase: Whether to convert to lowercase
                - remove_special_chars: Whether to remove special characters
                - normalize_whitespace: Whether to normalize whitespace
                - remove_stopwords: Whether to remove stopwords
                - sentence_segmentation: Whether to segment into sentences
        """
        self.config = config
        self.lowercase = config.get('lowercase', True)
        self.remove_special_chars = config.get('remove_special_chars', True)
        self.normalize_whitespace = config.get('normalize_whitespace', True)
        self.remove_stopwords = config.get('remove_stopwords', False)
        self.sentence_segmentation = config.get('sentence_segmentation', False)
        
        if self.remove_stopwords and HAS_NLTK:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()
    
    def clean(self, text: str) -> str:
        """
        Clean text applying configured transformations.
        
        Args:
            text: Raw text to clean
        
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Step 1: Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Step 2: Fix common OCR errors
        text = self._fix_ocr_errors(text)
        
        # Step 3: Remove special characters (optionally)
        if self.remove_special_chars:
            text = self._remove_special_chars(text)
        
        # Step 4: Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        # Step 5: Remove stopwords (optionally)
        if self.remove_stopwords and self.stop_words:
            text = self._remove_stopwords(text)
        
        return text.strip()
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors."""
        # Fix common OCR mistakes
        replacements = {
            r'\bl\b': '1',  # lowercase L to 1 in isolated context
            r'0O': '00',     # zero followed by O
            r'O0': '00',     # O followed by zero
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """
        Remove special characters, keeping alphanumeric and basic punctuation.
        """
        # Keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\-\(\)\'"]', '', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace - collapse multiple spaces and newlines."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        return text
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove English stopwords."""
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def split_by_questions(self, text: str, delimiter: str = "Q") -> Dict[str, Any]:
        """
        Split text into questions based on delimiter.
        
        Args:
            text: Text containing multiple questions
            delimiter: Question prefix (e.g., "Q" for "Q1:", "Q2:")
        
        Returns:
            Dictionary with structure:
            {
                "questions": {
                    "Q1": "answer text...",
                    "Q2": "answer text...",
                    ...
                },
                "count": N
            }
        
        Examples:
            Input: "Q1. What is AI? AI is... Q2. Define ML? ML is..."
            Output: {
                "questions": {
                    "Q1": "What is AI? AI is...",
                    "Q2": "Define ML? ML is..."
                },
                "count": 2
            }
        """
        questions = {}

        # Support noisy OCR variants such as "q.no.1", "question 1", "q 1", etc.
        patterns = [
            rf'\b{re.escape(delimiter)}\s*\.?\s*no\.?\s*([1-9]\d*)\b',
            r'\bquestion\s*([1-9]\d*)\b',
            rf'\b{re.escape(delimiter)}\s*([1-9]\d*)\b',
        ]

        matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                q_num = match.group(1).strip() if match.group(1) else ""
                matches.append((match.start(), match.end(), q_num))

        # Line-based question prompt pattern (e.g., "1. Discuss ...") helps with
        # exam/workbook files that also contain numbered sub-points like "1. TF".
        if len(matches) <= 1:
            prompt_pattern = re.compile(
                r'(?mi)^\s*(\d{1,2})\.\s+(.{8,120})$'
            )
            prompt_leads = re.compile(
                r'(?i)^(discuss|explain|define|describe|what|why|how|write|elaborate|differentiate|compare|list|state|give|analyze|mention|briefly)\b'
            )

            heading_matches = []
            for match in prompt_pattern.finditer(text):
                q_num = match.group(1).strip()
                line_text = match.group(2).strip()
                if prompt_leads.match(line_text):
                    heading_matches.append((match.start(), match.end(), q_num))

            if len(heading_matches) >= 2:
                matches = heading_matches

        # Fallback for reference/workbook formats like "1. ... 2. ...".
        if not matches:
            numbered_matches = []
            for match in re.finditer(r'(?<!\w)(\d{1,3})\s*[\.)\:\-]\s+', text):
                q_num = match.group(1).strip()
                numbered_matches.append((match.start(), match.end(), q_num))

            if self._looks_like_numbered_questions(numbered_matches):
                matches.extend(numbered_matches)

        # Secondary fallback for OCR dumps that preserve page markers.
        if not matches:
            for match in re.finditer(r'---\s*page\s*(\d+)\s*---', text, flags=re.IGNORECASE):
                matches.append((match.start(), match.end(), match.group(1).strip()))

        # Fallback to original strict pattern if no flexible matches are found.
        if not matches:
            pattern = rf'{re.escape(delimiter)}(\d+)[\.\:\s]+'
            parts = re.split(pattern, text, flags=re.IGNORECASE)
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    q_num = parts[i].strip()
                    q_text = parts[i + 1].strip()
                    if q_text:
                        questions[f"{delimiter}{q_num}"] = q_text

            return {
                "questions": questions,
                "count": len(questions)
            }

        # Keep first match at a given start position to avoid duplicate pattern captures.
        unique_starts = {}
        for start, end, q_num in sorted(matches, key=lambda x: (x[0], -(x[1] - x[0]))):
            if start not in unique_starts:
                unique_starts[start] = (end, q_num)

        ordered = [(start, data[0], data[1]) for start, data in sorted(unique_starts.items(), key=lambda x: x[0])]

        auto_index = 1
        for i, (_, end, q_num) in enumerate(ordered):
            next_start = ordered[i + 1][0] if i + 1 < len(ordered) else len(text)
            q_text = text[end:next_start].strip()
            if q_text:
                if not q_num:
                    q_num = str(auto_index)
                    auto_index += 1
                q_id = f"{delimiter}{q_num}"
                if q_id in questions:
                    questions[q_id] = f"{questions[q_id]} {q_text}".strip()
                else:
                    questions[q_id] = q_text
        
        return {
            "questions": questions,
            "count": len(questions)
        }

    def _looks_like_numbered_questions(self, matches: List[Tuple[int, int, str]]) -> bool:
        """Heuristic to avoid treating arbitrary numbers as question delimiters."""
        if not matches:
            return False

        numbers = [int(item[2]) for item in matches if item[2].isdigit()]
        if not numbers:
            return False

        if len(numbers) == 1:
            return numbers[0] == 1

        starts_like_questions = numbers[0] <= 3
        monotonic_steps = sum(1 for i in range(1, len(numbers)) if numbers[i] >= numbers[i - 1])
        close_steps = sum(
            1 for i in range(1, len(numbers))
            if numbers[i] >= numbers[i - 1] and (numbers[i] - numbers[i - 1]) <= 2
        )

        mostly_monotonic = monotonic_steps >= max(1, len(numbers) - 2)
        mostly_close = close_steps >= max(1, len(numbers) - 2)

        return starts_like_questions and mostly_monotonic and mostly_close
    
    def save_processed(self, file_id: str, result: Dict[str, Any],
                      output_dir: str = None) -> str:
        """
        Save preprocessing result to JSON file.
        
        Args:
            file_id: Unique file identifier
            result: Processing result dictionary
            output_dir: Directory to save results
        
        Returns:
            Path to saved JSON file
        """
        if output_dir is None:
            output_dir = str(Path(tempfile.gettempdir()) / 'processed')

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{file_id}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed text saved to {filepath}")
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Failed to save processed text: {e}")
            raise
