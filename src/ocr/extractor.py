"""OCR extraction module using PaddleOCR with Tesseract/EasyOCR fallback."""

import logging
import json
import io
from pathlib import Path
from typing import Dict, List, Any

from PIL import Image

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    PaddleOCR = None
    HAS_PADDLEOCR = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    pytesseract = None
    HAS_TESSERACT = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    easyocr = None
    HAS_EASYOCR = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    convert_from_path = None
    HAS_PDF2IMAGE = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    fitz = None
    HAS_PYMUPDF = False

logger = logging.getLogger(__name__)


class OCRExtractor:
    """Extract text from PDF documents using OCR."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OCR extractor.
        
        Args:
            config: Dictionary with OCR configuration including:
                - engine: "paddleocr" or "tesseract"
                - language: Language code (e.g., "en")
                - dpi: DPI for PDF conversion
                - use_gpu: Whether to use GPU (for PaddleOCR)
        """
        self.config = config
        self.engine = config.get('engine', 'paddleocr')
        self.language = config.get('language', 'en')
        self.dpi = config.get('dpi', 300)
        self.use_gpu = config.get('use_gpu', False)
        self.tesseract_available = False
        self.easyocr_reader = None
        
        self.ocr = None
        self._initialize_engine()
    
    def _initialize_engine(self) -> None:
        """Initialize the OCR engine (PaddleOCR or Tesseract)."""
        if self.engine == 'paddleocr':
            if not HAS_PADDLEOCR:
                logger.warning("PaddleOCR not installed, falling back to Tesseract")
                self.engine = 'tesseract'
            else:
                # PaddleOCR 2.x and 3.x use different initialization arguments.
                # Try a few compatible signatures before falling back to Tesseract.
                paddle_init_options = [
                    {
                        "use_angle_cls": True,
                        "lang": self.language,
                        "use_gpu": self.use_gpu,
                    },
                    {
                        "use_angle_cls": True,
                        "lang": self.language,
                    },
                    {
                        "lang": self.language,
                    },
                ]

                for kwargs in paddle_init_options:
                    try:
                        self.ocr = PaddleOCR(**kwargs)
                        logger.info(f"PaddleOCR initialized successfully with args: {list(kwargs.keys())}")
                        return
                    except Exception as e:
                        logger.warning(f"PaddleOCR init attempt failed for args {list(kwargs.keys())}: {e}")

                logger.error("Failed to initialize PaddleOCR with all supported argument variants")
                self.engine = 'tesseract'

        if self.engine == 'tesseract':
            if HAS_TESSERACT:
                try:
                    # Validate that the native tesseract binary is available, not just pytesseract package.
                    pytesseract.get_tesseract_version()
                    self.tesseract_available = True
                    logger.info("Using Tesseract as OCR engine")
                    return
                except Exception as e:
                    self.tesseract_available = False
                    logger.warning(f"Tesseract binary not available, trying EasyOCR fallback: {e}")
            else:
                logger.warning("pytesseract package not installed, trying EasyOCR fallback")

            if HAS_EASYOCR:
                try:
                    self.easyocr_reader = easyocr.Reader([self.language], gpu=self.use_gpu)
                    self.engine = 'easyocr'
                    logger.info("Using EasyOCR as OCR engine")
                    return
                except Exception as e:
                    logger.warning(f"EasyOCR initialization failed: {e}")

            if not HAS_TESSERACT:
                logger.error("No OCR engine available")
                raise RuntimeError("No OCR engine available")
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with structure:
            {
                "pages": [
                    {"page": 1, "text": "extracted text..."},
                    ...
                ],
                "full_text": "combined text from all pages",
                "page_count": N
            }
        
        Raises:
            FileNotFoundError: If PDF file not found
            RuntimeError: If OCR extraction fails
        """
        pdf_path = Path(pdf_path)
        
        try:
            # Convert PDF pages to images
            logger.info(f"Converting PDF to images: {pdf_path}")
            images = self._convert_pdf_to_images(pdf_path)
            native_text_pages = self._extract_native_pdf_text_pages(pdf_path)
            
            pages = []
            full_text = ""
            
            for page_num, image in enumerate(images, 1):
                logger.info(f"Extracting text from page {page_num}/{len(images)}")
                native_text = native_text_pages[page_num - 1] if page_num - 1 < len(native_text_pages) else ""
                if native_text.strip():
                    text = native_text.strip()
                else:
                    text = self._extract_from_image(image)
                
                pages.append({
                    "page": page_num,
                    "text": text
                })
                full_text += f"\n--- Page {page_num} ---\n{text}\n"

            if images and not any((p.get("text") or "").strip() for p in pages):
                raise RuntimeError(
                    "No text could be extracted from PDF pages. Install Tesseract OCR binary or enable PaddleOCR/EasyOCR."
                )
            
            logger.info(f"Successfully extracted text from {len(images)} pages")
            
            return {
                "pages": pages,
                "full_text": full_text.strip(),
                "page_count": len(images)
            }
        
        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")
            raise RuntimeError(f"PDF extraction failed: {str(e)}")

    def _extract_native_pdf_text_pages(self, pdf_path: Path) -> List[str]:
        """Extract embedded text directly from PDF pages when available."""
        if not HAS_PYMUPDF or fitz is None:
            return []

        pages: List[str] = []
        try:
            with fitz.open(str(pdf_path)) as doc:
                for page in doc:
                    pages.append((page.get_text("text") or "").strip())
        except Exception as e:
            logger.warning(f"Native PDF text extraction failed: {e}")
            return []

        return pages

    def _convert_pdf_to_images(self, pdf_path: Path) -> List[Any]:
        """Convert PDF pages to PIL images using pdf2image or a PyMuPDF fallback."""
        if HAS_PDF2IMAGE and convert_from_path is not None:
            try:
                return convert_from_path(str(pdf_path), dpi=self.dpi)
            except Exception as e:
                logger.warning(f"pdf2image conversion failed, trying PyMuPDF fallback: {e}")

        if not HAS_PYMUPDF or fitz is None:
            raise RuntimeError("Unable to render PDF pages. Install Poppler for pdf2image or install PyMuPDF")

        images = []
        try:
            scale = self.dpi / 72.0
            matrix = fitz.Matrix(scale, scale)
            with fitz.open(str(pdf_path)) as doc:
                for page in doc:
                    pix = page.get_pixmap(matrix=matrix)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    images.append(img.convert("RGB"))
            return images
        except Exception as e:
            raise RuntimeError(f"PyMuPDF conversion failed: {e}") from e

    def extract_missing_keywords(self, student_text: str, reference_text: str) -> List[str]:
        """
        Extract keywords present in reference text but missing in student text.

        Args:
            student_text: Student answer text
            reference_text: Reference answer text

        Returns:
            List of missing keywords, ordered by frequency in reference text
        """
        if not student_text or not reference_text:
            return []

        try:
            def tokenize(text: str) -> List[str]:
                common = {
                    'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were',
                    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                    'did', 'will', 'would', 'could', 'should', 'may', 'might',
                    'can', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with',
                    'from', 'as', 'if', 'but', 'that', 'this', 'which', 'who',
                    'what', 'when', 'where', 'why', 'how'
                }
                return [w for w in text.lower().split() if len(w) > 2 and w not in common]

            student_tokens = set(tokenize(student_text))
            reference_tokens = set(tokenize(reference_text))
            missing = reference_tokens - student_tokens

            return sorted(missing, key=lambda x: reference_text.lower().count(x), reverse=True)[:10]
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def _extract_from_image(self, image) -> str:
        """
        Extract text from image using configured OCR engine.
        
        Args:
            image: Pillow Image object
        
        Returns:
            Extracted text string
        """
        try:
            if self.engine == 'paddleocr' and self.ocr:
                return self._extract_paddleocr(image)
            if self.engine == 'easyocr' and self.easyocr_reader is not None:
                return self._extract_easyocr(image)
            if self.engine == 'tesseract' and not self.tesseract_available:
                return ""
            else:
                return self._extract_tesseract(image)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _extract_paddleocr(self, image) -> str:
        """Extract text using PaddleOCR."""
        try:
            if not HAS_NUMPY or np is None:
                logger.error("numpy is required for PaddleOCR image input")
                return ""

            image_array = np.array(image)
            result = self.ocr.ocr(image_array, cls=True)

            texts: List[str] = []

            def collect_text(node: Any) -> None:
                if isinstance(node, (list, tuple)):
                    # Common Paddle output leaf: ("recognized text", confidence)
                    if len(node) == 2 and isinstance(node[0], str):
                        texts.append(node[0])
                        return
                    for child in node:
                        collect_text(child)

            collect_text(result)
            return " ".join(texts).strip()
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return ""
    
    def _extract_tesseract(self, image) -> str:
        """Extract text using Tesseract."""
        try:
            text = pytesseract.image_to_string(image, lang=self.language)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return ""

    def _extract_easyocr(self, image) -> str:
        """Extract text using EasyOCR."""
        try:
            if not HAS_NUMPY or np is None:
                logger.error("numpy is required for EasyOCR image input")
                return ""

            image_array = np.array(image)
            result = self.easyocr_reader.readtext(image_array, detail=0, paragraph=True)
            return " ".join(result).strip() if result else ""
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""
    
    def save_extraction(self, file_id: str, result: Dict[str, Any], 
                       output_dir: str = "data/extracted") -> str:
        """
        Save extraction result to JSON file.
        
        Args:
            file_id: Unique file identifier
            result: Extraction result dictionary
            output_dir: Directory to save results
        
        Returns:
            Path to saved JSON file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{file_id}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Extraction saved to {filepath}")
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Failed to save extraction: {e}")
            raise
