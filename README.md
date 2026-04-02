# Smart Evaluation System for Handwritten Answer Sheets

A complete AI-powered solution for automatically evaluating handwritten answer sheets using OCR, semantic similarity analysis, and intelligent feedback generation.

## Features

- **Automatic OCR Processing**: Extract text from handwritten answer sheets using PaddleOCR with Tesseract fallback
- **Text Preprocessing**: Clean, normalize, and segment extracted text into questions
- **Semantic Similarity Analysis**: Compare student answers with reference answers using transformer models
- **Intelligent Grading**: Partial scoring system with configurable thresholds
- **Feedback Generation**: Question-wise feedback with remarks, missing keywords, and improvement areas
- **Visualization**: Auto-generated charts (bar, pie, line) for comprehensive results
- **Database Integration**: MongoDB and MySQL support for result persistence
- **Web Interface**: Responsive HTML UI for uploads and results viewing

## System Architecture

### Phase 1: Configuration & Setup

- `smart_eval_config.yaml`: Complete configuration for all modules
- `smart_eval_requirements.txt`: All dependencies (17 packages)
- Project directory structure with modular organization

### Phase 2: File Upload API

- `smart_eval_app.py` - Flask application with:
  - `POST /upload/answer-sheet`: Upload student answer sheets
  - `POST /upload/reference`: Upload reference materials
  - `GET /uploads/<file_id>`: Retrieve file metadata

### Phase 3: OCR Text Extraction

- `src/ocr/extractor.py` - OCRExtractor class:
  - PDF to image conversion (DPI 300)
  - PaddleOCR extraction with GPU support option
  - Tesseract fallback if PaddleOCR unavailable
  - Page-by-page processing with JSON persistence

### Phase 4: Text Preprocessing

- `src/preprocessing/cleaner.py` - TextCleaner class:
  - Lowercase conversion
  - OCR error fixing (regex patterns)
  - Special character removal
  - Whitespace normalization
  - Stopword removal (NLTK)
  - Question splitting by delimiter

### Phase 5: Evaluation Engine

#### 5A - Similarity Engine

- `src/evaluation/similarity.py`:
  - SentenceTransformer-based semantic similarity
  - Batch processing for efficiency
  - Missing keyword extraction

#### 5B - Grading System

- `src/evaluation/grader.py`:
  - Threshold-based partial scoring (0.8→100%, 0.6→75%, 0.4→50%, 0.2→25%, 0.0→0%)
  - Automatic letter grade assignment (A+ to F)
  - Total score calculation

#### 5C - Feedback Generation

- `src/evaluation/feedback.py`:
  - Question-wise feedback with remarks (4 tiers)
  - Missing keywords extraction
  - Weak areas identification

### Phase 6: Visualization

- `src/visualization/charts.py` - ChartGenerator:
  - Bar charts (marks comparison)
  - Pie charts (score distribution)
  - Line charts (performance trends)
  - PNG output at 300 DPI

### Phase 7: Database Layer

- `src/db/database.py` - Database class:
  - MongoDB support (primary)
  - MySQL support (fallback)
  - Methods: save_upload, save_extraction, save_result, get_result, get_submissions_by_subject

### Phase 8: Frontend Templates

- `templates/smart_eval_index.html`: File upload interface with AJAX pipeline
- `templates/smart_eval_results.html`: Results dashboard with stats and charts
- `templates/smart_eval_feedback.html`: Detailed feedback cards per question

### Phase 9: Test Suite

- `tests/test_ocr.py`: OCR extraction tests
- `tests/test_preprocessing.py`: Text cleaning and splitting tests
- `tests/test_similarity.py`: Similarity engine and grading tests
- `tests/test_feedback.py`: Feedback generation and chart creation tests

### Phase 10: API Routes

Complete Flask API with integrated pipeline:

- `POST /ocr/extract`: Text extraction from file
- `POST /preprocess`: Text cleaning and question splitting
- `POST /evaluate`: Full evaluation (similarity → grading → feedback → charts)
- `GET /results/<submission_id>`: Retrieve evaluation results
- `GET /results/<submission_id>/feedback`: Get detailed feedback
- `POST /pipeline/run`: End-to-end pipeline in single request

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

1. **Clone/Extract the Project**

```bash
cd smart_eval
```

2. **Create Virtual Environment**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r smart_eval_requirements.txt
```

### Configuration

Edit `smart_eval_config.yaml` to customize:

- OCR engine (paddleocr/tesseract)
- Model name for similarity analysis
- Scoring thresholds
- Database connection
- Upload folder locations

## Usage

### Starting the Server

```bash
python smart_eval_app.py
```

Server runs on `http://localhost:5000` by default (configurable in config.yaml)

### Web Interface

1. **Upload Interface** (`http://localhost:5000/`)
   - Upload answer sheet PDF
   - Upload reference material PDF
   - Enter subject and total marks
   - Click "Evaluate Answers"

2. **Results Dashboard** (`/results`)
   - View overall score and grade
   - See question-wise marks
   - View auto-generated charts
   - Access detailed feedback

3. **Feedback Page** (`/feedback?id=<submission_id>`)
   - Question-by-question analysis
   - Remarks and suggestions
   - Missing keywords highlighted
   - Areas for improvement

### API Usage

#### Full Pipeline (Single Request)

```bash
curl -X POST http://localhost:5000/pipeline/run \
  -F "answer_sheet=@student_answers.pdf" \
  -F "reference=@model_answers.pdf" \
  -F "subject=Mathematics" \
  -F "total_marks=100"
```

#### Step-by-Step Pipeline

1. **Upload Files**

```bash
curl -X POST http://localhost:5000/upload/answer-sheet \
  -F "file=@student_answers.pdf"
# Returns: {"file_id": "UUID", ...}

curl -X POST http://localhost:5000/upload/reference \
  -F "file=@model_answers.pdf" \
  -F "subject=Mathematics"
# Returns: {"file_id": "UUID", ...}
```

2. **Extract Text**

```bash
curl -X POST http://localhost:5000/ocr/extract \
  -H "Content-Type: application/json" \
  -d '{"file_id": "UUID", "file_type": "answer_sheet"}'
# Returns: Extracted text with page count
```

3. **Preprocess Text**

```bash
curl -X POST http://localhost:5000/preprocess \
  -H "Content-Type: application/json" \
  -d '{"file_id": "UUID"}'
# Returns: Cleaned text split into questions
```

4. **Evaluate**

```bash
curl -X POST http://localhost:5000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "answer_file_id": "UUID1",
    "reference_file_id": "UUID2",
    "subject": "Mathematics",
    "total_marks": 100
  }'
# Returns: Complete evaluation with feedback and charts
```

5. **Retrieve Results**

```bash
curl http://localhost:5000/results/<submission_id>
# Returns: Full results JSON

curl http://localhost:5000/results/<submission_id>/feedback
# Returns: Detailed feedback per question
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ocr.py -v

# Run with coverage
pytest tests/ --cov=src/
```

## Configuration Options

### Server Settings

```yaml
server:
  host: 0.0.0.0
  port: 5000
  debug: false
  max_upload_size_mb: 20
```

### OCR Engine

```yaml
ocr:
  engine: paddleocr # or tesseract
  language: en
  dpi: 300
  use_gpu: false
```

### Evaluation

```yaml
evaluation:
  model: all-MiniLM-L6-v2 # SentenceTransformer model
  similarity_threshold: 0.5
  question_delimiter: Q
```

### Grading

```yaml
grading:
  total_marks: 100
  partial_scoring: true
  partial_scoring_levels:
    - threshold: 0.8
      percentage: 100
    - threshold: 0.6
      percentage: 75
    # ... more thresholds
```

### Database

```yaml
database:
  type: mongodb # or mysql
  mongodb:
    uri: mongodb://localhost:27017
    database: smart_eval
  mysql:
    host: localhost
    user: root
    password: password
    database: smart_eval
```

## Project Structure

```
smart_eval/
├── smart_eval_app.py              # Flask application
├── smart_eval_config.yaml         # Configuration file
├── smart_eval_requirements.txt    # Dependencies
├── templates/
│   ├── smart_eval_index.html      # Upload interface
│   ├── smart_eval_results.html    # Results dashboard
│   └── smart_eval_feedback.html   # Feedback cards
├── static/                        # Static files (CSS, JS)
├── uploads/
│   ├── answer_sheets/            # Student answer PDFs
│   └── references/               # Reference material
├── data/
│   ├── extracted/               # OCR extraction results
│   └── processed/               # Preprocessed text
├── results/
│   ├── extracted/              # Extraction metadata
│   ├── processed/              # Processing results
│   └── charts/                 # Generated visualization
├── src/
│   ├── ocr/
│   │   ├── __init__.py
│   │   └── extractor.py       # OCR extraction
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── cleaner.py         # Text cleaning
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── similarity.py      # Semantic similarity
│   │   ├── grader.py          # Grading engine
│   │   └── feedback.py        # Feedback generation
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── charts.py          # Chart generation
│   └── db/
│       ├── __init__.py
│       └── database.py        # Database abstraction
├── tests/
│   ├── __init__.py
│   ├── test_ocr.py
│   ├── test_preprocessing.py
│   ├── test_similarity.py
│   └── test_feedback.py
└── README.md                   # This file
```

## Error Handling & Fallbacks

### Graceful Degradation

- PaddleOCR → Tesseract (if PaddleOCR unavailable)
- MongoDB → MySQL (if MongoDB unavailable)
- SentenceTransformer loaded from cache or huggingface

### Logging

All operations logged to console/file:

- OCR processing
- Text preprocessing
- Evaluation pipeline
- Database operations
- Chart generation

### Recovery

- Failed stages return descriptive error messages
- Partial results available if evaluation incomplete
- File uploads validated before processing

## Performance Considerations

- **OCR**: GPU acceleration available via PaddleOCR `use_gpu: true`
- **Similarity**: Batch processing for multiple questions
- **Charts**: Generated on-demand, cached in results/charts/
- **Database**: Indexed queries for fast retrieval

## Security

- **File Uploads**:
  - Secure filename generation
  - File type validation (PDF only)
  - 20MB size limit (configurable)

- **File Paths**:
  - UUID-based file IDs in responses
  - Never expose raw file paths in API

- **Database**:
  - Connection strings in config (not hardcoded)
  - Credentials should use environment variables for production

## Troubleshooting

### PaddleOCR Installation Issues

```bash
pip install paddlepaddle paddleocr --no-deps
pip install pillow numpy
```

### SentenceTransformer Model Download

First run will download ~80MB model automatically. Requires internet connection.

```bash
# Pre-download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### MongoDB Connection Failed

```bash
# Check MongoDB is running
mongosh  # or mongo

# Use MySQL instead in config.yaml
database:
  type: mysql
```

### Port Already in Use

Change port in config.yaml:

```yaml
server:
  port: 8000 # Different port
```

## Design Patterns Used

1. **Factory Pattern**: Engine initialization (OCR, Database)
2. **Strategy Pattern**: Pluggable extraction engines
3. **Repository Pattern**: Database abstraction
4. **Chain of Responsibility**: Text cleaning pipeline
5. **Template Method**: Save operations with consistent structure

## Future Enhancements

- [ ] Handwriting quality assessment
- [ ] Answer sheet layout detection
- [ ] Student identity verification
- [ ] Batch evaluation dashboard
- [ ] Performance analytics and trends
- [ ] REST API documentation (Swagger)
- [ ] Docker containerization
- [ ] Authentication and authorization

## Contributing

1. Follow existing code style and patterns
2. Add tests for new features
3. Update configuration examples
4. Document API changes

## License

MIT 2. If needed on Windows, set:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Inputs

The loader supports:

1. CSV files
2. JSON and JSONL files
3. Labeled folders where each subfolder name is the label and files are text/markdown

For score regression training, expected columns are:

1. student_text
2. reference_text
3. keywords
4. max_marks
5. target_marks

## End-to-End Training

Run full pipeline (load, preprocess, split, train, evaluate, version, best-model update):

```bash
python train_pipeline.py --dataset data/sample_training_data.csv --retrain
```

Optional run comparison table:

```bash
python train_pipeline.py --dataset data/sample_training_data.csv --retrain --compare
```

## Metrics Visualization

```bash
python visualize_metrics.py --output logs/metrics_trend.png
```

## Inference CLI

```bash
python scripts/predict_score.py --student "your answer" --reference "expected answer" --keywords "k1,k2,k3" --max-marks 10
```

## Flask Serving

```bash
python app.py
```

Serving uses models/best_model.pkl by default and falls back to legacy model_artifacts files if needed. API routes and response schema stay compatible.
