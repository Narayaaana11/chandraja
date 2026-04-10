# Smart Eval

Smart Eval is a Flask-based answer-sheet evaluation system that can:

- Accept answer-sheet and reference PDFs
- Extract text with OCR (native PDF text when available, OCR fallback otherwise)
- Split and align answers by question
- Compute semantic similarity scores
- Generate marks, grade, and feedback
- Save and retrieve evaluation results
- Train a scoring model and use it for prediction utilities

## What You Get

- Web UI for full evaluation flow
- REST API for integration and automation
- Optional MongoDB/MySQL persistence layer
- Training pipeline for model artifacts under models/
- Test suite for API and core modules

## End-to-End Evaluation Pipeline

1. Upload answer sheet and reference
2. OCR extraction
3. Text preprocessing and question splitting
4. Similarity scoring
5. Grading
6. Feedback generation
7. Chart generation
8. Optional database persistence

## Project Structure

- smart_eval_app.py: Flask app and API routes
- smart_eval_config.yaml: Runtime config for web/API evaluation pipeline
- .env: Optional environment variables (HF token, DB URI, etc.)
- train_pipeline.py: End-to-end model training pipeline with dataset hash checks
- scripts/train_model.py: Quick model training command
- scripts/predict_score.py: Score prediction from text inputs
- visualize_metrics.py: Generate MAE/R2 trend plots from run history
- verify_system.py: Environment and route contract checks
- src/: Core OCR, preprocessing, evaluation, DB, and ML modules
- templates/: UI pages
- tests/: Pytest suite
- models/, logs/, results/, uploads/: generated artifacts

## Prerequisites

- Python 3.11+ (project currently runs on Python 3.14 as well)
- pip
- Optional: MongoDB or MySQL if you want persistent result storage
- Optional OCR backends:
  - PaddleOCR (only installed automatically for Python < 3.12 due package constraints)
  - Tesseract binary in PATH for pytesseract backend
  - EasyOCR as a fallback backend (install manually)

Notes:

- The app allows PDF uploads by default (configured in smart_eval_config.yaml).
- First run may download sentence-transformer model weights from Hugging Face.

## Installation

### Windows PowerShell

```powershell
cd C:\Users\naray\Downloads\smart_eval_project\smart_eval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r smart_eval_requirements.txt
```

Optional OCR fallback package:

```powershell
pip install easyocr
```

If you get script execution policy errors while activating venv:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### macOS/Linux

```bash
cd /path/to/smart_eval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r smart_eval_requirements.txt
```

## Configuration

### 1) Evaluation app config: smart_eval_config.yaml

Key sections:

- server: host, port, debug, max upload size
- ocr: engine and OCR settings
- preprocessing: cleaning options
- evaluation: model name, similarity threshold, question delimiter
- grading: total marks and partial scoring thresholds
- database: mongodb/mysql connection settings
- storage: upload/results folders and allowed extensions

Common edits:

- Change host/port under server
- Change database type and connection settings under database
- Adjust allowed file types under storage.allowed_extensions

### 2) Optional environment variables: .env

Included variables:

- TESSERACT_CMD
- HF_TOKEN
- MONGODB_URI
- DB_NAME

Important:

- Running python smart_eval_app.py does not automatically guarantee .env loading unless your environment is configured for it.
- If needed, set env vars in your shell/session directly before launch.

### 3) ML config for training: config.yaml (optional)

ML modules in src/smart_eval/config.py read config.yaml (if present) for training paths/tuning settings and merge with defaults.

If config.yaml is absent, safe defaults are used.

## Run the Project

### 1) Optional system verification

```powershell
python verify_system.py
```

### 2) Start server

```powershell
python smart_eval_app.py
```

Expected local URLs:

- http://127.0.0.1:5000
- http://localhost:5000

### 3) Health check

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/health"
```

### 4) Stop server

Press Ctrl+C in the server terminal.

## How to Use the Web App (Full Flow)

1. Open http://127.0.0.1:5000
2. Upload answer PDF and reference PDF
3. Set subject and max marks
4. Click the evaluate action (UI calls /pipeline/run)
5. Note the returned submission_id
6. Open results page with query param:
   - /results?id=<submission_id>
7. Open feedback page with query param:
   - /feedback?id=<submission_id>

## API Usage

### API Route Reference

- GET /: Main evaluation page
- GET /health: Service readiness and component status
- GET /results: Results page UI (requires ?id=... to load a submission)
- GET /feedback: Feedback page UI (requires ?id=...)
- POST /upload/answer-sheet: Upload answer PDF
- POST /upload/reference: Upload reference PDF (+ optional subject, question_paper_id)
- GET /uploads/<file_id>: Get upload metadata
- POST /ocr/extract: OCR extraction for uploaded file_id
- POST /preprocess: Clean and split extracted text into questions
- POST /evaluate: Evaluate preprocessed answer/reference by file IDs
- GET /results/<submission_id>: Retrieve stored evaluation result
- GET /results/<submission_id>/feedback: Retrieve stored feedback
- POST /pipeline/run: One-shot upload->ocr->preprocess->evaluate pipeline

### One-Shot Pipeline (Recommended)

```powershell
$resp = Invoke-RestMethod -Uri "http://127.0.0.1:5000/pipeline/run" -Method Post -Form @{
  answer_sheet = Get-Item ".\path\to\answer.pdf"
  reference    = Get-Item ".\path\to\reference.pdf"
  subject      = "physics"
  total_marks  = "100"
}

$resp
$resp.submission_id
```

Get results and feedback:

```powershell
$submission = $resp.submission_id
Invoke-RestMethod -Uri "http://127.0.0.1:5000/results/$submission"
Invoke-RestMethod -Uri "http://127.0.0.1:5000/results/$submission/feedback"
```

### Step-by-Step Pipeline (Advanced / Debugging)

1. Upload files

```powershell
$answer = Invoke-RestMethod -Uri "http://127.0.0.1:5000/upload/answer-sheet" -Method Post -Form @{
  file = Get-Item ".\path\to\answer.pdf"
}

$reference = Invoke-RestMethod -Uri "http://127.0.0.1:5000/upload/reference" -Method Post -Form @{
  file              = Get-Item ".\path\to\reference.pdf"
  subject           = "physics"
  question_paper_id = "qp-101"
}
```

2. OCR extraction

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/ocr/extract" -Method Post -ContentType "application/json" -Body (
  @{ file_id = $answer.file_id; file_type = "answer_sheet" } | ConvertTo-Json
)

Invoke-RestMethod -Uri "http://127.0.0.1:5000/ocr/extract" -Method Post -ContentType "application/json" -Body (
  @{ file_id = $reference.file_id; file_type = "reference" } | ConvertTo-Json
)
```

3. Preprocess

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/preprocess" -Method Post -ContentType "application/json" -Body (
  @{ file_id = $answer.file_id } | ConvertTo-Json
)

Invoke-RestMethod -Uri "http://127.0.0.1:5000/preprocess" -Method Post -ContentType "application/json" -Body (
  @{ file_id = $reference.file_id } | ConvertTo-Json
)
```

4. Evaluate

```powershell
$eval = Invoke-RestMethod -Uri "http://127.0.0.1:5000/evaluate" -Method Post -ContentType "application/json" -Body (
  @{
    answer_file_id    = $answer.file_id
    reference_file_id = $reference.file_id
    total_marks       = 100
    subject           = "physics"
  } | ConvertTo-Json
)

$eval
```

## Training and Prediction

### Train with quick script

```powershell
python scripts/train_model.py --dataset data/sample_training_data.csv
```

### Train with full pipeline (hash-aware)

```powershell
python train_pipeline.py --dataset data/sample_training_data.csv
```

Useful flags:

- --force: retrain even if dataset hash has not changed
- --retrain: explicit retrain trigger
- --compare: print historical runs sorted by R2

Example:

```powershell
python train_pipeline.py --dataset data/sample_training_data.csv --force --compare
```

### Predict marks using trained model

```powershell
python scripts/predict_score.py `
  --student "Photosynthesis uses chlorophyll and sunlight." `
  --reference "Plants use sunlight and chlorophyll to produce glucose and oxygen." `
  --keywords "photosynthesis,chlorophyll,glucose,oxygen" `
  --max-marks 10
```

### Visualize training trends

```powershell
python visualize_metrics.py --output logs/metrics_trend.png
```

## Dataset Schema for Model Training

CSV must contain these columns:

- student_text
- reference_text
- keywords
- max_marks
- target_marks

Sample file available at data/sample_training_data.csv.

## Testing

Run all tests:

```powershell
pytest -q
```

Run specific suites:

```powershell
pytest tests/test_app_routes.py -q
pytest tests/test_ocr.py -q
pytest tests/test_similarity.py -q
```

## Output and Artifact Locations

- uploads/: uploaded answer sheets and references
- results/extracted/: OCR output JSON
- results/processed/: cleaned/split question JSON
- results/charts/: generated charts
- models/: trained model artifacts and manifest
- model_artifacts/: legacy model compatibility artifacts
- logs/run_history.json: training run history

## Known Behaviors and Notes

- On Python 3.14, PaddleOCR packages are skipped by requirement markers; OCR falls back to Tesseract or EasyOCR.
- If no OCR backend is available, OCR endpoints return service errors.
- /results/<submission_id> and /results/<submission_id>/feedback require database persistence to have stored the submission.
- The UI references plagiarism in some legacy template code, but no Flask /plagiarism route is currently exposed.

## Troubleshooting

### App starts but first request is slow

Reason: sentence-transformers model download on first run. Wait for model caching to finish.

### OCR fails

- Install Tesseract binary and add it to PATH, or
- Install EasyOCR (pip install easyocr)

### Database not saving results

- Confirm database settings in smart_eval_config.yaml
- Ensure MongoDB/MySQL service is reachable
- Check /health component readiness

### Port already in use

- Change server.port in smart_eval_config.yaml
- Restart app

### Upload returns 415

- File type not allowed by storage.allowed_extensions
- Default is PDF only

## Production Note

This project runs with Flask development server by default (app.run with debug config). For production deployment, use a production WSGI server and harden configuration/security accordingly.
