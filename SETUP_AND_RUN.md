# Smart Eval Setup and Run Guide

This guide covers local setup, running the app, validation, and pre-push checks.

## 1) Prerequisites

- Python 3.9+
- pip
- Tesseract OCR installed and available in PATH (recommended for OCR routes)

Optional on Windows if Tesseract is not in PATH:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## 2) Open the Project

```powershell
git clone https://github.com/Narayaaana11/chandraja.git
cd chandraja
```

## 3) Create and Activate Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4) Install Dependencies

Choose the mode you want.

### Mode A: Core App (`app.py`)

```powershell
pip install -r requirements.txt
```

### Mode B: Full Pipeline App (`smart_eval_app.py`)

```powershell
pip install -r smart_eval_requirements.txt
```

## 5) Run the Application

### Core App

```powershell
python app.py
```

- URL: `http://127.0.0.1:5000`
- Main APIs: `/evaluate`, `/plagiarism`

### Full Pipeline App

```powershell
python smart_eval_app.py
```

- URL: from `smart_eval_config.yaml` (`server.host` + `server.port`)
- Main APIs: `/upload/answer-sheet`, `/upload/reference`, `/ocr/extract`, `/preprocess`, `/evaluate`, `/pipeline/run`

## 6) Validate the Project

Run tests:

```powershell
python -m pytest -q
```

Run system verifier:

```powershell
python verify_system.py
```

## 7) Optional: Train and Visualize

Train:

```powershell
python train_pipeline.py --dataset data/sample_training_data.csv --retrain
```

Compare runs:

```powershell
python train_pipeline.py --dataset data/sample_training_data.csv --retrain --compare
```

Generate metric chart:

```powershell
python visualize_metrics.py --output logs/metrics_trend.png
```

## 8) Pre-Push Checklist

Run these before pushing:

```powershell
python -m ruff check . --select F401,F841
python -m pytest -q
python verify_system.py
```

## 9) GitHub Public Repository Checklist

- Add a license file (`LICENSE`)
- Keep security reporting instructions in `SECURITY.md`
- Keep contribution guidance in `CONTRIBUTING.md`
- Keep behavior expectations in `CODE_OF_CONDUCT.md`
- Ensure no local secrets are committed (`.env` remains ignored)

Then commit and push:

```powershell
git add .
git commit -m "chore: cleanup and validation updates"
git push
```
