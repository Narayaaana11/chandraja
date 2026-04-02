# Smart Eval - Run Guide

This file is a quick start guide to run the project locally.

## 1. Prerequisites

- Python 3.9+
- pip
- (Recommended) virtual environment
- Tesseract OCR installed and in PATH (required for OCR workflows)

On Windows, if Tesseract is not in PATH, set it before running the app:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## 2. Open Project Folder

```powershell
cd C:\Users\naray\Downloads\smart_eval_project\smart_eval
```

## 3. Create and Activate Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4. Install Dependencies

Choose one of these based on the app you want to run.

### Option A: Core app (app.py)

```powershell
pip install -r requirements.txt
```

### Option B: Full smart-eval app (smart_eval_app.py)

```powershell
pip install -r smart_eval_requirements.txt
```

Note: Option B includes OCR and DB-related packages and is heavier.

## 5. Run the App

### Option A: Core Flask app

```powershell
python app.py
```

- Runs at: http://127.0.0.1:5000
- Main route: `/`
- APIs: `/evaluate`, `/plagiarism`

### Option B: Full pipeline Flask app

```powershell
python smart_eval_app.py
```

- Runs at host/port from `smart_eval_config.yaml` (default port 5000)
- Main route: `/`
- Additional APIs include upload, OCR, preprocess, evaluate, and pipeline routes

## 6. Train the Scoring Model (Optional)

```powershell
python train_pipeline.py --dataset data/sample_training_data.csv --retrain
```

With run comparison table:

```powershell
python train_pipeline.py --dataset data/sample_training_data.csv --retrain --compare
```

## 7. Visualize Training Metrics (Optional)

```powershell
python visualize_metrics.py --output logs/metrics_trend.png
```

## 8. Verify System (Optional)

```powershell
python verify_system.py
```

## Troubleshooting

- If `ModuleNotFoundError` appears, confirm the virtual environment is active and dependencies are installed.
- If OCR fails, verify Tesseract installation and `TESSERACT_CMD` path.
- If port 5000 is busy, stop the conflicting process or change the app port in code/config.
