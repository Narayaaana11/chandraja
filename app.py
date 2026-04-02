import logging
import os
import sys
import uuid
from pathlib import Path

import pytesseract
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_eval.config import UPLOAD_DIR
from smart_eval.services.evaluation import (
    ContentScorer,
    SemanticScorer,
    detect_plagiarism,
    get_presentation_score,
    parse_keywords_from_form,
    preprocess_and_ocr,
)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def save_upload(file_storage):
    filename = secure_filename(file_storage.filename or "upload.png")
    if not filename:
        filename = "upload.png"
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    path = UPLOAD_DIR / unique_name
    file_storage.save(str(path))
    return str(path)


semantic_scorer = SemanticScorer()
content_scorer = ContentScorer(semantic_scorer)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        file = request.files.get("image")
        reference = request.form.get("reference", "")
        keywords_raw = request.form.get("keywords", "")
        max_marks = int(request.form.get("max_marks", 10))
        keywords = parse_keywords_from_form(keywords_raw)

        if not file:
            return jsonify({"error": "No image uploaded"}), 400
        if not reference:
            return jsonify({"error": "No reference answer provided"}), 400

        # Save and process image
        path = save_upload(file)

        # OCR
        extracted_text, image, gray = preprocess_and_ocr(path)

        # Scores
        content = content_scorer.get_content_score(extracted_text, reference, keywords, max_marks)
        presentation = get_presentation_score(image, gray)
        total = round(content["marks"] + presentation["total"], 2)

        # Grade
        pct = (total / (max_marks + 2)) * 100
        if pct >= 90: grade = "A+"
        elif pct >= 80: grade = "A"
        elif pct >= 70: grade = "B"
        elif pct >= 60: grade = "C"
        elif pct >= 50: grade = "D"
        else: grade = "F"

        return jsonify({
            "extracted_text": extracted_text,
            "content": content,
            "presentation": presentation,
            "total_score": total,
            "max_marks": max_marks + 2,
            "grade": grade,
            "semantic_mode": semantic_scorer.mode,
            "content_mode": content["content_mode"],
            "model_info": content.get("model_info", {}),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/plagiarism", methods=["POST"])
def plagiarism():
    try:
        files = request.files.getlist("images")
        if len(files) < 2:
            return jsonify({"error": "Upload at least 2 answer sheets"}), 400

        texts = []
        names = []
        for f in files:
            path = save_upload(f)
            text, _, _ = preprocess_and_ocr(path)
            texts.append(text)
            names.append(f.filename)

        results = detect_plagiarism(texts, names, semantic_scorer)
        return jsonify({"results": results, "students": names, "texts": texts})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
