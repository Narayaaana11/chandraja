"""Smart Evaluation System - Flask application entry point."""

import os
import logging
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

import yaml
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Import evaluation modules
try:
    from src.ocr.extractor import OCRExtractor
    HAS_OCR = True
except ImportError:
    logger_import = logging.getLogger(__name__)
    logger_import.warning("OCRExtractor not available")
    HAS_OCR = False

try:
    from src.preprocessing.cleaner import TextCleaner
    HAS_CLEANER = True
except ImportError:
    logger_import = logging.getLogger(__name__)
    logger_import.warning("TextCleaner not available")
    HAS_CLEANER = False

try:
    from src.evaluation.similarity import SimilarityEngine
    HAS_SIMILARITY = True
except Exception as e:
    logger_import = logging.getLogger(__name__)
    logger_import.warning(f"SimilarityEngine not available: {e}")
    HAS_SIMILARITY = False

try:
    from src.evaluation.grader import Grader
    HAS_GRADER = True
except ImportError:
    logger_import = logging.getLogger(__name__)
    logger_import.warning("Grader not available")
    HAS_GRADER = False

try:
    from src.evaluation.feedback import FeedbackGenerator
    HAS_FEEDBACK = True
except ImportError:
    logger_import = logging.getLogger(__name__)
    logger_import.warning("FeedbackGenerator not available")
    HAS_FEEDBACK = False

try:
    from src.visualization.charts import ChartGenerator
    HAS_CHARTS = True
except ImportError:
    logger_import = logging.getLogger(__name__)
    logger_import.warning("ChartGenerator not available")
    HAS_CHARTS = False

try:
    from src.db.database import Database
    HAS_DATABASE = True
except ImportError:
    logger_import = logging.getLogger(__name__)
    logger_import.warning("Database not available")
    HAS_DATABASE = False

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path: str = "smart_eval_config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found")
        raise

# Initialize Flask app
app = Flask(__name__)
CONFIG = load_config()

# Initialize evaluation modules
ocr_extractor = None
text_cleaner = None
similarity_engine = None
grader = None
feedback_generator = None
chart_generator = None
database = None

if HAS_OCR:
    try:
        ocr_extractor = OCRExtractor(CONFIG)
        logger.info("OCRExtractor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize OCRExtractor: {str(e)}")

if HAS_CLEANER:
    try:
        text_cleaner = TextCleaner(CONFIG['preprocessing'])
        logger.info("TextCleaner initialized")
    except Exception as e:
        logger.error(f"Failed to initialize TextCleaner: {str(e)}")

if HAS_SIMILARITY:
    try:
        similarity_engine = SimilarityEngine(CONFIG['evaluation'])
        logger.info("SimilarityEngine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize SimilarityEngine: {str(e)}")

if HAS_GRADER:
    try:
        grader = Grader(CONFIG['grading'])
        logger.info("Grader initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Grader: {str(e)}")

if HAS_FEEDBACK:
    try:
        feedback_generator = FeedbackGenerator(CONFIG['grading'])
        logger.info("FeedbackGenerator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize FeedbackGenerator: {str(e)}")

if HAS_CHARTS:
    try:
        chart_generator = ChartGenerator(CONFIG)
        logger.info("ChartGenerator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize ChartGenerator: {str(e)}")

if HAS_DATABASE:
    try:
        database = Database(CONFIG['database'])
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Database: {str(e)}")

# Set up upload folders
UPLOAD_FOLDER = Path(CONFIG['storage']['upload_folder'])
RESULTS_FOLDER = Path(CONFIG['storage']['results_folder'])
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = CONFIG['server']['max_upload_size_mb'] * 1024 * 1024

ALLOWED_EXTENSIONS = set(CONFIG['storage']['allowed_extensions'])

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_upload(file) -> tuple[bool, str]:
    """
    Validate uploaded file.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file or file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, ""

@app.route('/', methods=['GET'])
def index():
    """Render main frontend page."""
    return render_template('smart_eval_index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "Smart Evaluation System",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@app.route('/results', methods=['GET'])
def results_page():
    """Render evaluation results page."""
    return render_template('smart_eval_results.html')

@app.route('/feedback', methods=['GET'])
def feedback_page():
    """Render detailed feedback page."""
    return render_template('smart_eval_feedback.html')

@app.route('/upload/answer-sheet', methods=['POST'])
def upload_answer_sheet():
    """
    Upload answer sheet PDF.
    
    Returns:
        JSON: {
            "status": "ok" | "error",
            "file_id": "uuid",
            "filename": "original_name",
            "message": "error message if status=error"
        }
    """
    try:
        if 'file' not in request.files:
            logger.warning("Upload attempt without file field")
            return jsonify({"status": "error", "message": "No file field provided"}), 400
        
        file = request.files['file']
        is_valid, error_msg = validate_upload(file)
        
        if not is_valid:
            logger.warning(f"Invalid file upload: {error_msg}")
            return jsonify({"status": "error", "message": error_msg}), 415
        
        # Generate unique file ID and secure filename
        file_id = str(uuid.uuid4())
        original_name = secure_filename(file.filename)
        filename = f"{file_id}_{original_name}"
        
        # Create answer_sheets subdirectory
        answer_sheets_dir = UPLOAD_FOLDER / "answer_sheets"
        answer_sheets_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = answer_sheets_dir / filename
        file.save(str(filepath))

        if HAS_DATABASE and database is not None:
            try:
                database.save_upload(
                    file_id=file_id,
                    filename=original_name,
                    file_type='answer_sheet',
                    subject=None,
                    path=str(filepath),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            except Exception as db_err:
                logger.error(f"Failed to save upload metadata: {str(db_err)}")
        
        logger.info(f"Answer sheet uploaded: {file_id} ({original_name})")
        
        return jsonify({
            "status": "ok",
            "file_id": file_id,
            "filename": original_name,
            "upload_time": datetime.now(timezone.utc).isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error uploading answer sheet: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.route('/upload/reference', methods=['POST'])
def upload_reference():
    """
    Upload reference material (PDF or text).
    
    Form fields:
        - file: PDF or text file
        - subject: (optional) subject name
        - question_paper_id: (optional) question paper ID
    
    Returns:
        JSON: {
            "status": "ok" | "error",
            "file_id": "uuid",
            "filename": "original_name",
            "metadata": {"subject": "...", "question_paper_id": "..."},
            "message": "error message if status=error"
        }
    """
    try:
        if 'file' not in request.files:
            logger.warning("Reference upload attempt without file field")
            return jsonify({"status": "error", "message": "No file field provided"}), 400
        
        file = request.files['file']
        is_valid, error_msg = validate_upload(file)
        
        if not is_valid:
            logger.warning(f"Invalid reference file upload: {error_msg}")
            return jsonify({"status": "error", "message": error_msg}), 415
        
        # Extract metadata from form
        subject = request.form.get('subject', 'general')
        question_paper_id = request.form.get('question_paper_id', '')
        
        # Generate unique file ID and secure filename
        file_id = str(uuid.uuid4())
        original_name = secure_filename(file.filename)
        filename = f"{file_id}_{original_name}"
        
        # Create references subdirectory by subject
        references_dir = UPLOAD_FOLDER / "references" / subject
        references_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = references_dir / filename
        file.save(str(filepath))

        if HAS_DATABASE and database is not None:
            try:
                database.save_upload(
                    file_id=file_id,
                    filename=original_name,
                    file_type='reference',
                    subject=subject,
                    path=str(filepath),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            except Exception as db_err:
                logger.error(f"Failed to save upload metadata: {str(db_err)}")
        
        logger.info(f"Reference uploaded: {file_id} (subject: {subject})")
        
        return jsonify({
            "status": "ok",
            "file_id": file_id,
            "filename": original_name,
            "metadata": {
                "subject": subject,
                "question_paper_id": question_paper_id
            },
            "upload_time": datetime.now(timezone.utc).isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error uploading reference: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.route('/uploads/<file_id>', methods=['GET'])
def get_upload_info(file_id: str):
    """
    Get information about an uploaded file.
    
    Returns:
        JSON: File metadata (filename, upload_time, etc.)
        or 404 if not found
    """
    try:
        # Search in both answer_sheets and references directories
        answer_sheets_dir = UPLOAD_FOLDER / "answer_sheets"
        references_dir = UPLOAD_FOLDER / "references"
        
        # Search through answer sheets
        if answer_sheets_dir.exists():
            for file_path in answer_sheets_dir.glob(f"{file_id}_*"):
                if file_path.is_file():
                    return jsonify({
                        "file_id": file_id,
                        "filename": file_path.name.replace(f"{file_id}_", ""),
                        "type": "answer_sheet",
                        "upload_time": Path(file_path).stat().st_mtime
                    }), 200
        
        # Search through references
        if references_dir.exists():
            for subject_dir in references_dir.iterdir():
                if subject_dir.is_dir():
                    for file_path in subject_dir.glob(f"{file_id}_*"):
                        if file_path.is_file():
                            return jsonify({
                                "file_id": file_id,
                                "filename": file_path.name.replace(f"{file_id}_", ""),
                                "type": "reference",
                                "subject": subject_dir.name,
                                "upload_time": Path(file_path).stat().st_mtime
                            }), 200
        
        logger.warning(f"File not found: {file_id}")
        return jsonify({"status": "error", "message": "File not found"}), 404
    
    except Exception as e:
        logger.error(f"Error retrieving file info: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.route('/ocr/extract', methods=['POST'])
def ocr_extract():
    """
    Extract text from uploaded file using OCR.
    
    Request JSON:
        - file_id: UUID of uploaded file
        - file_type: "answer_sheet" or "reference"
    
    Returns:
        JSON: {
            "status": "ok" | "error",
            "file_id": "...",
            "pages": [...],
            "full_text": "...",
            "page_count": N,
            "preview": "first 200 chars"
        }
    """
    try:
        if not HAS_OCR or ocr_extractor is None:
            return jsonify({"status": "error", "message": "OCR service not available"}), 503

        data = request.get_json(silent=True) or {}
        file_id = data.get('file_id')
        file_type = data.get('file_type', 'answer_sheet')
        
        if not file_id:
            return jsonify({"status": "error", "message": "file_id required"}), 400
        
        # Find the actual file
        actual_file = None
        if file_type == 'reference':
            references_dir = UPLOAD_FOLDER / "references"
            if references_dir.exists():
                for pattern_match in references_dir.rglob(f"{file_id}_*"):
                    if pattern_match.is_file():
                        actual_file = pattern_match
                        break
        else:
            answer_sheets_dir = UPLOAD_FOLDER / "answer_sheets"
            if answer_sheets_dir.exists():
                for pattern_match in answer_sheets_dir.glob(f"{file_id}_*"):
                    if pattern_match.is_file():
                        actual_file = pattern_match
                        break
        
        if not actual_file:
            logger.warning(f"OCR requested for non-existent file: {file_id}")
            return jsonify({"status": "error", "message": "File not found"}), 404
        
        # Extract text using OCR
        logger.info(f"Extracting text from {file_id}")
        result = ocr_extractor.extract_from_pdf(str(actual_file))
        
        # Save extraction to database
        extraction_output = Path(CONFIG['storage']['results_folder']) / 'extracted'
        extraction_output.mkdir(parents=True, exist_ok=True)
        ocr_extractor.save_extraction(file_id, result, str(extraction_output))

        if HAS_DATABASE and database is not None:
            try:
                database.save_extraction(
                    file_id=file_id,
                    pages=result.get('pages', []),
                    full_text=result.get('full_text', ''),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            except Exception as db_err:
                logger.error(f"Failed to save extraction metadata: {str(db_err)}")
        
        # Log extraction info
        logger.info(f"Text extraction complete: {file_id} ({result['page_count']} pages)")
        
        return jsonify({
            "status": "ok",
            "file_id": file_id,
            "pages": result['pages'],
            "full_text": result['full_text'],
            "page_count": result['page_count'],
            "preview": result['full_text'][:200] + "..." if len(result['full_text']) > 200 else result['full_text']
        }), 200
    
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/preprocess', methods=['POST'])
def preprocess():
    """
    Preprocess and split text into questions.
    
    Request JSON:
        - file_id: UUID of extracted file
    
    Returns:
        JSON: {
            "status": "ok" | "error",
            "file_id": "...",
            "question_count": N,
            "questions": {"Q1": "...", "Q2": "..."}
        }
    """
    try:
        if not HAS_CLEANER or text_cleaner is None:
            return jsonify({"status": "error", "message": "Preprocessing service not available"}), 503

        data = request.get_json(silent=True) or {}
        file_id = data.get('file_id')
        
        if not file_id:
            return jsonify({"status": "error", "message": "file_id required"}), 400
        
        # Load extracted text
        extracted_file = Path(CONFIG['storage']['results_folder']) / 'extracted' / f'{file_id}.json'
        if not extracted_file.exists():
            return jsonify({"status": "error", "message": "Extracted file not found"}), 404
        
        with open(extracted_file, 'r', encoding='utf-8') as f:
            extraction = json.load(f)
        
        full_text = extraction['full_text']
        
        # Clean text
        logger.info(f"Preprocessing text for {file_id}")
        cleaned_text = text_cleaner.clean(full_text)
        
        # Split by questions
        delimiter = CONFIG['evaluation'].get('question_delimiter', 'Q')
        questions_result = text_cleaner.split_by_questions(cleaned_text, delimiter=delimiter)
        
        # Save processed text
        processed_output = Path(CONFIG['storage']['results_folder']) / 'processed'
        processed_output.mkdir(parents=True, exist_ok=True)
        text_cleaner.save_processed(file_id, questions_result, str(processed_output))
        
        logger.info(f"Preprocessing complete: {file_id} ({questions_result['count']} questions)")
        
        return jsonify({
            "status": "ok",
            "file_id": file_id,
            "question_count": questions_result['count'],
            "questions": questions_result['questions']
        }), 200
    
    except Exception as e:
        logger.error(f"Error preprocessing: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Full evaluation pipeline: similarity → grading → feedback → charts.
    
    Request JSON:
        - answer_file_id: UUID of preprocessed answer file
        - reference_file_id: UUID of preprocessed reference file
        - subject: (optional) subject name
        - total_marks: (optional) total marks for evaluation
    
    Returns:
        JSON: {
            "status": "ok" | "error",
            "submission_id": "...",
            "answer_file_id": "...",
            "reference_file_id": "...",
            "total": total_marks_achieved,
            "out_of": total_marks_possible,
            "percentage": score_percentage,
            "grade": letter_grade,
            "questions": [...],
            "feedback": [...],
            "charts": {
                "bar_chart": "path",
                "pie_chart": "path",
                "line_chart": "path"
            }
        }
    """
    try:
        if not all([
            HAS_SIMILARITY and similarity_engine is not None,
            HAS_GRADER and grader is not None,
            HAS_FEEDBACK and feedback_generator is not None,
            HAS_CHARTS and chart_generator is not None,
        ]):
            return jsonify({"status": "error", "message": "Evaluation service not fully available"}), 503

        data = request.get_json(silent=True) or {}
        answer_file_id = data.get('answer_file_id')
        reference_file_id = data.get('reference_file_id')
        subject = data.get('subject', 'general')
        total_marks = data.get('total_marks', CONFIG['grading']['total_marks'])
        
        if not answer_file_id or not reference_file_id:
            return jsonify({"status": "error", "message": "answer_file_id and reference_file_id required"}), 400
        
        # Load preprocessed files
        processed_dir = Path(CONFIG['storage']['results_folder']) / 'processed'
        answer_file = processed_dir / f'{answer_file_id}.json'
        reference_file = processed_dir / f'{reference_file_id}.json'
        
        if not answer_file.exists() or not reference_file.exists():
            return jsonify({"status": "error", "message": "Preprocessed files not found"}), 404
        
        with open(answer_file, 'r', encoding='utf-8') as f:
            answer_data = json.load(f)
        
        with open(reference_file, 'r', encoding='utf-8') as f:
            reference_data = json.load(f)
        
        logger.info(f"Starting evaluation: {answer_file_id} vs {reference_file_id}")
        
        # Compute similarity scores
        student_questions = answer_data['questions']
        reference_questions = reference_data['questions']
        
        similarity_scores = similarity_engine.compute_batch(student_questions, reference_questions)
        
        # Grade all questions
        per_question_marks = total_marks / len(student_questions) if student_questions else total_marks
        grading_result = grader.grade_all(similarity_scores, per_question_marks)
        
        # Generate feedback
        feedback_list = []
        for qid, score in similarity_scores.items():
            missing_keywords = similarity_engine.extract_missing_keywords(
                student_questions.get(qid, ''),
                reference_questions.get(qid, '')
            )
            marks_info = grading_result['questions'].get(qid, {})
            feedback_item = feedback_generator.generate(
                qid,
                score,
                missing_keywords,
                marks_info
            )
            feedback_list.append(feedback_item)
        
        # Generate charts
        charts_dir = Path(CONFIG['storage']['results_folder']) / 'charts'
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for charts
        question_scores = {qid: grading_result['questions'][qid] for qid in grading_result['questions']}
        
        charts = chart_generator.generate_all(grading_result, str(charts_dir))
        
        # Create submission ID and save to database
        submission_id = str(uuid.uuid4())
        
        if HAS_DATABASE:
            try:
                database.save_result(
                    submission_id,
                    answer_file_id,
                    reference_file_id,
                    grading_result,
                    feedback_list,
                    datetime.now(timezone.utc).isoformat()
                )
            except Exception as db_err:
                logger.error(f"Database save failed: {str(db_err)}")
        
        logger.info(f"Evaluation complete: {submission_id} (score: {grading_result['percentage']}%)")
        
        return jsonify({
            "status": "ok",
            "submission_id": submission_id,
            "answer_file_id": answer_file_id,
            "reference_file_id": reference_file_id,
            "total": grading_result['total'],
            "out_of": grading_result['out_of'],
            "percentage": grading_result['percentage'],
            "grade": grading_result['grade'],
            "questions": list(grading_result['questions'].keys()),
            "feedback": feedback_list,
            "charts": charts
        }), 200
    
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/results/<submission_id>', methods=['GET'])
def get_results(submission_id: str):
    """
    Get evaluation results for a submission.
    
    Returns:
        JSON: Complete evaluation results with charts and feedback
    """
    try:
        if not HAS_DATABASE:
            return jsonify({"status": "error", "message": "Results service not available"}), 503
        
        result = database.get_result(submission_id)
        
        if not result:
            logger.warning(f"Results not found: {submission_id}")
            return jsonify({"status": "error", "message": "Results not found"}), 404
        
        logger.info(f"Retrieved results: {submission_id}")
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/results/<submission_id>/feedback', methods=['GET'])
def get_feedback(submission_id: str):
    """
    Get detailed feedback for a submission.
    
    Returns:
        JSON: Detailed feedback with remarks, keywords, and weak areas
    """
    try:
        if not HAS_DATABASE:
            return jsonify({"status": "error", "message": "Results service not available"}), 503
        
        result = database.get_result(submission_id)
        
        if not result:
            logger.warning(f"Feedback not found: {submission_id}")
            return jsonify({"status": "error", "message": "Feedback not found"}), 404
        
        logger.info(f"Retrieved feedback: {submission_id}")
        
        return jsonify({
            "submission_id": submission_id,
            "total": result.get('total'),
            "out_of": result.get('out_of'),
            "percentage": result.get('percentage'),
            "grade": result.get('grade'),
            "feedback": result.get('feedback', [])
        }), 200
    
    except Exception as e:
        logger.error(f"Error retrieving feedback: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/pipeline/run', methods=['POST'])
def pipeline_run():
    """
    Full pipeline: Upload → OCR → Preprocess → Evaluate → Results.
    
    Request JSON (multipart form data):
        - answer_sheet: PDF file
        - reference: PDF file
        - subject: (optional) subject name
        - total_marks: (optional) total marks
    
    Returns:
        JSON: Complete evaluation result
    """
    try:
        if not all([
            HAS_OCR and ocr_extractor is not None,
            HAS_CLEANER and text_cleaner is not None,
            HAS_SIMILARITY and similarity_engine is not None,
            HAS_GRADER and grader is not None,
            HAS_FEEDBACK and feedback_generator is not None,
            HAS_CHARTS and chart_generator is not None,
        ]):
            return jsonify({"status": "error", "message": "Pipeline services not fully available"}), 503

        if 'answer_sheet' not in request.files or 'reference' not in request.files:
            return jsonify({"status": "error", "message": "Both answer_sheet and reference files required"}), 400
        
        answer_file = request.files['answer_sheet']
        reference_file = request.files['reference']
        subject = request.form.get('subject', 'general')
        total_marks = int(request.form.get('total_marks', CONFIG['grading']['total_marks']))
        
        # Validate files
        for f in [answer_file, reference_file]:
            is_valid, error_msg = validate_upload(f)
            if not is_valid:
                return jsonify({"status": "error", "message": error_msg}), 415
        
        # Upload answer sheet
        answer_file_id = str(uuid.uuid4())
        answer_filename = f"{answer_file_id}_{secure_filename(answer_file.filename)}"
        answer_sheets_dir = UPLOAD_FOLDER / "answer_sheets"
        answer_sheets_dir.mkdir(parents=True, exist_ok=True)
        answer_path = answer_sheets_dir / answer_filename
        answer_file.save(str(answer_path))

        if HAS_DATABASE and database is not None:
            try:
                database.save_upload(
                    file_id=answer_file_id,
                    filename=secure_filename(answer_file.filename),
                    file_type='answer_sheet',
                    subject=None,
                    path=str(answer_path),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            except Exception as db_err:
                logger.error(f"Failed to save answer upload metadata: {str(db_err)}")
        
        # Upload reference
        reference_file_id = str(uuid.uuid4())
        reference_filename = f"{reference_file_id}_{secure_filename(reference_file.filename)}"
        references_dir = UPLOAD_FOLDER / "references" / subject
        references_dir.mkdir(parents=True, exist_ok=True)
        reference_path = references_dir / reference_filename
        reference_file.save(str(reference_path))

        if HAS_DATABASE and database is not None:
            try:
                database.save_upload(
                    file_id=reference_file_id,
                    filename=secure_filename(reference_file.filename),
                    file_type='reference',
                    subject=subject,
                    path=str(reference_path),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            except Exception as db_err:
                logger.error(f"Failed to save reference upload metadata: {str(db_err)}")
        
        logger.info(f"Pipeline: Uploaded files {answer_file_id}, {reference_file_id}")
        
        # Extract text from both files
        answer_extraction = ocr_extractor.extract_from_pdf(str(answer_path))
        reference_extraction = ocr_extractor.extract_from_pdf(str(reference_path))
        
        extraction_output = Path(CONFIG['storage']['results_folder']) / 'extracted'
        extraction_output.mkdir(parents=True, exist_ok=True)
        ocr_extractor.save_extraction(answer_file_id, answer_extraction, str(extraction_output))
        ocr_extractor.save_extraction(reference_file_id, reference_extraction, str(extraction_output))

        if HAS_DATABASE and database is not None:
            try:
                ts = datetime.now(timezone.utc).isoformat()
                database.save_extraction(
                    file_id=answer_file_id,
                    pages=answer_extraction.get('pages', []),
                    full_text=answer_extraction.get('full_text', ''),
                    timestamp=ts
                )
                database.save_extraction(
                    file_id=reference_file_id,
                    pages=reference_extraction.get('pages', []),
                    full_text=reference_extraction.get('full_text', ''),
                    timestamp=ts
                )
            except Exception as db_err:
                logger.error(f"Failed to save extraction metadata: {str(db_err)}")
        
        logger.info(f"Pipeline: OCR complete")
        
        # Preprocess both texts
        delimiter = CONFIG['evaluation'].get('question_delimiter', 'Q')
        answer_cleaned = text_cleaner.clean(answer_extraction['full_text'])
        answer_questions = text_cleaner.split_by_questions(answer_cleaned, delimiter=delimiter)
        
        reference_cleaned = text_cleaner.clean(reference_extraction['full_text'])
        reference_questions = text_cleaner.split_by_questions(reference_cleaned, delimiter=delimiter)
        
        processed_output = Path(CONFIG['storage']['results_folder']) / 'processed'
        processed_output.mkdir(parents=True, exist_ok=True)
        text_cleaner.save_processed(answer_file_id, answer_questions, str(processed_output))
        text_cleaner.save_processed(reference_file_id, reference_questions, str(processed_output))
        
        logger.info(f"Pipeline: Preprocessing complete")
        
        # Evaluate
        similarity_scores = similarity_engine.compute_batch(
            answer_questions['questions'],
            reference_questions['questions']
        )
        
        per_question_marks = total_marks / len(answer_questions['questions']) if answer_questions['questions'] else total_marks
        grading_result = grader.grade_all(similarity_scores, per_question_marks)
        
        # Generate feedback
        feedback_list = []
        for qid, score in similarity_scores.items():
            missing_keywords = similarity_engine.extract_missing_keywords(
                answer_questions['questions'].get(qid, ''),
                reference_questions['questions'].get(qid, '')
            )
            marks_info = grading_result['questions'].get(qid, {})
            feedback_item = feedback_generator.generate(qid, score, missing_keywords, marks_info)
            feedback_list.append(feedback_item)
        
        # Generate charts
        charts_dir = Path(CONFIG['storage']['results_folder']) / 'charts'
        charts_dir.mkdir(parents=True, exist_ok=True)
        charts = chart_generator.generate_all(grading_result, str(charts_dir))
        
        # Save to database
        submission_id = str(uuid.uuid4())
        if HAS_DATABASE:
            try:
                database.save_result(
                    submission_id,
                    answer_file_id,
                    reference_file_id,
                    grading_result,
                    feedback_list,
                    datetime.now(timezone.utc).isoformat()
                )
            except Exception as db_err:
                logger.error(f"Database save failed: {str(db_err)}")
        
        logger.info(f"Pipeline: Complete - {submission_id}")
        
        return jsonify({
            "status": "ok",
            "submission_id": submission_id,
            "total": grading_result['total'],
            "out_of": grading_result['out_of'],
            "percentage": grading_result['percentage'],
            "grade": grading_result['grade'],
            "feedback": feedback_list,
            "charts": charts
        }), 200
    
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    max_size = CONFIG['server']['max_upload_size_mb']
    return jsonify({
        "status": "error",
        "message": f"File too large. Maximum size: {max_size}MB"
    }), 413

if __name__ == '__main__':
    host = CONFIG['server']['host']
    port = CONFIG['server']['port']
    debug = CONFIG['server']['debug']
    
    logger.info(f"Starting Smart Evaluation System on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
