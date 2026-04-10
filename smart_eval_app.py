"""Smart Evaluation System - Flask application entry point."""

import logging
import json
import uuid
import re
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

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


def _require_positive_number(value: Any, field_name: str) -> float:
    """Validate and normalize a positive numeric config value."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a positive number") from None

    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{field_name} must be a positive number")

    return numeric


def validate_config(config: Any) -> dict:
    """Validate required configuration shape and critical value ranges."""
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a YAML mapping")

    required_sections = ("server", "storage", "grading", "evaluation")
    missing_sections = [section for section in required_sections if not isinstance(config.get(section), dict)]
    if missing_sections:
        raise ValueError(f"Missing or invalid config sections: {', '.join(missing_sections)}")

    server_cfg = config["server"]
    storage_cfg = config["storage"]
    grading_cfg = config["grading"]
    evaluation_cfg = config["evaluation"]

    server_cfg["max_upload_size_mb"] = _require_positive_number(
        server_cfg.get("max_upload_size_mb"), "server.max_upload_size_mb"
    )
    grading_cfg["total_marks"] = _require_positive_number(
        grading_cfg.get("total_marks"), "grading.total_marks"
    )

    similarity_threshold = evaluation_cfg.get("similarity_threshold", 0.5)
    try:
        threshold_value = float(similarity_threshold)
    except (TypeError, ValueError):
        raise ValueError("evaluation.similarity_threshold must be between 0 and 1") from None
    if not math.isfinite(threshold_value) or threshold_value < 0 or threshold_value > 1:
        raise ValueError("evaluation.similarity_threshold must be between 0 and 1")
    evaluation_cfg["similarity_threshold"] = threshold_value

    question_delimiter = evaluation_cfg.get("question_delimiter", "Q")
    if not isinstance(question_delimiter, str) or not question_delimiter.strip():
        raise ValueError("evaluation.question_delimiter must be a non-empty string")
    evaluation_cfg["question_delimiter"] = question_delimiter

    allowed_extensions = storage_cfg.get("allowed_extensions")
    if (
        not isinstance(allowed_extensions, list)
        or not allowed_extensions
        or any(not isinstance(ext, str) or not ext.strip() for ext in allowed_extensions)
    ):
        raise ValueError("storage.allowed_extensions must be a non-empty list of strings")
    storage_cfg["allowed_extensions"] = [ext.strip().lower() for ext in allowed_extensions]

    for folder_key in ("upload_folder", "results_folder"):
        folder_value = storage_cfg.get(folder_key)
        if not isinstance(folder_value, str) or not folder_value.strip():
            raise ValueError(f"storage.{folder_key} must be a non-empty string")

    return config


# Load configuration
def load_config(config_path: str = "smart_eval_config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"Invalid YAML in {config_path}: {str(exc)}")
        raise ValueError(f"Config file {config_path} contains invalid YAML") from exc

    try:
        return validate_config(loaded)
    except ValueError as exc:
        logger.error(f"Invalid configuration in {config_path}: {str(exc)}")
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
SAFE_FILE_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,128}$')

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


def validate_file_id(file_id: str) -> tuple[bool, str]:
    """Validate file identifier used in filesystem lookups."""
    if not file_id:
        return False, "file_id required"
    if not SAFE_FILE_ID_PATTERN.fullmatch(str(file_id)):
        return False, "Invalid file_id format"
    return True, ""


def parse_total_marks(raw_value: Any, default_value: Any) -> tuple[float | None, str]:
    """Parse and validate total marks from request payload."""
    candidate = default_value if raw_value in (None, "") else raw_value

    try:
        marks = float(candidate)
    except (TypeError, ValueError):
        return None, "total_marks must be a positive number"

    if not math.isfinite(marks) or marks <= 0:
        return None, "total_marks must be a positive number"

    return marks, ""


def load_json_dict(path: Path) -> tuple[dict | None, str]:
    """Load JSON file and ensure it contains a top-level dictionary."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except json.JSONDecodeError:
        return None, "invalid_json"
    except OSError:
        return None, "io_error"

    if not isinstance(payload, dict):
        return None, "invalid_payload"

    return payload, ""


def add_persistence_warning(warnings: list[dict], operation: str, reason: str) -> None:
    """Append a structured persistence warning for partial-degradation responses."""
    warning = {
        "operation": operation,
        "message": "Database persistence failed",
        "reason": reason,
    }
    warnings.append(warning)


def attach_persistence_warnings(payload: dict, warnings: list[dict]) -> dict:
    """Attach persistence warning metadata to successful API responses."""
    if warnings:
        payload["persistence_warning"] = "Some metadata could not be persisted"
        payload["persistence_warnings"] = warnings
    return payload


def get_component_status() -> dict:
    """Report availability of core runtime components for readiness checks."""
    components = {
        "ocr": {"enabled": bool(HAS_OCR), "initialized": ocr_extractor is not None},
        "preprocessing": {"enabled": bool(HAS_CLEANER), "initialized": text_cleaner is not None},
        "similarity": {"enabled": bool(HAS_SIMILARITY), "initialized": similarity_engine is not None},
        "grader": {"enabled": bool(HAS_GRADER), "initialized": grader is not None},
        "feedback": {"enabled": bool(HAS_FEEDBACK), "initialized": feedback_generator is not None},
        "charts": {"enabled": bool(HAS_CHARTS), "initialized": chart_generator is not None},
        "database": {"enabled": bool(HAS_DATABASE), "initialized": database is not None},
    }

    for details in components.values():
        details["available"] = details["enabled"] and details["initialized"]

    return components


def align_reference_questions(student_questions: dict, reference_questions: dict) -> tuple[dict, str]:
    """
    Align reference questions to student question IDs.

    If question IDs don't overlap (e.g., OCR parsed answer as Q1/Q2 and
    reference as 1/2 or R1/R2), fall back to positional pairing.

    Returns:
        Tuple of (aligned_reference_questions, alignment_mode)
    """
    if not student_questions or not reference_questions:
        return reference_questions, "direct"

    student_keys = list(student_questions.keys())
    reference_keys = list(reference_questions.keys())

    if set(student_keys).intersection(reference_keys):
        return reference_questions, "direct"

    aligned_reference_questions = {}
    reference_values = list(reference_questions.values())

    for idx, student_key in enumerate(student_keys):
        aligned_reference_questions[student_key] = reference_values[idx] if idx < len(reference_values) else ""

    logger.warning(
        "No overlapping question IDs detected between answer and reference; using positional alignment fallback"
    )
    return aligned_reference_questions, "positional_fallback"


def get_comparable_question_maps(student_questions: dict, aligned_reference_questions: dict) -> tuple[dict, dict]:
    """Return only question pairs where reference text is present and non-empty."""
    comparable_student = {}
    comparable_reference = {}

    for qid, student_text in (student_questions or {}).items():
        ref_text = aligned_reference_questions.get(qid, "")
        if isinstance(ref_text, str) and ref_text.strip():
            comparable_student[qid] = student_text
            comparable_reference[qid] = ref_text

    return comparable_student, comparable_reference


def _is_meaningful_question_text(text: str, min_chars: int = 28, min_words: int = 6) -> bool:
    """Filter OCR fragments like "space for rough work" from matching candidates."""
    if not isinstance(text, str):
        return False

    stripped = text.strip()
    if len(stripped) < min_chars:
        return False

    tokens = re.findall(r"[a-zA-Z0-9]+", stripped.lower())
    return len(tokens) >= min_words


def _pair_similarity(student_text: str, reference_text: str, sim_engine: Any) -> float:
    """Compute text similarity with engine fallback to lexical overlap."""
    if not student_text or not reference_text:
        return 0.0

    if sim_engine is not None and hasattr(sim_engine, 'compute'):
        try:
            score = float(sim_engine.compute(student_text, reference_text))
            if score < 0.0:
                return 0.0
            if score > 1.0:
                return 1.0
            return score
        except Exception:
            pass

    student_tokens = set(re.findall(r"[a-zA-Z0-9]+", student_text.lower()))
    reference_tokens = set(re.findall(r"[a-zA-Z0-9]+", reference_text.lower()))
    if not student_tokens or not reference_tokens:
        return 0.0

    intersection = len(student_tokens & reference_tokens)
    union = len(student_tokens | reference_tokens)
    jaccard = intersection / union if union else 0.0
    coverage = intersection / len(reference_tokens) if reference_tokens else 0.0
    return max(0.0, min(1.0, 0.6 * jaccard + 0.4 * coverage))


def _semantic_reference_to_student_match(student_questions: dict,
                                         reference_questions: dict,
                                         sim_engine: Any) -> tuple[dict, dict]:
    """Match each reference question to one or more semantically related student fragments."""
    student_candidates = [
        (qid, text)
        for qid, text in (student_questions or {}).items()
        if isinstance(text, str) and text.strip() and _is_meaningful_question_text(text)
    ]
    reference_candidates = [
        (qid, text)
        for qid, text in (reference_questions or {}).items()
        if isinstance(text, str) and text.strip() and _is_meaningful_question_text(text, min_chars=20, min_words=4)
    ]

    # Keep system resilient when OCR output is short/noisy.
    if not student_candidates:
        student_candidates = [
            (qid, text)
            for qid, text in (student_questions or {}).items()
            if isinstance(text, str) and text.strip()
        ]
    if not reference_candidates:
        reference_candidates = [
            (qid, text)
            for qid, text in (reference_questions or {}).items()
            if isinstance(text, str) and text.strip()
        ]

    if not student_candidates or not reference_candidates:
        return {}, {}

    student_map = {qid: text for qid, text in student_candidates}
    reference_map = {qid: text for qid, text in reference_candidates}

    # Precompute pair scores for assignment and fallback lookups.
    student_to_refs = {qid: [] for qid, _ in student_candidates}
    reference_to_students = {qid: [] for qid, _ in reference_candidates}
    for ref_qid, ref_text in reference_candidates:
        for stu_qid, stu_text in student_candidates:
            score = _pair_similarity(stu_text, ref_text, sim_engine)
            student_to_refs[stu_qid].append((score, ref_qid))
            reference_to_students[ref_qid].append((score, stu_qid))

    # Assign each student fragment to the reference it best matches.
    assigned_students = {qid: [] for qid, _ in reference_candidates}
    for stu_qid, _ in student_candidates:
        ranked_refs = sorted(student_to_refs.get(stu_qid, []), key=lambda x: x[0], reverse=True)
        if not ranked_refs:
            continue
        best_score, best_ref_qid = ranked_refs[0]
        assigned_students[best_ref_qid].append((best_score, stu_qid))

    student_order = {qid: idx for idx, (qid, _) in enumerate(student_candidates)}
    max_fragments_per_reference = max(
        1,
        min(3, int(round(len(student_candidates) / max(1, len(reference_candidates)))))
    )

    matched_student = {}
    matched_reference = {}

    for ref_qid, _ in reference_candidates:
        candidate_students = sorted(
            assigned_students.get(ref_qid, []),
            key=lambda x: x[0],
            reverse=True
        )

        if not candidate_students:
            # Fallback to the globally best single student fragment for this reference.
            global_best = sorted(reference_to_students.get(ref_qid, []), key=lambda x: x[0], reverse=True)
            if global_best:
                candidate_students = [global_best[0]]

        selected_student_ids = []
        if candidate_students:
            top_score = candidate_students[0][0]
            for idx, (score, stu_qid) in enumerate(candidate_students):
                if idx == 0:
                    selected_student_ids.append(stu_qid)
                    continue

                if len(selected_student_ids) >= max_fragments_per_reference:
                    break

                # Include additional fragments when likely related; keep this
                # permissive because OCR fragments can be noisy and incomplete.
                if score >= 0.12 or (top_score > 0 and score >= top_score * 0.45):
                    selected_student_ids.append(stu_qid)

        if not selected_student_ids:
            continue

        selected_student_ids = sorted(
            list(dict.fromkeys(selected_student_ids)),
            key=lambda qid: student_order.get(qid, 10**9)
        )
        merged_student_text = "\n".join(
            student_map[qid].strip() for qid in selected_student_ids if student_map.get(qid, "").strip()
        ).strip()

        if merged_student_text:
            matched_student[ref_qid] = merged_student_text
            matched_reference[ref_qid] = reference_map[ref_qid]

    # Ensure every reference has at least one matched student fragment.
    for ref_qid, _ in reference_candidates:
        if ref_qid in matched_student:
            continue
        best_candidates = sorted(reference_to_students.get(ref_qid, []), key=lambda x: x[0], reverse=True)
        for _, stu_qid in best_candidates:
            fallback_text = student_map.get(stu_qid, "").strip()
            if fallback_text:
                matched_student[ref_qid] = fallback_text
                matched_reference[ref_qid] = reference_map[ref_qid]
                break

    return matched_student, matched_reference


def select_comparable_question_pairs(student_questions: dict,
                                     reference_questions: dict,
                                     sim_engine: Any) -> tuple[dict, dict, str]:
    """
    Select robust comparable pairs for grading.

    Uses direct/positional pairing for normal cases and semantic fallback when
    answer/reference question counts are highly imbalanced.
    """
    aligned_reference_questions, alignment_mode = align_reference_questions(student_questions, reference_questions)
    comparable_student, comparable_reference = get_comparable_question_maps(
        student_questions,
        aligned_reference_questions
    )

    if not comparable_student:
        return {}, {}, alignment_mode

    # Trigger semantic remap when answer OCR creates many extra pseudo-questions.
    if len(student_questions or {}) > len(reference_questions or {}) + 1 and len(reference_questions or {}) >= 2:
        semantic_student, semantic_reference = _semantic_reference_to_student_match(
            student_questions,
            reference_questions,
            sim_engine
        )
        if semantic_student:
            logger.warning(
                "Question count mismatch detected (%s answer vs %s reference); using semantic alignment fallback",
                len(student_questions or {}),
                len(reference_questions or {})
            )
            return semantic_student, semantic_reference, "semantic_fallback"

    return comparable_student, comparable_reference, alignment_mode


def apply_semantic_fallback_score_adjustment(similarity_scores: dict,
                                             alignment_mode: str,
                                             answer_question_count: int,
                                             reference_question_count: int) -> dict:
    """
    Compensate similarity scores in semantic-fallback mode for noisy OCR splits.

    This adjustment is intentionally scoped to semantic fallback so direct and
    positional evaluation behavior remains unchanged.
    """
    if alignment_mode != "semantic_fallback" or not similarity_scores:
        return similarity_scores

    eval_cfg = CONFIG.get("evaluation", {})
    boost_cfg = eval_cfg.get("semantic_fallback_boost", {})

    base_multiplier = float(boost_cfg.get("multiplier", 1.85))
    intercept = float(boost_cfg.get("intercept", 0.05))
    max_multiplier = float(boost_cfg.get("max_multiplier", 2.10))
    low_score_threshold = float(boost_cfg.get("low_score_threshold", 0.12))
    low_score_cap = float(boost_cfg.get("low_score_cap", 0.18))

    ratio = (answer_question_count / reference_question_count) if reference_question_count else 1.0
    ratio_bonus = max(0.0, ratio - 2.0) * 0.10
    multiplier = min(max_multiplier, max(1.0, base_multiplier + ratio_bonus))

    adjusted_scores = {}
    for qid, score in (similarity_scores or {}).items():
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = 0.0

        boosted_score = numeric_score * multiplier + intercept

        # Keep clearly unrelated answers from receiving excessive uplift.
        if numeric_score < low_score_threshold:
            boosted_score = min(boosted_score, low_score_cap)

        adjusted_scores[qid] = max(0.0, min(1.0, boosted_score))

    logger.warning(
        "Applying semantic fallback score adjustment (multiplier=%.2f, intercept=%.2f)",
        multiplier,
        intercept,
    )
    return adjusted_scores

@app.route('/', methods=['GET'])
def index():
    """Render main frontend page."""
    return render_template('smart_eval_index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    component_status = get_component_status()
    pipeline_components = ("ocr", "preprocessing", "similarity", "grader", "feedback", "charts")
    pipeline_ready = all(component_status[name]["available"] for name in pipeline_components)

    return jsonify({
        "status": "ok",
        "service": "Smart Evaluation System",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_ready": pipeline_ready,
        "components": component_status,
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

        persistence_warnings = []

        if HAS_DATABASE:
            if database is None:
                add_persistence_warning(persistence_warnings, 'save_upload_answer_sheet', 'database_not_initialized')
            else:
                try:
                    saved = database.save_upload(
                        file_id=file_id,
                        filename=original_name,
                        file_type='answer_sheet',
                        subject=None,
                        path=str(filepath),
                        timestamp=datetime.now(timezone.utc).isoformat()
                    )
                    if not saved:
                        add_persistence_warning(persistence_warnings, 'save_upload_answer_sheet', 'database_returned_false')
                except Exception as db_err:
                    logger.error(f"Failed to save upload metadata: {str(db_err)}")
                    add_persistence_warning(persistence_warnings, 'save_upload_answer_sheet', 'database_exception')
        
        logger.info(f"Answer sheet uploaded: {file_id} ({original_name})")
        
        response_payload = {
            "status": "ok",
            "file_id": file_id,
            "filename": original_name,
            "upload_time": datetime.now(timezone.utc).isoformat()
        }

        return jsonify(attach_persistence_warnings(response_payload, persistence_warnings)), 200
    
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
        subject = secure_filename(request.form.get('subject', 'general')) or 'general'
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

        persistence_warnings = []

        if HAS_DATABASE:
            if database is None:
                add_persistence_warning(persistence_warnings, 'save_upload_reference', 'database_not_initialized')
            else:
                try:
                    saved = database.save_upload(
                        file_id=file_id,
                        filename=original_name,
                        file_type='reference',
                        subject=subject,
                        path=str(filepath),
                        timestamp=datetime.now(timezone.utc).isoformat()
                    )
                    if not saved:
                        add_persistence_warning(persistence_warnings, 'save_upload_reference', 'database_returned_false')
                except Exception as db_err:
                    logger.error(f"Failed to save upload metadata: {str(db_err)}")
                    add_persistence_warning(persistence_warnings, 'save_upload_reference', 'database_exception')
        
        logger.info(f"Reference uploaded: {file_id} (subject: {subject})")
        
        response_payload = {
            "status": "ok",
            "file_id": file_id,
            "filename": original_name,
            "metadata": {
                "subject": subject,
                "question_paper_id": question_paper_id
            },
            "upload_time": datetime.now(timezone.utc).isoformat()
        }

        return jsonify(attach_persistence_warnings(response_payload, persistence_warnings)), 200
    
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
        is_valid_id, id_error = validate_file_id(file_id)
        if not is_valid_id:
            return jsonify({"status": "error", "message": id_error}), 400

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

        is_valid_id, id_error = validate_file_id(file_id)
        if not is_valid_id:
            return jsonify({"status": "error", "message": id_error}), 400

        if file_type not in ('answer_sheet', 'reference'):
            return jsonify({"status": "error", "message": "file_type must be 'answer_sheet' or 'reference'"}), 400

        persistence_warnings = []
        
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

        if HAS_DATABASE:
            if database is None:
                add_persistence_warning(persistence_warnings, 'save_extraction', 'database_not_initialized')
            else:
                try:
                    saved = database.save_extraction(
                        file_id=file_id,
                        pages=result.get('pages', []),
                        full_text=result.get('full_text', ''),
                        timestamp=datetime.now(timezone.utc).isoformat()
                    )
                    if not saved:
                        add_persistence_warning(persistence_warnings, 'save_extraction', 'database_returned_false')
                except Exception as db_err:
                    logger.error(f"Failed to save extraction metadata: {str(db_err)}")
                    add_persistence_warning(persistence_warnings, 'save_extraction', 'database_exception')
        
        # Log extraction info
        logger.info(f"Text extraction complete: {file_id} ({result['page_count']} pages)")
        
        response_payload = {
            "status": "ok",
            "file_id": file_id,
            "pages": result['pages'],
            "full_text": result['full_text'],
            "page_count": result['page_count'],
            "preview": result['full_text'][:200] + "..." if len(result['full_text']) > 200 else result['full_text']
        }

        return jsonify(attach_persistence_warnings(response_payload, persistence_warnings)), 200
    
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

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

        is_valid_id, id_error = validate_file_id(file_id)
        if not is_valid_id:
            return jsonify({"status": "error", "message": id_error}), 400
        
        # Load extracted text
        extracted_file = Path(CONFIG['storage']['results_folder']) / 'extracted' / f'{file_id}.json'
        if not extracted_file.exists():
            return jsonify({"status": "error", "message": "Extracted file not found"}), 404

        extraction, load_status = load_json_dict(extracted_file)
        if load_status in ('invalid_json', 'invalid_payload'):
            return jsonify({"status": "error", "message": "Extracted file is not valid JSON"}), 422
        if load_status == 'io_error':
            return jsonify({"status": "error", "message": "Failed to read extracted file"}), 500

        full_text = extraction.get('full_text') if extraction else None
        if not isinstance(full_text, str):
            return jsonify({"status": "error", "message": "Extracted file missing full_text"}), 422
        
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
        return jsonify({"status": "error", "message": "Internal server error"}), 500

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
        
        if not answer_file_id or not reference_file_id:
            return jsonify({"status": "error", "message": "answer_file_id and reference_file_id required"}), 400

        for file_id in [answer_file_id, reference_file_id]:
            is_valid_id, id_error = validate_file_id(file_id)
            if not is_valid_id:
                return jsonify({"status": "error", "message": id_error}), 400

        total_marks, marks_error = parse_total_marks(
            data.get('total_marks'),
            CONFIG['grading']['total_marks']
        )
        if marks_error:
            return jsonify({"status": "error", "message": marks_error}), 400

        persistence_warnings = []
        
        # Load preprocessed files
        processed_dir = Path(CONFIG['storage']['results_folder']) / 'processed'
        answer_file = processed_dir / f'{answer_file_id}.json'
        reference_file = processed_dir / f'{reference_file_id}.json'
        
        if not answer_file.exists() or not reference_file.exists():
            return jsonify({"status": "error", "message": "Preprocessed files not found"}), 404
        
        answer_data, answer_load_status = load_json_dict(answer_file)
        reference_data, reference_load_status = load_json_dict(reference_file)
        if answer_load_status in ('invalid_json', 'invalid_payload') or reference_load_status in ('invalid_json', 'invalid_payload'):
            return jsonify({"status": "error", "message": "Preprocessed files are not valid JSON"}), 422
        if answer_load_status == 'io_error' or reference_load_status == 'io_error':
            return jsonify({"status": "error", "message": "Failed to read preprocessed files"}), 500
        
        logger.info(f"Starting evaluation: {answer_file_id} vs {reference_file_id}")
        
        # Compute similarity scores
        student_questions = answer_data.get('questions', {}) if answer_data else {}
        reference_questions = reference_data.get('questions', {}) if reference_data else {}
        if not isinstance(student_questions, dict) or not isinstance(reference_questions, dict):
            return jsonify({"status": "error", "message": "Preprocessed files must contain questions as an object"}), 422

        comparable_student_questions, comparable_reference_questions, alignment_mode = select_comparable_question_pairs(
            student_questions,
            reference_questions,
            similarity_engine
        )

        if not comparable_student_questions:
            return jsonify({
                "status": "error",
                "message": "No comparable answer/reference question pairs found after preprocessing"
            }), 422
        
        similarity_scores = similarity_engine.compute_batch(comparable_student_questions, comparable_reference_questions)
        similarity_scores = apply_semantic_fallback_score_adjustment(
            similarity_scores,
            alignment_mode,
            len(student_questions),
            len(reference_questions),
        )
        
        # Grade all questions
        per_question_marks = total_marks / max(1, len(comparable_student_questions))
        grading_result = grader.grade_all(similarity_scores, per_question_marks)
        
        # Generate feedback
        feedback_list = []
        for qid, score in similarity_scores.items():
            missing_keywords = similarity_engine.extract_missing_keywords(
                comparable_student_questions.get(qid, ''),
                comparable_reference_questions.get(qid, '')
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
        
        charts = chart_generator.generate_all(grading_result, str(charts_dir))
        
        # Create submission ID and save to database
        submission_id = str(uuid.uuid4())
        
        if HAS_DATABASE:
            if database is None:
                add_persistence_warning(persistence_warnings, 'save_result', 'database_not_initialized')
            else:
                try:
                    saved = database.save_result(
                        submission_id,
                        answer_file_id,
                        reference_file_id,
                        grading_result,
                        feedback_list,
                        datetime.now(timezone.utc).isoformat()
                    )
                    if not saved:
                        add_persistence_warning(persistence_warnings, 'save_result', 'database_returned_false')
                except Exception as db_err:
                    logger.error(f"Database save failed: {str(db_err)}")
                    add_persistence_warning(persistence_warnings, 'save_result', 'database_exception')
        
        logger.info(f"Evaluation complete: {submission_id} (score: {grading_result['percentage']}%)")
        
        response_payload = {
            "status": "ok",
            "submission_id": submission_id,
            "answer_file_id": answer_file_id,
            "reference_file_id": reference_file_id,
            "total": grading_result['total'],
            "out_of": grading_result['out_of'],
            "percentage": grading_result['percentage'],
            "grade": grading_result['grade'],
            "question_alignment_mode": alignment_mode,
            "compared_question_count": len(similarity_scores),
            "questions": list(grading_result['questions'].keys()),
            "feedback": feedback_list,
            "charts": charts
        }

        return jsonify(attach_persistence_warnings(response_payload, persistence_warnings)), 200
    
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.route('/results/<submission_id>', methods=['GET'])
def get_results(submission_id: str):
    """
    Get evaluation results for a submission.
    
    Returns:
        JSON: Complete evaluation results with charts and feedback
    """
    try:
        if not HAS_DATABASE or database is None:
            return jsonify({"status": "error", "message": "Results service not available"}), 503
        
        result = database.get_result(submission_id)
        
        if not result:
            logger.warning(f"Results not found: {submission_id}")
            return jsonify({"status": "error", "message": "Results not found"}), 404
        
        logger.info(f"Retrieved results: {submission_id}")
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.route('/results/<submission_id>/feedback', methods=['GET'])
def get_feedback(submission_id: str):
    """
    Get detailed feedback for a submission.
    
    Returns:
        JSON: Detailed feedback with remarks, keywords, and weak areas
    """
    try:
        if not HAS_DATABASE or database is None:
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
        return jsonify({"status": "error", "message": "Internal server error"}), 500

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
        subject_raw = request.form.get('subject', 'general')
        subject = secure_filename(subject_raw) or 'general'

        total_marks, marks_error = parse_total_marks(
            request.form.get('total_marks'),
            CONFIG['grading']['total_marks']
        )
        if marks_error:
            return jsonify({"status": "error", "message": marks_error}), 400

        persistence_warnings = []
        
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

        if HAS_DATABASE:
            if database is None:
                add_persistence_warning(persistence_warnings, 'save_upload_answer_sheet', 'database_not_initialized')
            else:
                try:
                    saved = database.save_upload(
                        file_id=answer_file_id,
                        filename=secure_filename(answer_file.filename),
                        file_type='answer_sheet',
                        subject=None,
                        path=str(answer_path),
                        timestamp=datetime.now(timezone.utc).isoformat()
                    )
                    if not saved:
                        add_persistence_warning(persistence_warnings, 'save_upload_answer_sheet', 'database_returned_false')
                except Exception as db_err:
                    logger.error(f"Failed to save answer upload metadata: {str(db_err)}")
                    add_persistence_warning(persistence_warnings, 'save_upload_answer_sheet', 'database_exception')
        
        # Upload reference
        reference_file_id = str(uuid.uuid4())
        reference_filename = f"{reference_file_id}_{secure_filename(reference_file.filename)}"
        references_dir = UPLOAD_FOLDER / "references" / subject
        references_dir.mkdir(parents=True, exist_ok=True)
        reference_path = references_dir / reference_filename
        reference_file.save(str(reference_path))

        if HAS_DATABASE:
            if database is None:
                add_persistence_warning(persistence_warnings, 'save_upload_reference', 'database_not_initialized')
            else:
                try:
                    saved = database.save_upload(
                        file_id=reference_file_id,
                        filename=secure_filename(reference_file.filename),
                        file_type='reference',
                        subject=subject,
                        path=str(reference_path),
                        timestamp=datetime.now(timezone.utc).isoformat()
                    )
                    if not saved:
                        add_persistence_warning(persistence_warnings, 'save_upload_reference', 'database_returned_false')
                except Exception as db_err:
                    logger.error(f"Failed to save reference upload metadata: {str(db_err)}")
                    add_persistence_warning(persistence_warnings, 'save_upload_reference', 'database_exception')
        
        logger.info(f"Pipeline: Uploaded files {answer_file_id}, {reference_file_id}")
        
        # Extract text from both files
        answer_extraction = ocr_extractor.extract_from_pdf(str(answer_path))
        reference_extraction = ocr_extractor.extract_from_pdf(str(reference_path))
        
        extraction_output = Path(CONFIG['storage']['results_folder']) / 'extracted'
        extraction_output.mkdir(parents=True, exist_ok=True)
        ocr_extractor.save_extraction(answer_file_id, answer_extraction, str(extraction_output))
        ocr_extractor.save_extraction(reference_file_id, reference_extraction, str(extraction_output))

        if HAS_DATABASE:
            if database is None:
                add_persistence_warning(persistence_warnings, 'save_extraction_answer_sheet', 'database_not_initialized')
                add_persistence_warning(persistence_warnings, 'save_extraction_reference', 'database_not_initialized')
            else:
                try:
                    ts = datetime.now(timezone.utc).isoformat()
                    saved_answer_extraction = database.save_extraction(
                        file_id=answer_file_id,
                        pages=answer_extraction.get('pages', []),
                        full_text=answer_extraction.get('full_text', ''),
                        timestamp=ts
                    )
                    saved_reference_extraction = database.save_extraction(
                        file_id=reference_file_id,
                        pages=reference_extraction.get('pages', []),
                        full_text=reference_extraction.get('full_text', ''),
                        timestamp=ts
                    )
                    if not saved_answer_extraction:
                        add_persistence_warning(persistence_warnings, 'save_extraction_answer_sheet', 'database_returned_false')
                    if not saved_reference_extraction:
                        add_persistence_warning(persistence_warnings, 'save_extraction_reference', 'database_returned_false')
                except Exception as db_err:
                    logger.error(f"Failed to save extraction metadata: {str(db_err)}")
                    add_persistence_warning(persistence_warnings, 'save_extraction_answer_sheet', 'database_exception')
                    add_persistence_warning(persistence_warnings, 'save_extraction_reference', 'database_exception')
        
        logger.info("Pipeline: OCR complete")
        
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
        
        logger.info("Pipeline: Preprocessing complete")
        
        # Evaluate
        student_questions = answer_questions.get('questions', {})
        reference_questions_map = reference_questions.get('questions', {})
        comparable_student_questions, comparable_reference_questions, alignment_mode = select_comparable_question_pairs(
            student_questions,
            reference_questions_map,
            similarity_engine
        )

        if not comparable_student_questions:
            return jsonify({
                "status": "error",
                "message": "No comparable answer/reference question pairs found after preprocessing"
            }), 422

        similarity_scores = similarity_engine.compute_batch(
            comparable_student_questions,
            comparable_reference_questions
        )
        similarity_scores = apply_semantic_fallback_score_adjustment(
            similarity_scores,
            alignment_mode,
            len(student_questions),
            len(reference_questions_map),
        )
        
        per_question_marks = total_marks / max(1, len(comparable_student_questions))
        grading_result = grader.grade_all(similarity_scores, per_question_marks)
        
        # Generate feedback
        feedback_list = []
        for qid, score in similarity_scores.items():
            missing_keywords = similarity_engine.extract_missing_keywords(
                comparable_student_questions.get(qid, ''),
                comparable_reference_questions.get(qid, '')
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
            if database is None:
                add_persistence_warning(persistence_warnings, 'save_result', 'database_not_initialized')
            else:
                try:
                    saved = database.save_result(
                        submission_id,
                        answer_file_id,
                        reference_file_id,
                        grading_result,
                        feedback_list,
                        datetime.now(timezone.utc).isoformat()
                    )
                    if not saved:
                        add_persistence_warning(persistence_warnings, 'save_result', 'database_returned_false')
                except Exception as db_err:
                    logger.error(f"Database save failed: {str(db_err)}")
                    add_persistence_warning(persistence_warnings, 'save_result', 'database_exception')
        
        logger.info(f"Pipeline: Complete - {submission_id}")
        
        response_payload = {
            "status": "ok",
            "submission_id": submission_id,
            "total": grading_result['total'],
            "out_of": grading_result['out_of'],
            "percentage": grading_result['percentage'],
            "grade": grading_result['grade'],
            "question_alignment_mode": alignment_mode,
            "compared_question_count": len(similarity_scores),
            "feedback": feedback_list,
            "charts": charts
        }

        return jsonify(attach_persistence_warnings(response_payload, persistence_warnings)), 200
    
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

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
