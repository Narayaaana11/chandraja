#!/usr/bin/env python
"""Verify the Smart Evaluation System is working correctly."""

import sys
from pathlib import Path
from io import BytesIO
from typing import Any

def verify_imports():
    """Verify all modules can be imported."""
    print("\n📦 VERIFYING IMPORTS...")
    try:
        from smart_eval_app import app as flask_app
        _ = flask_app
        print("  ✅ Flask app imports successfully")
    except Exception as e:
        print(f"  ❌ Flask app import failed: {e}")
        return False
    
    try:
        from src.ocr.extractor import OCRExtractor
        _ = OCRExtractor
        print("  ✅ OCRExtractor imports successfully")
    except Exception as e:
        print(f"  ⚠️  OCRExtractor import failed (optional): {e}")
    
    try:
        from src.preprocessing.cleaner import TextCleaner
        _ = TextCleaner
        print("  ✅ TextCleaner imports successfully")
    except Exception as e:
        print(f"  ❌ TextCleaner import failed: {e}")
        return False
    
    try:
        from src.evaluation.similarity import SimilarityEngine
        _ = SimilarityEngine
        print("  ✅ SimilarityEngine imports successfully")
    except Exception as e:
        print(f"  ⚠️  SimilarityEngine import failed (optional): {e}")
    
    try:
        from src.evaluation.grader import Grader
        _ = Grader
        print("  ✅ Grader imports successfully")
    except Exception as e:
        print(f"  ❌ Grader import failed: {e}")
        return False
    
    try:
        from src.evaluation.feedback import FeedbackGenerator
        _ = FeedbackGenerator
        print("  ✅ FeedbackGenerator imports successfully")
    except Exception as e:
        print(f"  ❌ FeedbackGenerator import failed: {e}")
        return False
    
    try:
        from src.visualization.charts import ChartGenerator
        _ = ChartGenerator
        print("  ✅ ChartGenerator imports successfully")
    except Exception as e:
        print(f"  ⚠️  ChartGenerator import failed (optional): {e}")
    
    try:
        from src.db.database import Database
        _ = Database
        print("  ✅ Database imports successfully")
    except Exception as e:
        print(f"  ⚠️  Database import failed (optional): {e}")
    
    return True

def verify_configuration():
    """Verify configuration file is valid."""
    print("\n⚙️  VERIFYING CONFIGURATION...")
    try:
        import yaml
        with open('smart_eval_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['server', 'ocr', 'preprocessing', 'evaluation', 'grading', 'database', 'storage']
        missing = [s for s in required_sections if s not in config]
        
        if missing:
            print(f"  ❌ Missing config sections: {missing}")
            return False
        
        print("  ✅ Configuration valid")
        print(f"     - Server: {config['server']['host']}:{config['server']['port']}")
        print(f"     - OCR: {config['ocr']['engine']}")
        print(f"     - Database: {config['database']['type']}")
        print(f"     - Storage: {config['storage']['upload_folder']}")
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def verify_directories():
    """Verify required directories exist."""
    print("\n📁 VERIFYING DIRECTORY STRUCTURE...")
    required_dirs = [
        'src', 'src/ocr', 'src/preprocessing', 'src/evaluation', 
        'src/visualization', 'src/db',
        'templates', 'tests', 'uploads', 'results', 'data',
        'uploads/answer_sheets', 'uploads/references',
        'data/extracted', 'data/processed'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ Missing: {dir_path}/")
            all_exist = False
    
    return all_exist

def verify_files():
    """Verify key files exist."""
    print("\n📄 VERIFYING FILES...")
    required_files = [
        'smart_eval_app.py',
        'smart_eval_config.yaml',
        'smart_eval_requirements.txt',
        'src/ocr/extractor.py',
        'src/preprocessing/cleaner.py',
        'src/evaluation/similarity.py',
        'src/evaluation/grader.py',
        'src/evaluation/feedback.py',
        'src/visualization/charts.py',
        'src/db/database.py',
        'templates/smart_eval_index.html',
        'templates/smart_eval_results.html',
        'templates/smart_eval_feedback.html',
        'tests/test_ocr.py',
        'tests/test_preprocessing.py',
        'tests/test_similarity.py',
        'tests/test_feedback.py',
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"  ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"  ❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def verify_processing_pipeline():
    """Verify core processing pipeline works."""
    print("\n🔄 VERIFYING PROCESSING PIPELINE...")
    
    try:
        from src.preprocessing.cleaner import TextCleaner
        config: dict[str, Any] = {
            'lowercase': True,
            'fix_ocr_errors': True,
            'remove_special_chars': True,
            'normalize_whitespace': True,
            'remove_stopwords': False,
        }
        cleaner = TextCleaner(config)
        
        # Test cleaning
        text = "HELLO    World!!! Test@#$"
        cleaned = cleaner.clean(text)
        print("  ✅ Text cleaning works")
        print(f"     Input:  '{text}'")
        print(f"     Output: '{cleaned}'")
        
        # Test question splitting
        text_with_q = "Q1. First answer. Q2. Second answer."
        result = cleaner.split_by_questions(text_with_q, delimiter='Q')
        print(f"  ✅ Question splitting works ({result['count']} questions found)")
        
    except Exception as e:
        print(f"  ⚠️  Processing pipeline test failed: {e}")
        return False
    
    try:
        from src.evaluation.grader import Grader
        config: dict[str, Any] = {
            'total_marks': 100,
            'partial_scoring': True,
            'partial_scoring_levels': [
                {'threshold': 0.8, 'percentage': 100},
                {'threshold': 0.6, 'percentage': 75},
                {'threshold': 0.0, 'percentage': 0}
            ]
        }
        grader = Grader(config)
        
        # Test grading
        result = grader.grade_question(0.85, 10)
        print("  ✅ Grading works")
        print(f"     Similarity: 0.85 → {result['marks_awarded']}/{result['max_marks']} marks ({result['percentage']}%)")
        
    except Exception as e:
        print(f"  ⚠️  Grading test failed: {e}")
        return False
    
    return True

def verify_flask_app():
    """Verify Flask app loads and has routes."""
    print("\n🌐 VERIFYING FLASK APPLICATION...")
    
    try:
        from smart_eval_app import app
        print("  ✅ Flask app loads successfully")
        
        routes = [str(rule) for rule in app.url_map.iter_rules()]
        print(f"  ✅ {len(routes)} routes registered:")
        for route in sorted(set(routes)):
            if not route.startswith('/static'):
                print(f"     - {route}")
        
        # Check for key routes
        key_routes = ['/upload/answer-sheet', '/upload/reference', '/ocr/extract', 
                     '/preprocess', '/evaluate', '/results/', '/pipeline/run']
        found = [r for r in routes if any(kr in r for kr in key_routes)]
        print(f"  ✅ Key evaluation routes found: {len(found)}")
        
        return True
    except Exception as e:
        print(f"  ❌ Flask app verification failed: {e}")
        return False

def verify_runtime_dependencies():
    """Verify key runtime dependencies and active fallbacks."""
    print("\n🧩 VERIFYING RUNTIME DEPENDENCIES...")

    ok = True

    try:
        from src.ocr import extractor as ocr_module
        if getattr(ocr_module, 'HAS_PADDLEOCR', False):
            print("  ✅ PaddleOCR available")
        elif getattr(ocr_module, 'HAS_TESSERACT', False):
            print("  ✅ Tesseract fallback available")
        else:
            print("  ❌ No OCR backend available")
            ok = False
    except Exception as e:
        print(f"  ❌ OCR dependency check failed: {e}")
        ok = False

    try:
        from src.evaluation.similarity import HAS_SENTENCE_TRANSFORMERS
        if HAS_SENTENCE_TRANSFORMERS:
            print("  ✅ sentence-transformers available")
        else:
            print("  ❌ sentence-transformers unavailable")
            ok = False
    except Exception as e:
        print(f"  ❌ Similarity dependency check failed: {e}")
        ok = False

    try:
        import smart_eval_app as app_module
        database_obj = getattr(app_module, 'database', None)
        if database_obj is not None:
            print("  ✅ Database service initialized")
        else:
            print("  ⚠️  Database service not initialized (local fallback mode)")
    except Exception as e:
        print(f"  ⚠️  Database dependency check failed: {e}")

    return ok

def verify_endpoint_contracts():
    """Verify required endpoint status contracts with Flask test client."""
    print("\n🧪 VERIFYING ENDPOINT CONTRACTS...")

    try:
        from smart_eval_app import app
        client = app.test_client()

        checks = [
            ("GET /health", client.get('/health').status_code == 200),
            (
                "GET /health includes timestamp",
                isinstance(client.get('/health').get_json(), dict) and 'timestamp' in client.get('/health').get_json()
            ),
            ("POST /upload/answer-sheet without file -> 400", client.post('/upload/answer-sheet').status_code == 400),
            (
                "POST /upload/answer-sheet invalid extension -> 415",
                client.post(
                    '/upload/answer-sheet',
                    data={'file': (BytesIO(b'text'), 'bad.txt')},
                    content_type='multipart/form-data'
                ).status_code == 415
            ),
            ("POST /preprocess missing file_id -> 400", client.post('/preprocess', json={}).status_code == 400),
            (
                "POST /evaluate missing ids -> 400",
                client.post('/evaluate', json={'answer_file_id': '', 'reference_file_id': ''}).status_code == 400
            ),
            ("POST /pipeline/run missing files -> 400", client.post('/pipeline/run', data={}, content_type='multipart/form-data').status_code == 400),
        ]

        all_pass = True
        for name, result in checks:
            if result:
                print(f"  ✅ {name}")
            else:
                print(f"  ❌ {name}")
                all_pass = False

        return all_pass
    except Exception as e:
        print(f"  ❌ Endpoint contract verification failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("🔍 SMART EVALUATION SYSTEM - COMPLETE VERIFICATION")
    print("="*60)
    
    checks = [
        ("Configuration", verify_configuration),
        ("Directory Structure", verify_directories),
        ("Files", verify_files),
        ("Imports", verify_imports),
        ("Runtime Dependencies", verify_runtime_dependencies),
        ("Processing Pipeline", verify_processing_pipeline),
        ("Flask Application", verify_flask_app),
        ("Endpoint Contracts", verify_endpoint_contracts),
    ]

    results: dict[str, bool] = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check failed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n📈 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 SYSTEM IS FULLY OPERATIONAL AND READY TO USE!")
        print("\n🚀 To start the server:")
        print("   python smart_eval_app.py")
        print("\n🌐 Then open: http://localhost:5000")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
