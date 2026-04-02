"""Integration tests for API route contracts."""

import io
import json
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    """Load app module and isolate storage paths for route tests."""
    module = importlib.import_module('smart_eval_app')

    uploads = tmp_path / 'uploads'
    results = tmp_path / 'results'
    uploads.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(module.CONFIG['storage'], 'upload_folder', str(uploads))
    monkeypatch.setitem(module.CONFIG['storage'], 'results_folder', str(results))
    monkeypatch.setattr(module, 'UPLOAD_FOLDER', uploads)
    monkeypatch.setattr(module, 'RESULTS_FOLDER', results)

    module.app.config['TESTING'] = True
    module.app.config['UPLOAD_FOLDER'] = str(uploads)

    return module


@pytest.fixture
def client(app_module):
    """Return Flask test client."""
    return app_module.app.test_client()


class DummySimilarity:
    def compute_batch(self, student_questions, reference_questions):
        return {qid: 0.85 for qid in student_questions if qid in reference_questions}

    def extract_missing_keywords(self, student_text, reference_text):
        return ['keyword'] if reference_text and 'keyword' in reference_text.lower() and 'keyword' not in student_text.lower() else []


class DummyGrader:
    def grade_all(self, similarity_scores, per_question_marks):
        questions = {}
        total = 0.0
        for qid, score in similarity_scores.items():
            marks = round(per_question_marks * score, 2)
            questions[qid] = {
                'marks_awarded': marks,
                'max_marks': per_question_marks,
                'percentage': round(score * 100, 2),
                'similarity': score,
            }
            total += marks

        out_of = round(per_question_marks * len(questions), 2)
        percentage = round((total / out_of * 100), 2) if out_of else 0

        return {
            'questions': questions,
            'total': round(total, 2),
            'out_of': out_of,
            'percentage': percentage,
            'grade': 'A' if percentage >= 90 else 'B',
        }


class DummyFeedback:
    def generate(self, qid, score, missing_keywords, marks_info):
        return {
            'question': qid,
            'similarity_percentage': int(score * 100),
            'marks_awarded': marks_info.get('marks_awarded', 0),
            'max_marks': marks_info.get('max_marks', 0),
            'percentage': marks_info.get('percentage', 0),
            'remark': 'Good answer. Covers most key points.' if score >= 0.8 else 'Partial answer.',
            'missing_keywords': missing_keywords,
            'weak_areas': [] if score >= 0.8 else ['Missing details'],
        }


class DummyCharts:
    def generate_all(self, grading_result, charts_dir):
        Path(charts_dir).mkdir(parents=True, exist_ok=True)
        return {
            'bar_chart': str(Path(charts_dir) / 'bar_chart.png'),
            'pie_chart': str(Path(charts_dir) / 'pie_chart.png'),
            'line_chart': str(Path(charts_dir) / 'line_chart.png'),
        }


class DummyOCR:
    def extract_from_pdf(self, path):
        return {
            'pages': [{'page': 1, 'text': 'Q1. Answer one'}],
            'full_text': 'Q1. Answer one',
            'page_count': 1,
        }

    def save_extraction(self, file_id, result, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / f'{file_id}.json', 'w', encoding='utf-8') as f:
            json.dump(result, f)
        return str(Path(output_dir) / f'{file_id}.json')


class DummyCleaner:
    def clean(self, text):
        return text.lower()

    def split_by_questions(self, text, delimiter='Q'):
        return {'questions': {'Q1': 'answer one', 'Q2': 'answer two'}, 'count': 2}

    def save_processed(self, file_id, result, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / f'{file_id}.json', 'w', encoding='utf-8') as f:
            json.dump(result, f)
        return str(Path(output_dir) / f'{file_id}.json')


class DummyDatabase:
    def __init__(self, result=None):
        self._result = result

    def get_result(self, submission_id):
        return self._result


class RecordingDatabase:
    def __init__(self):
        self.uploads = []
        self.extractions = []
        self.results = []

    def save_upload(self, file_id, filename, file_type, subject, path, timestamp):
        self.uploads.append({
            'file_id': file_id,
            'filename': filename,
            'file_type': file_type,
            'subject': subject,
            'path': path,
            'timestamp': timestamp,
        })
        return True

    def save_extraction(self, file_id, pages, full_text, timestamp):
        self.extractions.append({
            'file_id': file_id,
            'pages': pages,
            'full_text': full_text,
            'timestamp': timestamp,
        })
        return True

    def save_result(self, submission_id, answer_file_id, reference_file_id, scores, feedback, timestamp):
        self.results.append({
            'submission_id': submission_id,
            'answer_file_id': answer_file_id,
            'reference_file_id': reference_file_id,
            'scores': scores,
            'feedback': feedback,
            'timestamp': timestamp,
        })
        return True


class TestRouteContracts:
    def test_health_has_timestamp(self, client):
        response = client.get('/health')
        assert response.status_code == 200
        payload = response.get_json()
        assert payload['status'] == 'ok'
        assert 'timestamp' in payload

    def test_upload_answer_sheet_missing_file_returns_400(self, client):
        response = client.post('/upload/answer-sheet')
        assert response.status_code == 400

    def test_upload_answer_sheet_invalid_extension_returns_415(self, client):
        response = client.post(
            '/upload/answer-sheet',
            data={'file': (io.BytesIO(b'test'), 'note.txt')},
            content_type='multipart/form-data',
        )
        assert response.status_code == 415

    def test_upload_answer_sheet_success_returns_file_id(self, client):
        response = client.post(
            '/upload/answer-sheet',
            data={'file': (io.BytesIO(b'%PDF-1.4\n%mock'), 'answer.pdf')},
            content_type='multipart/form-data',
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload['status'] == 'ok'
        assert payload['file_id']

    def test_upload_reference_success_returns_metadata(self, client):
        response = client.post(
            '/upload/reference',
            data={
                'file': (io.BytesIO(b'%PDF-1.4\n%mock'), 'reference.pdf'),
                'subject': 'physics',
                'question_paper_id': 'qp-101',
            },
            content_type='multipart/form-data',
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload['status'] == 'ok'
        assert payload['metadata']['subject'] == 'physics'
        assert payload['metadata']['question_paper_id'] == 'qp-101'

    def test_get_upload_info_success_for_answer_sheet(self, client):
        upload = client.post(
            '/upload/answer-sheet',
            data={'file': (io.BytesIO(b'%PDF-1.4\n%mock'), 'answer.pdf')},
            content_type='multipart/form-data',
        )
        file_id = upload.get_json()['file_id']

        response = client.get(f'/uploads/{file_id}')
        assert response.status_code == 200
        payload = response.get_json()
        assert payload['type'] == 'answer_sheet'

    def test_get_upload_info_success_for_reference(self, client):
        upload = client.post(
            '/upload/reference',
            data={
                'file': (io.BytesIO(b'%PDF-1.4\n%mock'), 'reference.pdf'),
                'subject': 'physics',
            },
            content_type='multipart/form-data',
        )
        file_id = upload.get_json()['file_id']

        response = client.get(f'/uploads/{file_id}')
        assert response.status_code == 200
        payload = response.get_json()
        assert payload['type'] == 'reference'
        assert payload['subject'] == 'physics'

    def test_get_upload_info_returns_404_for_unknown_id(self, client):
        response = client.get('/uploads/not-a-real-id')
        assert response.status_code == 404

    def test_preprocess_missing_file_id_returns_400(self, client):
        response = client.post('/preprocess', json={})
        assert response.status_code == 400

    def test_preprocess_returns_503_when_service_unavailable(self, client, app_module, monkeypatch):
        monkeypatch.setattr(app_module, 'HAS_CLEANER', False)
        monkeypatch.setattr(app_module, 'text_cleaner', None)
        response = client.post('/preprocess', json={'file_id': 'x'})
        assert response.status_code == 503

    def test_ocr_extract_missing_file_id_returns_400(self, client):
        response = client.post('/ocr/extract', json={})
        assert response.status_code == 400

    def test_ocr_extract_returns_503_when_service_unavailable(self, client, app_module, monkeypatch):
        monkeypatch.setattr(app_module, 'HAS_OCR', False)
        monkeypatch.setattr(app_module, 'ocr_extractor', None)
        response = client.post('/ocr/extract', json={'file_id': 'x'})
        assert response.status_code == 503

    def test_ocr_extract_returns_404_for_unknown_file(self, client):
        response = client.post('/ocr/extract', json={'file_id': 'missing-id', 'file_type': 'answer_sheet'})
        assert response.status_code == 404

    def test_ocr_extract_reference_success(self, client, app_module, monkeypatch):
        file_id = 'ref-ocr-1'
        subject_dir = Path(app_module.CONFIG['storage']['upload_folder']) / 'references' / 'chemistry'
        subject_dir.mkdir(parents=True, exist_ok=True)
        file_path = subject_dir / f'{file_id}_reference.pdf'
        file_path.write_bytes(b'%PDF-1.4\n%mock')

        monkeypatch.setattr(app_module, 'HAS_OCR', True)
        monkeypatch.setattr(app_module, 'ocr_extractor', DummyOCR())

        response = client.post('/ocr/extract', json={'file_id': file_id, 'file_type': 'reference'})

        assert response.status_code == 200
        payload = response.get_json()
        assert payload['status'] == 'ok'
        assert payload['file_id'] == file_id
        assert payload['page_count'] == 1

    def test_preprocess_returns_404_for_missing_extraction(self, client):
        response = client.post('/preprocess', json={'file_id': 'missing'})
        assert response.status_code == 404

    def test_preprocess_success_writes_processed_output(self, client, app_module):
        file_id = 'abc123'
        extracted_dir = Path(app_module.CONFIG['storage']['results_folder']) / 'extracted'
        extracted_dir.mkdir(parents=True, exist_ok=True)

        with open(extracted_dir / f'{file_id}.json', 'w', encoding='utf-8') as f:
            json.dump({'full_text': 'Q1. First answer Q2. Second answer'}, f)

        response = client.post('/preprocess', json={'file_id': file_id})

        assert response.status_code == 200
        payload = response.get_json()
        assert payload['status'] == 'ok'
        assert payload['question_count'] >= 1

        processed_file = Path(app_module.CONFIG['storage']['results_folder']) / 'processed' / f'{file_id}.json'
        assert processed_file.exists()

    def test_evaluate_missing_ids_returns_400(self, client):
        response = client.post('/evaluate', json={})
        assert response.status_code == 400

    def test_evaluate_returns_503_when_service_unavailable(self, client, app_module, monkeypatch):
        monkeypatch.setattr(app_module, 'HAS_SIMILARITY', False)
        monkeypatch.setattr(app_module, 'similarity_engine', None)
        response = client.post('/evaluate', json={'answer_file_id': 'a', 'reference_file_id': 'b'})
        assert response.status_code == 503

    def test_evaluate_returns_404_when_processed_files_missing(self, client):
        response = client.post('/evaluate', json={'answer_file_id': 'a', 'reference_file_id': 'b'})
        assert response.status_code == 404

    def test_evaluate_success_returns_required_fields(self, client, app_module, monkeypatch):
        answer_file_id = 'ans123'
        reference_file_id = 'ref123'
        processed_dir = Path(app_module.CONFIG['storage']['results_folder']) / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)

        answer_data = {'questions': {'Q1': 'keyword is present', 'Q2': 'another answer'}, 'count': 2}
        reference_data = {'questions': {'Q1': 'keyword and concept', 'Q2': 'another reference'}, 'count': 2}

        with open(processed_dir / f'{answer_file_id}.json', 'w', encoding='utf-8') as f:
            json.dump(answer_data, f)

        with open(processed_dir / f'{reference_file_id}.json', 'w', encoding='utf-8') as f:
            json.dump(reference_data, f)

        monkeypatch.setattr(app_module, 'HAS_SIMILARITY', True)
        monkeypatch.setattr(app_module, 'HAS_GRADER', True)
        monkeypatch.setattr(app_module, 'HAS_FEEDBACK', True)
        monkeypatch.setattr(app_module, 'HAS_CHARTS', True)
        monkeypatch.setattr(app_module, 'HAS_DATABASE', False)

        monkeypatch.setattr(app_module, 'similarity_engine', DummySimilarity())
        monkeypatch.setattr(app_module, 'grader', DummyGrader())
        monkeypatch.setattr(app_module, 'feedback_generator', DummyFeedback())
        monkeypatch.setattr(app_module, 'chart_generator', DummyCharts())

        response = client.post(
            '/evaluate',
            json={
                'answer_file_id': answer_file_id,
                'reference_file_id': reference_file_id,
                'subject': 'science',
                'total_marks': 100,
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload['status'] == 'ok'
        assert 'submission_id' in payload
        assert 'percentage' in payload
        assert 'feedback' in payload
        assert 'charts' in payload

    def test_pipeline_run_missing_files_returns_400(self, client):
        response = client.post('/pipeline/run', data={}, content_type='multipart/form-data')
        assert response.status_code == 400

    def test_pipeline_run_success_end_to_end(self, client, app_module, monkeypatch):
        db = RecordingDatabase()
        monkeypatch.setattr(app_module, 'HAS_OCR', True)
        monkeypatch.setattr(app_module, 'HAS_CLEANER', True)
        monkeypatch.setattr(app_module, 'HAS_SIMILARITY', True)
        monkeypatch.setattr(app_module, 'HAS_GRADER', True)
        monkeypatch.setattr(app_module, 'HAS_FEEDBACK', True)
        monkeypatch.setattr(app_module, 'HAS_CHARTS', True)
        monkeypatch.setattr(app_module, 'HAS_DATABASE', True)

        monkeypatch.setattr(app_module, 'ocr_extractor', DummyOCR())
        monkeypatch.setattr(app_module, 'text_cleaner', DummyCleaner())
        monkeypatch.setattr(app_module, 'similarity_engine', DummySimilarity())
        monkeypatch.setattr(app_module, 'grader', DummyGrader())
        monkeypatch.setattr(app_module, 'feedback_generator', DummyFeedback())
        monkeypatch.setattr(app_module, 'chart_generator', DummyCharts())
        monkeypatch.setattr(app_module, 'database', db)

        response = client.post(
            '/pipeline/run',
            data={
                'answer_sheet': (io.BytesIO(b'%PDF-1.4\n%mock answer'), 'answer.pdf'),
                'reference': (io.BytesIO(b'%PDF-1.4\n%mock ref'), 'reference.pdf'),
                'subject': 'biology',
                'total_marks': '100',
            },
            content_type='multipart/form-data',
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload['status'] == 'ok'
        assert payload['submission_id']
        assert len(db.uploads) == 2
        assert len(db.extractions) == 2
        assert len(db.results) == 1

    def test_results_returns_503_without_database(self, client, app_module, monkeypatch):
        monkeypatch.setattr(app_module, 'HAS_DATABASE', False)
        response = client.get('/results/any-submission-id')
        assert response.status_code == 503

    def test_feedback_returns_503_without_database(self, client, app_module, monkeypatch):
        monkeypatch.setattr(app_module, 'HAS_DATABASE', False)
        response = client.get('/results/any-submission-id/feedback')
        assert response.status_code == 503

    def test_results_returns_404_when_submission_missing(self, client, app_module, monkeypatch):
        monkeypatch.setattr(app_module, 'HAS_DATABASE', True)
        monkeypatch.setattr(app_module, 'database', DummyDatabase(result=None))

        response = client.get('/results/nope')
        assert response.status_code == 404

    def test_results_success_with_database(self, client, app_module, monkeypatch):
        stored = {
            'submission_id': 'sub1',
            'total': 75,
            'out_of': 100,
            'percentage': 75.0,
            'grade': 'B',
            'feedback': [{'question': 'Q1'}],
        }
        monkeypatch.setattr(app_module, 'HAS_DATABASE', True)
        monkeypatch.setattr(app_module, 'database', DummyDatabase(result=stored))

        response = client.get('/results/sub1')
        assert response.status_code == 200
        payload = response.get_json()
        assert payload['submission_id'] == 'sub1'
        assert payload['percentage'] == 75.0

    def test_feedback_success_with_database(self, client, app_module, monkeypatch):
        stored = {
            'submission_id': 'sub2',
            'total': 88,
            'out_of': 100,
            'percentage': 88.0,
            'grade': 'B+',
            'feedback': [{'question': 'Q1', 'remark': 'Good'}],
        }
        monkeypatch.setattr(app_module, 'HAS_DATABASE', True)
        monkeypatch.setattr(app_module, 'database', DummyDatabase(result=stored))

        response = client.get('/results/sub2/feedback')
        assert response.status_code == 200
        payload = response.get_json()
        assert payload['submission_id'] == 'sub2'
        assert len(payload['feedback']) == 1
