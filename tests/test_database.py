"""Unit tests for database abstraction layer using mocked backends."""

from src.db.database import Database


class FakeMongoCollection:
    def __init__(self):
        self.docs = []

    def insert_one(self, document):
        self.docs.append(document)

    def find_one(self, query):
        key, value = next(iter(query.items()))
        for doc in self.docs:
            if doc.get(key) == value:
                return dict(doc)
        return None

    def find(self, query):
        key, value = next(iter(query.items()))
        return [dict(doc) for doc in self.docs if doc.get(key) == value]


class FakeMongoDB:
    def __init__(self):
        self.uploads = FakeMongoCollection()
        self.extractions = FakeMongoCollection()
        self.results = FakeMongoCollection()


def build_db_with_fake_mongo():
    db = Database({'type': 'unsupported', 'db_name': 'smart_eval'})
    db.db_type = 'mongodb'
    db.db = FakeMongoDB()
    return db


class TestDatabaseAbstraction:
    def test_save_upload_mongodb_success(self):
        db = build_db_with_fake_mongo()

        ok = db.save_upload(
            file_id='f1',
            filename='answer.pdf',
            file_type='answer_sheet',
            subject=None,
            path='uploads/answer_sheets/f1_answer.pdf',
            timestamp='2026-03-28T00:00:00+00:00',
        )

        assert ok is True
        assert len(db.db.uploads.docs) == 1
        assert db.db.uploads.docs[0]['file_id'] == 'f1'

    def test_save_extraction_mongodb_success(self):
        db = build_db_with_fake_mongo()

        ok = db.save_extraction(
            file_id='f2',
            pages=[{'page': 1, 'text': 'hello'}],
            full_text='hello',
            timestamp='2026-03-28T00:00:00+00:00',
        )

        assert ok is True
        assert len(db.db.extractions.docs) == 1
        assert db.db.extractions.docs[0]['page_count'] == 1

    def test_save_result_and_get_result_mongodb_success(self):
        db = build_db_with_fake_mongo()

        ok = db.save_result(
            submission_id='s1',
            answer_file_id='a1',
            reference_file_id='r1',
            scores={
                'total': 80,
                'out_of': 100,
                'percentage': 80.0,
                'grade': 'B+',
                'questions': {'Q1': {'marks_awarded': 8}},
            },
            feedback=[{'question': 'Q1', 'remark': 'Good'}],
            timestamp='2026-03-28T00:00:00+00:00',
        )

        assert ok is True

        fetched = db.get_result('s1')
        assert fetched is not None
        assert fetched['submission_id'] == 's1'
        assert fetched['grade'] == 'B+'

    def test_get_result_returns_none_when_missing(self):
        db = build_db_with_fake_mongo()
        fetched = db.get_result('missing')
        assert fetched is None

    def test_get_submissions_by_subject_filters(self):
        db = build_db_with_fake_mongo()
        db.db.results.insert_one({'submission_id': 's1', 'subject': 'math'})
        db.db.results.insert_one({'submission_id': 's2', 'subject': 'science'})

        math = db.get_submissions_by_subject('math')

        assert len(math) == 1
        assert math[0]['submission_id'] == 's1'

    def test_get_performance_trend_no_student_records(self):
        db = build_db_with_fake_mongo()
        trend = db.get_performance_trend('student-1')
        assert trend == []

    def test_returns_false_when_database_unavailable(self):
        db = Database({'type': 'unsupported'})
        db.db_type = 'mongodb'
        db.db = None

        ok = db.save_upload('f', 'x.pdf', 'answer_sheet', None, '/tmp/x.pdf', 'now')

        assert ok is False
