"""Database operations for MongoDB and MySQL."""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# MongoDB imports
try:
    from pymongo import MongoClient
    HAS_MONGODB = True
except ImportError:
    HAS_MONGODB = False

# MySQL imports
try:
    import pymysql
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False


class Database:
    """Database abstraction layer supporting MongoDB and MySQL."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database connection.
        
        Args:
            config: Database configuration with:
                - type: "mongodb" or "mysql"
                - mongodb_uri: MongoDB connection string
                - db_name: Database name
                - For MySQL: host, user, password, port
        """
        self.config = config
        self.db_type = config.get('type', 'mongodb')
        self.db_name = config.get('db_name', 'smart_eval')
        self.db = None
        
        if self.db_type == 'mongodb':
            self._init_mongodb()
        elif self.db_type == 'mysql':
            self._init_mysql()
        else:
            logger.warning(f"Unknown database type: {self.db_type}")
    
    def _init_mongodb(self) -> None:
        """Initialize MongoDB connection."""
        if not HAS_MONGODB:
            logger.error("pymongo not installed")
            return
        
        try:
            uri = self.config.get('mongodb_uri', 'mongodb://localhost:27017/')
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Test connection
            client.admin.command('ismaster')
            self.db = client[self.db_name]
            logger.info(f"MongoDB connected: {self.db_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.db = None
    
    def _init_mysql(self) -> None:
        """Initialize MySQL connection."""
        if not HAS_MYSQL:
            logger.error("pymysql not installed")
            return
        
        try:
            conn = pymysql.connect(
                host=self.config.get('host', 'localhost'),
                user=self.config.get('user', 'root'),
                password=self.config.get('password', ''),
                port=self.config.get('port', 3306),
                database=self.db_name
            )
            self.db = conn
            logger.info(f"MySQL connected: {self.db_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {e}")
            self.db = None
    
    def save_upload(self, file_id: str, filename: str, file_type: str,
                   subject: Optional[str], path: str, timestamp: str) -> bool:
        """
        Save file upload metadata.
        
        Args:
            file_id: Unique file identifier
            filename: Original filename
            file_type: "answer_sheet" or "reference"
            subject: Subject (optional,for references)
            path: File path
            timestamp: Upload timestamp
        
        Returns:
            True if successful, False otherwise
        """
        try:
            document = {
                "file_id": file_id,
                "filename": filename,
                "file_type": file_type,
                "subject": subject,
                "path": path,
                "timestamp": timestamp
            }
            
            if self.db_type == 'mongodb' and self.db:
                self.db.uploads.insert_one(document)
                logger.info(f"Upload saved to MongoDB: {file_id}")
                return True
            elif self.db_type == 'mysql' and self.db:
                cursor = self.db.cursor()
                sql = """
                    INSERT INTO uploads (file_id, filename, file_type, subject, path, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (file_id, filename, file_type, subject, path, timestamp))
                self.db.commit()
                logger.info(f"Upload saved to MySQL: {file_id}")
                return True
            else:
                logger.warning("Database not available")
                return False
        
        except Exception as e:
            logger.error(f"Error saving upload: {e}")
            return False
    
    def save_extraction(self, file_id: str, pages: List[Dict],
                       full_text: str, timestamp: str) -> bool:
        """
        Save OCR extraction result.
        
        Args:
            file_id: File identifier
            pages: List of page dictionaries
            full_text: Full extracted text
            timestamp: Extraction timestamp
        
        Returns:
            True if successful
        """
        try:
            document = {
                "file_id": file_id,
                "pages": pages,
                "full_text": full_text,
                "page_count": len(pages),
                "timestamp": timestamp
            }
            
            if self.db_type == 'mongodb' and self.db:
                self.db.extractions.insert_one(document)
                logger.info(f"Extraction saved: {file_id}")
                return True
            elif self.db_type == 'mysql' and self.db:
                cursor = self.db.cursor()
                sql = """
                    INSERT INTO extractions (file_id, page_count, full_text, timestamp)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql, (file_id, len(pages), full_text, timestamp))
                self.db.commit()
                return True
            else:
                return False
        
        except Exception as e:
            logger.error(f"Error saving extraction: {e}")
            return False
    
    def save_result(self, submission_id: str, answer_file_id: str,
                   reference_file_id: str, scores: Dict[str, Any],
                   feedback: List[Dict], timestamp: str) -> bool:
        """
        Save evaluation result.
        
        Args:
            submission_id: Unique submission identifier
            answer_file_id: Answer sheet file ID
            reference_file_id: Reference file ID
            scores: Grading results
            feedback: Question-wise feedback
            timestamp: Evaluation timestamp
        
        Returns:
            True if successful
        """
        try:
            document = {
                "submission_id": submission_id,
                "answer_file_id": answer_file_id,
                "reference_file_id": reference_file_id,
                "total_score": scores.get("total"),
                "out_of": scores.get("out_of"),
                "percentage": scores.get("percentage"),
                "grade": scores.get("grade"),
                "questions": scores.get("questions"),
                "feedback": feedback,
                "timestamp": timestamp
            }
            
            if self.db_type == 'mongodb' and self.db:
                self.db.results.insert_one(document)
                logger.info(f"Result saved: {submission_id}")
                return True
            elif self.db_type == 'mysql' and self.db:
                cursor = self.db.cursor()
                sql = """
                    INSERT INTO results
                    (submission_id, answer_file_id, reference_file_id, total_score, out_of, percentage, grade, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (submission_id, answer_file_id, reference_file_id,
                                    scores.get("total"), scores.get("out_of"),
                                    scores.get("percentage"), scores.get("grade"), timestamp))
                self.db.commit()
                return True
            else:
                return False
        
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            return False
    
    def get_result(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve evaluation result.
        
        Args:
            submission_id: Submission identifier
        
        Returns:
            Result dictionary or None if not found
        """
        try:
            if self.db_type == 'mongodb' and self.db:
                result = self.db.results.find_one({"submission_id": submission_id})
                if result:
                    result.pop('_id', None)  # Remove MongoDB ID
                return result
            elif self.db_type == 'mysql' and self.db:
                cursor = self.db.cursor()
                cursor.execute("SELECT * FROM results WHERE submission_id = %s", (submission_id,))
                # This is simplified - actual implementation would map to dict
                return cursor.fetchone()
            return None
        
        except Exception as e:
            logger.error(f"Error retrieving result: {e}")
            return None
    
    def get_submissions_by_subject(self, subject: str) -> List[Dict[str, Any]]:
        """
        Get all submissions for a subject.
        
        Args:
            subject: Subject name
        
        Returns:
            List of submission dictionaries
        """
        try:
            if self.db_type == 'mongodb' and self.db:
                results = list(self.db.results.find({"subject": subject}))
                for r in results:
                    r.pop('_id', None)
                return results
            else:
                logger.warning("get_submissions_by_subject not fully implemented for MySQL")
                return []
        
        except Exception as e:
            logger.error(f"Error retrieving submissions: {e}")
            return []
    
    def get_performance_trend(self, student_id: str) -> List[Dict[str, Any]]:
        """
        Get performance trend for a student.
        
        Args:
            student_id: Student identifier
        
        Returns:
            List of results sorted by timestamp
        """
        try:
            if self.db_type == 'mongodb' and self.db:
                results = list(self.db.results.find(
                    {"student_id": student_id}
                ).sort("timestamp", 1))
                for r in results:
                    r.pop('_id', None)
                return results
            else:
                logger.warning("get_performance_trend not fully implemented for MySQL")
                return []
        
        except Exception as e:
            logger.error(f"Error retrieving performance trend: {e}")
            return []
