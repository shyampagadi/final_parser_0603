import os
import logging
import psycopg2
import uuid
import traceback
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import Json
import time

from config.config import (
    ENABLE_POSTGRES,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    get_postgres_connection_string
)

logger = logging.getLogger(__name__)

# Global connection pool
_connection_pool = None

def get_connection_pool(min_conn=1, max_conn=10):
    """
    Get or create the global connection pool
    
    Args:
        min_conn: Minimum number of connections
        max_conn: Maximum number of connections
        
    Returns:
        ThreadedConnectionPool instance
    """
    global _connection_pool
    
    if _connection_pool is None:
        try:
            _connection_pool = ThreadedConnectionPool(
                min_conn,
                max_conn,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD
            )
            logger.info(f"Created PostgreSQL connection pool with {min_conn}-{max_conn} connections")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {str(e)}")
            raise
    
    return _connection_pool

class PostgresHandler:
    """Handle PostgreSQL database operations"""
    
    def __init__(self):
        """Initialize PostgreSQL handler"""
        self.host = POSTGRES_HOST
        self.port = POSTGRES_PORT
        self.database = POSTGRES_DB
        self.user = POSTGRES_USER
        self.password = POSTGRES_PASSWORD
        self.conn = None
        self.pool = None
        self.max_retries = 3  # Add retry count
        self.retry_delay = 1  # Seconds between retries
        self.initialize_pool()
        logger.info(f"PostgreSQL handler initialized for database: {self.database} on {self.host}")
        
    def initialize_pool(self):
        """Initialize connection pool"""
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                1,  # Min connections
                10,  # Max connections
                host=self.host,
                port=self.port,
                dbname=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=10,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            logger.info(f"Created PostgreSQL connection pool with 1-10 connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            self.pool = None
            
    def connect(self):
        """Connect to PostgreSQL database with retry logic"""
        for attempt in range(self.max_retries):
            try:
                if self.pool:
                    self.conn = self.pool.getconn()
                else:
                    self.conn = psycopg2.connect(
                        host=self.host,
                        port=self.port,
                        dbname=self.database,
                        user=self.user,
                        password=self.password,
                        connect_timeout=10
                    )
                    
                # Set connection parameters for stability
                if self.conn and not self.conn.closed:
                    self.conn.set_session(autocommit=True)
                    
                logger.info("Connected to PostgreSQL database successfully")
                return True
            except Exception as e:
                logger.error(f"Connection attempt {attempt+1}/{self.max_retries} failed: {str(e)}")
                if self.conn and not self.conn.closed:
                    self.conn.close()
                time.sleep(self.retry_delay)
                
        logger.error(f"Failed to connect to PostgreSQL database after {self.max_retries} attempts")
        return False

    # Add a method to check connection and reconnect if needed
    def ensure_connection(self):
        """Ensure connection is active, reconnect if needed"""
        try:
            if self.conn is None or self.conn.closed:
                logger.info("PostgreSQL connection closed or None, reconnecting...")
                return self.connect()
                
            # Test if connection is still alive with a simple query
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception as e:
            logger.warning(f"Connection test failed: {str(e)}, reconnecting...")
            if self.conn:
                try:
                    self.conn.close()
                except:
                    pass
            return self.connect()
    
    def close(self):
        """Close connection to PostgreSQL database"""
        try:
            if self.conn:
                if self.pool:
                    self.pool.putconn(self.conn)
                else:
                    self.conn.close()
                logger.info("Closed PostgreSQL connection")
        except Exception as e:
            logger.error(f"Error closing PostgreSQL connection: {str(e)}")
            
    def __enter__(self):
        """Context manager enter"""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def ensure_tables_exist(self):
        """Ensure required tables exist in database"""
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            
            with self.conn.cursor() as cursor:
                # First check if the table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'resume_pii'
                    );
                """)
                table_exists = cursor.fetchone()[0]
                
                if not table_exists:
                    # Create resume_pii table if it doesn't exist
                    cursor.execute("""
                        CREATE TABLE resume_pii (
                            resume_id UUID PRIMARY KEY,
                            name TEXT,
                            email TEXT,
                            phone_number TEXT,
                            address TEXT,
                            linkedin_url TEXT,
                            s3_bucket TEXT,
                            s3_key TEXT,
                            original_filename TEXT,
                            file_type TEXT,
                            created_dt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_dt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    logger.info("Created resume_pii table")
                else:
                    # Check if the columns exist
                    cursor.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'resume_pii';
                    """)
                    columns = [row[0] for row in cursor.fetchall()]
                    
                    # Check if we need to add the missing columns
                    required_columns = ['name', 'email', 'phone_number', 'address', 'linkedin_url', 'created_dt', 'updated_dt']
                    for column in required_columns:
                        if column not in columns:
                            logger.info(f"Adding missing column {column} to resume_pii table")
                            if column in ['created_dt', 'updated_dt']:
                                cursor.execute(f"ALTER TABLE resume_pii ADD COLUMN {column} TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
                            else:
                                cursor.execute(f"ALTER TABLE resume_pii ADD COLUMN {column} TEXT;")
                
                logger.info("Ensured resume_pii table exists with required columns")
                
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"Error ensuring tables exist: {str(e)}")
            raise
    
    def insert_resume_pii(
        self,
        resume_data: Dict[str, Any],
        s3_bucket: str,
        s3_key: str,
        original_filename: str,
        file_type: str,
        resume_id: Optional[str] = None
    ) -> str:
        """
        Insert or update resume PII data
        
        Args:
            resume_data: Resume data dictionary
            s3_bucket: S3 bucket name
            s3_key: S3 object key
            original_filename: Original filename
            file_type: File type (pdf, docx, etc.)
            resume_id: Optional resume ID (UUID)
            
        Returns:
            Resume ID (UUID)
        """
        conn = self.connect()
        
        # Generate UUID if not provided
        if resume_id is None:
            resume_id = str(uuid.uuid4())
        
        try:
            with conn.cursor() as cursor:
                # Insert or update resume PII data
                cursor.execute("""
                    INSERT INTO resume_pii (
                        resume_id, name, email, phone_number, address, linkedin_url,
                        s3_bucket, s3_key, original_filename, file_type, created_dt, updated_dt
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (resume_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        email = EXCLUDED.email,
                        phone_number = EXCLUDED.phone_number,
                        address = EXCLUDED.address,
                        linkedin_url = EXCLUDED.linkedin_url,
                        s3_bucket = EXCLUDED.s3_bucket,
                        s3_key = EXCLUDED.s3_key,
                        original_filename = EXCLUDED.original_filename,
                        file_type = EXCLUDED.file_type,
                        updated_dt = CURRENT_TIMESTAMP
                    RETURNING resume_id
                """, (
                    resume_id,
                    resume_data.get('name', ''),
                    resume_data.get('email', ''),
                    resume_data.get('phone_number', ''),
                    resume_data.get('address', ''),
                    resume_data.get('linkedin_url', ''),
                    s3_bucket,
                    s3_key,
                    original_filename,
                    file_type
                ))
                
                result = cursor.fetchone()
                conn.commit()
                
                logger.info(f"Inserted/updated resume PII data with ID: {resume_id}")
                return str(result[0]) if result else resume_id
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting resume PII data: {str(e)}")
            raise
    
    def batch_insert_resume_pii(self, batch_data):
        """
        Batch insert or update resume PII data
        
        Args:
            batch_data: List of dictionaries with resume data
            
        Returns:
            List of resume IDs
        """
        if not batch_data:
            return []
            
        # Ensure connection is active
        if not self.ensure_connection():
            logger.error("Failed to establish PostgreSQL connection for batch insert")
            raise Exception("Failed to establish PostgreSQL connection")
            
        resume_ids = []
        cursor = None
        
        try:
            # Set autocommit to False for batch transaction
            self.conn.autocommit = False
            cursor = self.conn.cursor()
            
            for item in batch_data:
                resume_data = item['resume_data']
                s3_bucket = item['s3_bucket']
                s3_key = item['s3_key']
                original_filename = item.get('original_filename', '')
                file_type = item.get('file_type', '')
                resume_id = item.get('resume_id')
                
                if not resume_id:
                    resume_id = str(uuid.uuid4())
                
                # Extract fields from resume_data
                name = resume_data.get('name', resume_data.get('full_name', ''))
                email = resume_data.get('email', '')
                phone = resume_data.get('phone_number', '')
                linkedin = resume_data.get('linkedin_url', resume_data.get('linkedin', ''))
                address = resume_data.get('address', '')
                
                # Prepare data for upsert
                try:
                    cursor.execute("""
                        INSERT INTO resume_pii (
                            resume_id, name, email, phone_number, linkedin_url, address,
                            s3_bucket, s3_key, original_filename, file_type, created_dt, updated_dt
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT (resume_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            email = EXCLUDED.email,
                            phone_number = EXCLUDED.phone_number,
                            linkedin_url = EXCLUDED.linkedin_url,
                            address = EXCLUDED.address,
                            s3_bucket = EXCLUDED.s3_bucket,
                            s3_key = EXCLUDED.s3_key,
                            original_filename = EXCLUDED.original_filename,
                            file_type = EXCLUDED.file_type,
                            updated_dt = NOW()
                    """, (
                        resume_id, name, email, phone, linkedin, address,
                        s3_bucket, s3_key, original_filename, file_type
                    ))
                    
                    resume_ids.append(resume_id)
                    
                except Exception as e:
                    logger.error(f"Error inserting/updating resume {resume_id}: {str(e)}")
                    # Attempt to reconnect and retry once
                    if "connection" in str(e).lower():
                        if self.ensure_connection():
                            cursor = self.conn.cursor()
                            try:
                                cursor.execute("""
                                    INSERT INTO resume_pii (
                                        resume_id, name, email, phone_number, linkedin_url, address,
                                        s3_bucket, s3_key, original_filename, file_type, created_dt, updated_dt
                                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                                    ON CONFLICT (resume_id) DO UPDATE SET
                                        name = EXCLUDED.name,
                                        email = EXCLUDED.email,
                                        phone_number = EXCLUDED.phone_number,
                                        linkedin_url = EXCLUDED.linkedin_url,
                                        address = EXCLUDED.address,
                                        s3_bucket = EXCLUDED.s3_bucket,
                                        s3_key = EXCLUDED.s3_key,
                                        original_filename = EXCLUDED.original_filename,
                                        file_type = EXCLUDED.file_type,
                                        updated_dt = NOW()
                                """, (
                                    resume_id, name, email, phone, linkedin, address,
                                    s3_bucket, s3_key, original_filename, file_type
                                ))
                                resume_ids.append(resume_id)
                            except Exception as retry_error:
                                logger.error(f"Retry failed for resume {resume_id}: {str(retry_error)}")
            
            # Commit the transaction after all inserts
            self.conn.commit()
            logger.info(f"Batch inserted/updated {len(resume_ids)} resume PII records")
            
        except Exception as e:
            # Roll back the transaction if there was an error
            if self.conn and not self.conn.closed:
                self.conn.rollback()
                
            logger.error(f"Error in batch insert: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Try to reconnect and restart the whole batch if it's a connection issue
            if "connection" in str(e).lower() and self.ensure_connection():
                return self.batch_insert_resume_pii(batch_data)
            
        finally:
            # Reset autocommit to true
            if self.conn and not self.conn.closed:
                self.conn.autocommit = True
                
            # Close cursor
            if cursor:
                cursor.close()
                
        return resume_ids
    
    def get_resume_pii(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """
        Get resume PII data by ID
        
        Args:
            resume_id: Resume ID (UUID)
            
        Returns:
            Resume PII data dictionary or None if not found
        """
        if not ENABLE_POSTGRES:
            return None
            
        try:
            if not self.conn or self.conn.closed:
                self.connect()
                
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = "SELECT * FROM resume_pii WHERE resume_id = %s;"
                cursor.execute(query, (resume_id,))
                result = cursor.fetchone()
                
                return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Error getting resume PII data: {str(e)}")
            return None 