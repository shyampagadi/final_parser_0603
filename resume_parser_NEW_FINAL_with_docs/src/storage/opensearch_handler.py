import json
import logging
import os
import time
import functools
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

from config.config import (
    ENABLE_OPENSEARCH,
    OPENSEARCH_ENDPOINT,
    OPENSEARCH_SERVERLESS,
    OPENSEARCH_COLLECTION_NAME,
    OPENSEARCH_INDEX,
    OPENSEARCH_REGION,
    OPENSEARCH_USERNAME,
    OPENSEARCH_PASSWORD,
    AWS_REGION,
    get_aws_credentials
)
from src.utils.bedrock_embeddings import BedrockEmbeddings

logger = logging.getLogger(__name__)

# Path to schema file
SCHEMA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', 'opensearch_schema.txt')

# Simple LRU cache for embeddings
class LRUCache:
    def __init__(self, capacity: int = 100):
        self.cache = {}
        self.capacity = capacity
        self.usage_order = []
    
    def get(self, key: str) -> Optional[List[float]]:
        if key in self.cache:
            # Update usage order
            self.usage_order.remove(key)
            self.usage_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: List[float]) -> None:
        if key in self.cache:
            # Update existing entry
            self.usage_order.remove(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used item
            oldest_key = self.usage_order.pop(0)
            del self.cache[oldest_key]
        
        # Add new entry
        self.cache[key] = value
        self.usage_order.append(key)

class OpenSearchHandler:
    """Handler for Amazon OpenSearch operations"""
    
    def __init__(self):
        """Initialize OpenSearch handler"""
        if not ENABLE_OPENSEARCH:
            logger.warning("OpenSearch is disabled in configuration")
            return
            
        self.endpoint = OPENSEARCH_ENDPOINT
        self.region = OPENSEARCH_REGION or AWS_REGION
        self.is_serverless = OPENSEARCH_SERVERLESS
        # Use the index name directly without any collection prefix - confirmed working pattern
        self.index_name = OPENSEARCH_INDEX or 'resume-embeddings'
        # We keep the collection name for reference, but don't use it in index paths
        self.collection_name = OPENSEARCH_COLLECTION_NAME
        
        # Initialize embedding cache
        self.embedding_cache = LRUCache(capacity=200)
        
        # Defer client initialization until needed
        self._client = None
        self._embeddings_client = None
        
        if not self.endpoint:
            logger.error("OpenSearch endpoint is not provided")
            raise ValueError("Missing OpenSearch endpoint")
        
        logger.info(f"Initialized OpenSearch handler for endpoint: {self.endpoint}, index: {self.index_name}")
        # Add clarifying log message about index access pattern
        if self.is_serverless and self.collection_name:
            logger.info("Using direct index name access (not collection/index format) for OpenSearch Serverless")
    
    @property
    def client(self):
        """Lazy load the OpenSearch client"""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    @property
    def embeddings_client(self):
        """Lazy load the embeddings client"""
        if self._embeddings_client is None:
            from src.utils.bedrock_embeddings import BedrockEmbeddings
            self._embeddings_client = BedrockEmbeddings()
            logger.info("Initialized embeddings client")
        return self._embeddings_client
    
    def _initialize_client(self):
        """Initialize the OpenSearch client with proper authentication"""
        try:
            # Service name is different for OpenSearch Serverless vs Domain
            service_name = 'aoss' if self.is_serverless else 'es'
            
            if self.is_serverless:
                # For OpenSearch Serverless
                session = boto3.Session()
                credentials = session.get_credentials()
                
                if credentials is None:
                    raise ValueError("Missing AWS credentials for OpenSearch Serverless")
                    
                awsauth = AWS4Auth(
                    credentials.access_key,
                    credentials.secret_key,
                    self.region,
                    service_name,
                    session_token=credentials.token
                )
                
                # Handle token issue with some versions of requests-aws4auth
                if hasattr(awsauth, 'auth') and 'X-Amz-Security-Token' in awsauth.auth:
                    awsauth.auth['X-Amz-Security-Token'] = credentials.token
            elif OPENSEARCH_USERNAME and OPENSEARCH_PASSWORD:
                # For OpenSearch Domain with Basic Auth
                self._client = OpenSearch(
                    hosts=[{'host': self.endpoint, 'port': 443}],
                    http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
                    use_ssl=True,
                    verify_certs=True,
                    connection_class=RequestsHttpConnection,
                    timeout=30
                )
                return
            else:
                # For OpenSearch Domain with IAM
                credentials = get_aws_credentials()
                awsauth = AWS4Auth(
                    credentials.get('aws_access_key_id', ''),
                    credentials.get('aws_secret_access_key', ''),
                    self.region,
                    service_name
                )
            
            # Create client with auth
            self._client = OpenSearch(
                hosts=[{'host': self.endpoint, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=30
            )
            
            # Test connection
            try:
                indices = self._client.indices.get_alias("*")
                logger.info(f"Successfully connected to OpenSearch. Found {len(indices)} indices.")
            except Exception as e:
                if '404' in str(e):
                    logger.info("Connected to OpenSearch (no indices found yet)")
                else:
                    logger.warning(f"Connection test warning: {str(e)[:300]}")
            
        except Exception as e:
            logger.error(f"Error initializing OpenSearch client: {str(e)[:300]}")
            raise
    
    def ensure_index_exists(self):
        """Ensure the OpenSearch index exists with the proper mappings"""
        if not ENABLE_OPENSEARCH:
            return False
            
        try:
            # Check if index already exists
            try:
                exists = self.client.indices.exists(index=self.index_name)
                if exists:
                    logger.info(f"OpenSearch index '{self.index_name}' already exists")
                    return True
            except Exception as e:
                logger.warning(f"Error checking if index exists: {str(e)[:300]}")
                
            # Define mapping for resume data with vector search capabilities - EXACTLY matching opensearch_schema.txt
            mapping = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 1024,
                        "analysis": {
                            "normalizer": {
                                "lowercase": {
                                    "type": "custom",
                                    "filter": ["lowercase"]
                                }
                            }
                        },
                        "number_of_shards": 1,
                        "number_of_replicas": 1
                    }
                },
                "mappings": {
                    "properties": {
                        "resume_id": {
                            "type": "keyword"
                        },
                        "resume_embedding": {
                            "type": "knn_vector",
                            "dimension": 1024,
                            "method": {
                                "name": "hnsw",
                                "engine": "nmslib",
                                "space_type": "cosinesimil",
                                "parameters": {
                                    "ef_construction": 1024,
                                    "m": 48
                                }
                            }
                        },
                        "summary": {
                            "type": "text"
                        },
                        "total_experience": {
                            "type": "float"
                        },
                        "skills": {
                            "type": "keyword",
                            "normalizer": "lowercase"
                        },
                        "positions": {
                            "type": "keyword"
                        },
                        "companies": {
                            "type": "nested",
                            "properties": {
                                "name": {
                                    "type": "keyword"
                                },
                                "duration": {
                                    "type": "text"
                                },
                                "description": {
                                    "type": "text"
                                },
                                "role": {
                                    "type": "keyword"
                                },
                                "technologies": {
                                    "type": "keyword"
                                }
                            }
                        },
                        "education": {
                            "type": "nested",
                            "properties": {
                                "degree": {
                                    "type": "keyword"
                                },
                                "institution": {
                                    "type": "keyword"
                                },
                                "year": {
                                    "type": "short"
                                }
                            }
                        },
                        "certifications": {
                            "type": "keyword"
                        },
                        "achievements": {
                            "type": "nested",
                            "properties": {
                                "type": {
                                    "type": "keyword"
                                },
                                "description": {
                                    "type": "text"
                                },
                                "metrics": {
                                    "type": "text"
                                }
                            }
                        },
                        "industries": {
                            "type": "keyword"
                        },
                        "projects": {
                            "type": "nested",
                            "properties": {
                                "name": {
                                    "type": "text"
                                },
                                "description": {
                                    "type": "text"
                                },
                                "technologies": {
                                    "type": "keyword"
                                },
                                "duration_months": {
                                    "type": "short"
                                },
                                "role": {
                                    "type": "keyword"
                                },
                                "metrics": {
                                    "type": "text"
                                }
                            }
                        },
                        "created_dt": {
                            "type": "date",
                            "format": "strict_date_optional_time"
                        },
                        "updated_dt": {
                            "type": "date",
                            "format": "strict_date_optional_time"
                        },
                        "collection_name": {
                            "type": "keyword"
                        }
                    }
                }
            }
            
            # Create the index
            logger.info(f"Creating OpenSearch index '{self.index_name}' with mapping")
            response = self.client.indices.create(
                index=self.index_name,
                body=mapping
            )
            
            logger.info(f"Index creation response: {response}")
            return 'acknowledged' in response and response['acknowledged']
            
        except Exception as e:
            error_msg = str(e)
            # Handle index already exists error gracefully
            if "resource_already_exists_exception" in error_msg or "index_already_exists_exception" in error_msg:
                logger.info(f"Index '{self.index_name}' already exists")
                return True
                
            logger.error(f"Error creating OpenSearch index: {error_msg[:300]}")
            return False
    
    def _prepare_document(self, resume_data: Dict[str, Any], resume_id: str, resume_text: str) -> Dict[str, Any]:
        """
        Prepare a document that matches the OpenSearch schema
        
        Args:
            resume_data: Resume data extracted by LLM
            resume_id: Unique identifier for the resume
            resume_text: Raw resume text
            
        Returns:
            Document ready for OpenSearch
        """
        # Create base document with metadata fields
        document = {
            'resume_id': resume_id,
            'collection_name': resume_data.get('collection_name', self.collection_name),
            'created_dt': datetime.now().isoformat(),
            'updated_dt': datetime.now().isoformat()
        }
        
        # ===== VECTOR EMBEDDING =====
        # Check if a pre-generated embedding is provided in resume_data
        if '_embedding_vector' in resume_data:
            document['resume_embedding'] = resume_data['_embedding_vector']
            logger.info("Using pre-generated embedding from resume_data")
        else:
            # Generate embedding for structured resume data + sanitized raw text
            try:
                # Import here to avoid circular imports
                from src.utils.bedrock_embeddings import create_standardized_text, BedrockEmbeddings
                
                # Create a standardized text representation from the resume data
                structured_text = create_standardized_text(resume_data)
                
                # Sanitize raw resume text to remove PII data
                sanitized_text = self._sanitize_resume_text(resume_text, resume_data)
                
                # Append sanitized resume text to capture any details not in structured format
                # This ensures we don't lose any information from the original resume
                combined_text = structured_text + "\n\n# ORIGINAL RESUME TEXT\n" + sanitized_text
                
                # Use a hash of the combined text as cache key
                import hashlib
                cache_key = hashlib.md5(combined_text.encode('utf-8')).hexdigest()
                embedding = self.embedding_cache.get(cache_key)
                
                if embedding is None:
                    # Cache miss - generate new embedding
                    start_time = time.time()
                    # Always use 1024 dimensions for consistency with OpenSearch schema
                    embedding = self.embeddings_client.get_embedding(combined_text, dimension=1024)
                    
                    self.embedding_cache.put(cache_key, embedding)
                    logger.info(f"Generated new embedding in {time.time() - start_time:.2f}s with dimension {len(embedding)}")
                else:
                    logger.info("Using cached embedding")
                    
                document['resume_embedding'] = embedding
                logger.info("Added resume embedding to document")
                
                # Ensure embedding has the correct dimension (1024)
                if len(document['resume_embedding']) != 1024:
                    logger.warning(f"Embedding dimension mismatch: got {len(document['resume_embedding'])}, expected 1024")
                    
                    # Truncate or pad as needed
                    if len(document['resume_embedding']) > 1024:
                        logger.info(f"Truncating embedding from {len(document['resume_embedding'])} to 1024")
                        document['resume_embedding'] = document['resume_embedding'][:1024]
                    else:
                        logger.info(f"Padding embedding from {len(document['resume_embedding'])} to 1024")
                        padding = [0.0] * (1024 - len(document['resume_embedding']))
                        document['resume_embedding'] = document['resume_embedding'] + padding
                        
                logger.info(f"Final embedding dimension: {len(document['resume_embedding'])}")
            except Exception as e:
                logger.warning(f"Could not generate resume embedding: {str(e)[:200]}")
                
                # Add a zero vector as fallback to avoid schema errors
                document['resume_embedding'] = [0.0] * 1024
                logger.info("Added zero vector as fallback embedding")
        
        # ===== SCALAR FIELDS =====
        
        # Total Experience (float)
        if 'total_experience' in resume_data:
            try:
                document['total_experience'] = float(resume_data['total_experience'])
            except (ValueError, TypeError):
                pass
        
        # Summary (text)
        if 'summary' in resume_data:
            if isinstance(resume_data['summary'], dict) and 'text' in resume_data['summary']:
                document['summary'] = resume_data['summary']['text']
            elif isinstance(resume_data['summary'], str):
                document['summary'] = resume_data['summary']
        
        # Positions (keyword array)
        if 'positions' in resume_data:
            if isinstance(resume_data['positions'], list):
                document['positions'] = resume_data['positions']
            elif isinstance(resume_data['positions'], dict) and 'values' in resume_data['positions']:
                document['positions'] = resume_data['positions']['values']
            elif isinstance(resume_data['positions'], str):
                document['positions'] = [resume_data['positions']]
        
        # Skills (keyword array)
        if 'skills' in resume_data:
            if isinstance(resume_data['skills'], list):
                document['skills'] = resume_data['skills']
            elif isinstance(resume_data['skills'], dict) and 'values' in resume_data['skills']:
                document['skills'] = resume_data['skills']['values']
            elif isinstance(resume_data['skills'], str):
                document['skills'] = [s.strip() for s in resume_data['skills'].split(',')]
        
        # Industries (keyword array)
        if 'industries' in resume_data:
            if isinstance(resume_data['industries'], list):
                document['industries'] = resume_data['industries']
            elif isinstance(resume_data['industries'], dict) and 'values' in resume_data['industries']:
                document['industries'] = resume_data['industries']['values']
            elif isinstance(resume_data['industries'], str):
                document['industries'] = [i.strip() for i in resume_data['industries'].split(',')]
        
        # Certifications (keyword array)
        if 'certifications' in resume_data:
            if isinstance(resume_data['certifications'], list):
                document['certifications'] = resume_data['certifications']
            elif isinstance(resume_data['certifications'], dict) and 'values' in resume_data['certifications']:
                document['certifications'] = resume_data['certifications']['values']
            elif isinstance(resume_data['certifications'], str):
                document['certifications'] = [c.strip() for c in resume_data['certifications'].split(',')]
        
        # ===== NESTED OBJECTS =====
        
        # Companies (nested)
        if 'companies' in resume_data and isinstance(resume_data['companies'], list):
            document['companies'] = []
            for company in resume_data['companies']:
                company_obj = {}
                
                # Name (keyword)
                if 'name' in company:
                    company_obj['name'] = self._extract_text_value(company['name'])
                
                # Duration - Handle the date_range field correctly
                if 'duration' in company:
                    duration_str = self._extract_text_value(company['duration'])
                    
                    # Store the original duration string
                    company_obj['duration'] = duration_str
                    
                    # Don't try to create a date_range object - this will be handled by OpenSearch mapping
                    # The date_range field is a special field that OpenSearch will parse from the string
                    # Format should be MM/YYYY-MM/YYYY
                
                # Description (text)
                if 'description' in company:
                    company_obj['description'] = self._extract_text_value(company['description'])
                
                # Role (keyword)
                if 'role' in company:
                    company_obj['role'] = self._extract_text_value(company['role'])
                
                # Technologies (keyword array)
                if 'technologies' in company:
                    tech_value = self._extract_text_value(company['technologies'])
                    if isinstance(tech_value, list):
                        company_obj['technologies'] = tech_value
                    elif isinstance(tech_value, str):
                        company_obj['technologies'] = [t.strip() for t in tech_value.split(',')]
                    else:
                        company_obj['technologies'] = []
                
                if company_obj:  # Only add if not empty
                    document['companies'].append(company_obj)
        
        # Education (nested)
        if 'education' in resume_data and isinstance(resume_data['education'], list):
            document['education'] = []
            for edu in resume_data['education']:
                if not isinstance(edu, dict):
                    continue
                    
                edu_obj = {}
                
                # Degree (keyword)
                if 'degree' in edu:
                    edu_obj['degree'] = self._extract_text_value(edu['degree'])
                
                # Institution (keyword)
                if 'institution' in edu:
                    edu_obj['institution'] = self._extract_text_value(edu['institution'])
                
                # Year (short)
                if 'year' in edu:
                    try:
                        year_value = self._extract_text_value(edu['year'])
                        if year_value and isinstance(year_value, (int, str)):
                            edu_obj['year'] = int(year_value)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert education year to integer: {edu['year']}")
                
                if edu_obj:  # Only add if not empty
                    document['education'].append(edu_obj)
        
        # Achievements (nested)
        if 'achievements' in resume_data and isinstance(resume_data['achievements'], list):
            document['achievements'] = []
            for achievement in resume_data['achievements']:
                if not isinstance(achievement, dict):
                    continue
                    
                achievement_obj = {}
                
                # Type (keyword)
                if 'type' in achievement:
                    achievement_obj['type'] = self._extract_text_value(achievement['type'])
                
                # Description (text)
                if 'description' in achievement:
                    achievement_obj['description'] = self._extract_text_value(achievement['description'])
                
                # Metrics (text)
                if 'metrics' in achievement:
                    achievement_obj['metrics'] = self._extract_text_value(achievement['metrics'])
                
                if achievement_obj:  # Only add if not empty
                    document['achievements'].append(achievement_obj)
        
        # Projects (nested)
        if 'projects' in resume_data and isinstance(resume_data['projects'], list):
            document['projects'] = []
            for project in resume_data['projects']:
                if not isinstance(project, dict):
                    continue
                    
                project_obj = {}
                
                # Name (text)
                if 'name' in project:
                    project_obj['name'] = self._extract_text_value(project['name'])
                
                # Description (text)
                if 'description' in project:
                    project_obj['description'] = self._extract_text_value(project['description'])
                
                # Technologies (keyword array)
                if 'technologies' in project:
                    tech_value = self._extract_text_value(project['technologies'])
                    if isinstance(tech_value, list):
                        project_obj['technologies'] = tech_value
                    elif isinstance(tech_value, str):
                        project_obj['technologies'] = [t.strip() for t in tech_value.split(',')]
                    else:
                        project_obj['technologies'] = []
                
                # Duration months (short)
                if 'duration_months' in project:
                    try:
                        duration_value = self._extract_text_value(project['duration_months'])
                        if duration_value and isinstance(duration_value, (int, str)):
                            project_obj['duration_months'] = int(duration_value)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert project duration to integer: {project['duration_months']}")
                
                # Role (keyword)
                if 'role' in project:
                    project_obj['role'] = self._extract_text_value(project['role'])
                
                # Metrics (text)
                if 'metrics' in project:
                    project_obj['metrics'] = self._extract_text_value(project['metrics'])
                
                if project_obj:  # Only add if not empty
                    document['projects'].append(project_obj)
        
        return document
    
    def _extract_text_value(self, field_value: Any) -> Any:
        """
        Extract text value from field which might be a string, dict with 'text' or 'values' key,
        or some other structure.
        
        Args:
            field_value: The field value to extract text from
            
        Returns:
            Extracted text value
        """
        if isinstance(field_value, dict):
            if 'text' in field_value:
                return field_value['text']
            elif 'values' in field_value:
                return field_value['values']
            elif 'value' in field_value:
                return field_value['value']
        return field_value
    
    def store_resume(self, resume_data: Dict[str, Any], resume_id: str, resume_text: str, pre_generated_embedding: Optional[List[float]] = None) -> bool:
        """
        Store resume data in OpenSearch
        
        Args:
            resume_data: Resume data extracted by LLM
            resume_id: Unique identifier for the resume
            resume_text: Raw resume text
            pre_generated_embedding: Optional pre-generated embedding vector
            
        Returns:
            True if successful, False otherwise
        """
        if not ENABLE_OPENSEARCH:
            logger.info("OpenSearch storage is disabled")
            return False
            
        try:
            # Ensure the index exists with proper mappings
            if not self.ensure_index_exists():
                logger.warning("Could not ensure OpenSearch index exists, but will try to store data anyway")
            
            logger.info(f"Storing resume data for resume_id={resume_id}")
            
            # Prepare document based on schema
            document = self._prepare_document(resume_data, resume_id, resume_text)
            
            # If we have a pre-generated embedding, use it
            if pre_generated_embedding is not None:
                logger.info("Using pre-generated embedding")
                # Pad to 1024 if needed for schema compatibility
                if len(pre_generated_embedding) < 1024:
                    pre_generated_embedding = pre_generated_embedding + [0.0] * (1024 - len(pre_generated_embedding))
                # Truncate if longer than 1024 (Titan model max)
                elif len(pre_generated_embedding) > 1024:
                    pre_generated_embedding = pre_generated_embedding[:1024]
                document['resume_embedding'] = pre_generated_embedding
            
            # Store in OpenSearch with retries
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Storing document in OpenSearch (attempt {attempt+1}/{max_retries})")
                    
                    # Always use the non-ID approach for serverless mode to avoid "Document ID is not supported" error
                    if self.is_serverless:
                        # For serverless, don't use document ID parameter
                        response = self.client.index(
                            index=self.index_name,
                            body=document
                        )
                    else:
                        # For managed OpenSearch, we can specify document ID
                        response = self.client.index(
                            index=self.index_name,
                            body=document,
                            id=resume_id,
                            refresh=True
                        )
                    
                    if response.get('result') in ('created', 'updated'):
                        logger.info(f"Successfully stored resume in OpenSearch with ID: {resume_id}")
                        return True
                    else:
                        logger.warning(f"Unexpected response: {json.dumps(response)[:200]}")
                        
                except Exception as e:
                    error_msg = str(e)[:300]
                    logger.warning(f"Error on attempt {attempt+1}/{max_retries}: {error_msg}")
                    
                    # Handle schema issues
                    if 'mapper_parsing_exception' in error_msg:
                        logger.info("Schema error detected, simplifying document")
                        # Remove problematic fields
                        if 'resume_embedding' in document:
                            logger.info("Removing resume_embedding field due to schema error")
                            del document['resume_embedding']
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        logger.error(f"All {max_retries} attempts failed")
                        return False
            
            return False
                
        except Exception as e:
            logger.error(f"Error storing resume: {str(e)[:300]}")
            return False
    
    def get_resume(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a resume by ID"""
        if not ENABLE_OPENSEARCH:
            return None
            
        try:
            # For serverless, we need to use search instead of direct ID lookup
            if self.is_serverless:
                logger.info(f"Using search query to find resume with ID: {resume_id} (serverless mode)")
                response = self.client.search(
                    index=self.index_name,
                    body={
                        "query": {
                            "term": {
                                "resume_id.keyword": resume_id
                            }
                        }
                    }
                )
                
                if response.get('hits', {}).get('total', {}).get('value', 0) > 0:
                    return response['hits']['hits'][0]['_source']
                else:
                    logger.warning(f"Resume with ID {resume_id} not found via search")
                    return None
            else:
                # For managed OpenSearch, we can use direct ID lookup
                response = self.client.get(
                    index=self.index_name,
                    id=resume_id
                )
                
                if response.get('found'):
                    return response.get('_source')
                else:
                    logger.warning(f"Resume with ID {resume_id} not found")
                    return None
                
        except Exception as e:
            logger.error(f"Error retrieving resume: {str(e)[:300]}")
            return None
    
    def delete_resume(self, resume_id: str) -> bool:
        """Delete a resume by ID"""
        if not ENABLE_OPENSEARCH:
            return False
            
        try:
            # For serverless, we need to use delete_by_query instead of direct ID deletion
            if self.is_serverless:
                logger.info(f"Using delete_by_query to delete resume with ID: {resume_id} (serverless mode)")
                response = self.client.delete_by_query(
                    index=self.index_name,
                    body={
                        "query": {
                            "term": {
                                "resume_id.keyword": resume_id
                            }
                        }
                    }
                )
                
                deleted = response.get('deleted', 0)
                if deleted > 0:
                    logger.info(f"Successfully deleted {deleted} documents with resume_id: {resume_id}")
                    return True
                else:
                    logger.warning(f"No documents found to delete with resume_id: {resume_id}")
                    return False
            else:
                # For managed OpenSearch, we can use direct ID deletion
                response = self.client.delete(
                    index=self.index_name,
                    id=resume_id,
                    refresh=not self.is_serverless
                )
                
                if response.get('result') == 'deleted':
                    logger.info(f"Successfully deleted resume with ID: {resume_id}")
                    return True
                else:
                    logger.warning(f"Unexpected response when deleting resume: {response}")
                    return False
                
        except Exception as e:
            logger.error(f"Error deleting resume: {str(e)[:300]}")
            return False
    
    def _sanitize_resume_text(self, resume_text: str, resume_data: Dict[str, Any]) -> str:
        """
        Sanitize raw resume text to remove PII data
        
        Args:
            resume_text: Original raw resume text
            resume_data: Parsed resume data containing PII fields
            
        Returns:
            Sanitized text with PII data removed
        """
        # Get PII data to remove
        pii_data = []
        
        # Add full name if available
        if full_name := resume_data.get('full_name'):
            if isinstance(full_name, str) and full_name.strip():
                pii_data.append(full_name)
                # Also add first and last name parts
                name_parts = full_name.split()
                if len(name_parts) > 0:
                    pii_data.append(name_parts[0])  # First name
                if len(name_parts) > 1:
                    pii_data.append(name_parts[-1])  # Last name
        
        # Add email if available
        if email := resume_data.get('email'):
            if isinstance(email, str) and email.strip():
                pii_data.append(email)
        
        # Add phone number if available
        if phone := resume_data.get('phone_number'):
            if isinstance(phone, str) and phone.strip():
                pii_data.append(phone)
        
        # Add LinkedIn profile if available
        if linkedin := resume_data.get('linkedin'):
            if isinstance(linkedin, str) and linkedin.strip():
                pii_data.append(linkedin)
        
        # Add address if available
        if address := resume_data.get('address'):
            if isinstance(address, str) and address.strip():
                pii_data.append(address)
                # Also add individual parts of the address
                address_parts = address.split(',')
                for part in address_parts:
                    if part.strip():
                        pii_data.append(part.strip())
        
        # Replace PII data with placeholders
        sanitized_text = resume_text
        for item in pii_data:
            if len(item) > 3:  # Only replace items of reasonable length
                sanitized_text = sanitized_text.replace(item, "[REDACTED]")
        
        return sanitized_text

    def store_resume_data(self, resume_data: Dict[str, Any], resume_id: Optional[str] = None, 
                      resume_text: Optional[str] = None, collection_name: Optional[str] = None) -> str:
        """
        Store resume data in OpenSearch with vector embedding
        
        Args:
            resume_data: Dictionary containing structured resume data
            resume_id: Optional resume UUID (will be generated if not provided)
            resume_text: Optional full text of the resume for embedding
            collection_name: Optional collection/category name
            
        Returns:
            Resume ID (UUID)
        """
        if not ENABLE_OPENSEARCH:
            return resume_id or str(uuid.uuid4())
            
        # Ensure the index exists before trying to store data
        self.ensure_index_exists()
            
        try:
            # Use provided ID or generate one
            if not resume_id:
                resume_id = str(uuid.uuid4())
                
            # Add collection name to resume data if provided
            if collection_name:
                resume_data['collection_name'] = collection_name
            
            # Store resume data - note that store_resume returns a boolean, not resume_id
            success = self.store_resume(resume_data, resume_id, resume_text)
            if success:
                logger.info(f"Successfully stored resume data in OpenSearch with ID: {resume_id}")
            else:
                logger.warning(f"Failed to store resume data in OpenSearch with ID: {resume_id}")
                
            # Always return the resume_id regardless of storage success
            return resume_id
                
        except Exception as e:
            logger.error(f"Error storing resume data: {str(e)[:300]}")
            return resume_id or str(uuid.uuid4())

    def resume_exists(self, resume_id: str) -> bool:
        """
        Check if a resume exists in the index
        
        Args:
            resume_id: The resume ID to check
            
        Returns:
            True if the resume exists, False otherwise
        """
        if not ENABLE_OPENSEARCH:
            return False
            
        try:
            # For serverless, we need to use search instead of direct ID lookup
            if self.is_serverless:
                response = self.client.search(
                    index=self.index_name,
                    body={
                        "query": {
                            "term": {
                                "resume_id.keyword": resume_id
                            }
                        }
                    },
                    size=1  # We only need to know if it exists
                )
                
                return response.get('hits', {}).get('total', {}).get('value', 0) > 0
            else:
                # For managed OpenSearch, we can use direct ID lookup
                return self.client.exists(
                    index=self.index_name,
                    id=resume_id
                )
                
        except Exception as e:
            logger.error(f"Error checking if resume exists: {str(e)[:300]}")
            return False 