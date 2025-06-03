import logging
import boto3
import time
from src.storage.opensearch_handler import OpenSearchHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_opensearch_connection():
    """Test connection to OpenSearch and check index status"""
    try:
        handler = OpenSearchHandler()
        logger.info(f"OpenSearch handler initialized for endpoint: {handler.endpoint}")
        logger.info(f"Is serverless: {handler.is_serverless}")
        
        # Test connection
        logger.info("Testing connection...")
        indices = handler.client.indices.get_alias("*")
        logger.info(f"Connected successfully. Found indices: {list(indices.keys())}")
        
        # Check if our index exists
        index_name = handler.index_name
        logger.info(f"Checking for index: {index_name}")
        exists = handler.client.indices.exists(index=index_name)
        logger.info(f"Index '{index_name}' exists: {exists}")
        
        if exists:
            # Check stats
            try:
                stats = handler.client.indices.stats(index=index_name)
                doc_count = stats['_all']['primaries']['docs']['count']
                logger.info(f"Index contains {doc_count} documents")
            except Exception as e:
                logger.error(f"Error getting stats: {str(e)}")
            
            # Try a simple search
            try:
                logger.info("Running a match_all query...")
                results = handler.client.search(
                    index=index_name,
                    body={"query": {"match_all": {}}}
                )
                total_hits = results['hits']['total']['value']
                logger.info(f"Query returned {total_hits} hits")
                
                if total_hits > 0:
                    logger.info("First hit:")
                    logger.info(results['hits']['hits'][0]['_source'])
                else:
                    logger.warning("No documents found in the index!")
            except Exception as e:
                logger.error(f"Error running search: {str(e)}")

        # Test creating the index
        if not exists:
            logger.info("Index doesn't exist, trying to create it...")
            created = handler.ensure_index_exists()
            logger.info(f"Index creation result: {created}")
            
        # Test saving a document
        logger.info("Testing document storage...")
        test_data = {
            "summary": "Test resume",
            "total_experience": 5.0,
            "skills": ["Python", "AWS", "OpenSearch"]
        }
        resume_id = handler.store_resume_data(test_data, resume_text="This is a test resume")
        logger.info(f"Stored test document with ID: {resume_id}")
        
        # Wait a moment for indexing
        logger.info("Waiting for indexing...")
        time.sleep(2)
        
        # Try to retrieve the document directly by ID
        try:
            logger.info(f"Getting document directly by ID: {resume_id}")
            try:
                doc = handler.client.get(index=index_name, id=resume_id)
                if doc.get('found'):
                    logger.info("Document found by direct ID lookup!")
                    logger.info(doc['_source'])
                else:
                    logger.warning("Document NOT found by direct ID lookup!")
            except Exception as e:
                logger.error(f"Error getting document by ID: {str(e)}")
                
            # Try a refresh and search again
            logger.info("Refreshing index and trying search again...")
            try:
                handler.client.indices.refresh(index=index_name)
            except Exception as e:
                logger.warning(f"Error refreshing index: {str(e)}")
                
            # Check if it was stored
            logger.info(f"Checking if test document was stored (ID: {resume_id})...")
            results = handler.client.search(
                index=index_name,
                body={
                    "query": {
                        "term": {
                            "resume_id.keyword": resume_id
                        }
                    }
                }
            )
            if results['hits']['total']['value'] > 0:
                logger.info("Test document found by term query!")
                logger.info(results['hits']['hits'][0]['_source'])
            else:
                logger.warning("Test document not found in search results!")
                
                # Try match query instead
                results = handler.client.search(
                    index=index_name,
                    body={
                        "query": {
                            "match": {
                                "resume_id": resume_id
                            }
                        }
                    }
                )
                if results['hits']['total']['value'] > 0:
                    logger.info("Test document found by match query!")
                    logger.info(results['hits']['hits'][0]['_source'])
                else:
                    logger.warning("Test document not found with match query either!")
                    
                    # Try wildcard
                    logger.info("Trying wildcard query...")
                    results = handler.client.search(
                        index=index_name,
                        body={
                            "query": {
                                "wildcard": {
                                    "resume_id": "*"
                                }
                            }
                        }
                    )
                    if results['hits']['total']['value'] > 0:
                        logger.info(f"Found {results['hits']['total']['value']} documents with wildcard query!")
                        logger.info(f"First document: {results['hits']['hits'][0]['_source']}")
                    else:
                        logger.warning("No documents found with wildcard query!")
                        
                    # Try match_all again
                    logger.info("Trying match_all query one more time...")
                    results = handler.client.search(
                        index=index_name,
                        body={
                            "query": {
                                "match_all": {}
                            }
                        }
                    )
                    logger.info(f"Match_all query returned {results['hits']['total']['value']} documents")
        except Exception as e:
            logger.error(f"Error searching for test document: {str(e)}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_opensearch_connection() 