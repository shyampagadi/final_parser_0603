import os
import json
import logging
import uuid
import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime
from typing import Dict, Any, Optional, List

from config.config import (
    ENABLE_DYNAMODB,
    DYNAMODB_TABLE_NAME,
    DYNAMODB_REGION,
    DYNAMODB_ENDPOINT,
    DYNAMODB_READ_CAPACITY_UNITS,
    DYNAMODB_WRITE_CAPACITY_UNITS,
    get_aws_credentials
)

logger = logging.getLogger(__name__)

class DynamoDBHandler:
    """Handler for Amazon DynamoDB operations"""
    
    def __init__(self, table_name: Optional[str] = None, region: Optional[str] = None, endpoint_url: Optional[str] = None):
        """
        Initialize DynamoDB handler
        
        Args:
            table_name: DynamoDB table name (optional)
            region: AWS region (optional)
            endpoint_url: DynamoDB endpoint URL for local testing (optional)
        """
        if not ENABLE_DYNAMODB:
            logger.warning("DynamoDB is disabled in configuration. Enable by setting ENABLE_DYNAMODB=true in .env")
            return
            
        self.table_name = table_name or DYNAMODB_TABLE_NAME
        self.region = region or DYNAMODB_REGION
        self.endpoint_url = endpoint_url or DYNAMODB_ENDPOINT
        
        # Validate required configuration
        if not self.table_name:
            logger.error("DynamoDB table name is not provided. Check your .env file.")
            raise ValueError("Missing DynamoDB table name")
        
        # Log configuration
        logger.info(f"DynamoDB Configuration:")
        logger.info(f"- Table Name: {self.table_name}")
        logger.info(f"- Region: {self.region}")
        logger.info(f"- Endpoint URL: {self.endpoint_url or 'Default AWS Endpoint'}")
        
        # Get AWS credentials explicitly for validation
        aws_creds = get_aws_credentials()
        
        # Remove region_name from credentials if it exists to avoid duplication
        if 'region_name' in aws_creds:
            logger.info(f"Region already specified in credentials, using: {aws_creds['region_name']}")
            self.region = aws_creds.pop('region_name')  # Use the region from credentials if present
        
        if not aws_creds.get('aws_access_key_id') or not aws_creds.get('aws_secret_access_key'):
            logger.error("AWS credentials are missing. Check your .env file or AWS configuration.")
            logger.error(f"AWS_ACCESS_KEY_ID: {'Present' if aws_creds.get('aws_access_key_id') else 'Missing'}")
            logger.error(f"AWS_SECRET_ACCESS_KEY: {'Present' if aws_creds.get('aws_secret_access_key') else 'Missing'}")
            raise ValueError("Missing AWS credentials for DynamoDB")
        
        try:
            logger.info("Initializing DynamoDB client...")
            
            # Create DynamoDB resource
            session_args = {
                'region_name': self.region,
                **aws_creds
            }
            
            # Add endpoint_url if specified
            if self.endpoint_url:
                session_args['endpoint_url'] = self.endpoint_url
                
            # Log the actual initialization parameters
            logger.info(f"Initializing boto3 DynamoDB resource with region: {self.region}")
            
            # Create the resource with properly handled parameters
            self.dynamodb = boto3.resource('dynamodb', **session_args)
                
            # Test the connection by listing tables
            try:
                logger.info("Testing DynamoDB connection by listing tables...")
                tables = list(self.dynamodb.tables.all())
                logger.info(f"Successfully connected to DynamoDB. Found {len(tables)} tables.")
            except Exception as conn_error:
                logger.error(f"Failed to list DynamoDB tables: {str(conn_error)}")
                logger.error("There may be an issue with your AWS credentials or permissions.")
                raise
                
            # Reference to the table
            self.table = self.dynamodb.Table(self.table_name)
            
            logger.info(f"Initialized DynamoDB handler for table: {self.table_name} in {self.region}")
            
        except Exception as e:
            logger.error(f"Error initializing DynamoDB handler: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def ensure_table_exists(self):
        """Ensure the DynamoDB table exists, creating it if necessary"""
        if not ENABLE_DYNAMODB:
            return
            
        try:
            # Check if table exists by using describe_table instead of list_tables
            try:
                logger.info(f"Checking if table {self.table_name} exists...")
                table_desc = self.dynamodb.meta.client.describe_table(TableName=self.table_name)
                logger.info(f"DynamoDB table {self.table_name} already exists")
                logger.debug(f"Table description: {json.dumps(table_desc, default=str)}")
                return
            except self.dynamodb.meta.client.exceptions.ResourceNotFoundException:
                # Table doesn't exist, create it
                logger.info(f"Table {self.table_name} doesn't exist, creating...")
                
            # Create table if it doesn't exist
            logger.info(f"Creating DynamoDB table: {self.table_name}")
            
            table = self.dynamodb.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {
                        'AttributeName': 'resume_id',
                        'KeyType': 'HASH'  # Partition key
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'resume_id',
                        'AttributeType': 'S'  # String
                    }
                ],
                BillingMode='PAY_PER_REQUEST'  # Use on-demand capacity for simplicity
            )
            
            # Wait for table to be created
            logger.info(f"Waiting for table {self.table_name} to be created...")
            table.meta.client.get_waiter('table_exists').wait(TableName=self.table_name)
            logger.info(f"Table {self.table_name} created successfully")
            
        except Exception as e:
            # If the error is that the table already exists, that's actually fine
            if "ResourceInUseException" in str(e) and "Table already exists" in str(e):
                logger.info(f"Table {self.table_name} already exists, continuing...")
                return
            else:
                # For other errors, log and raise
                logger.error(f"Error creating or checking DynamoDB table: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                raise
    
    def store_resume_data(self, resume_data: Dict[str, Any], resume_id: Optional[str] = None) -> str:
        """
        Store resume data in DynamoDB
        
        Args:
            resume_data: Resume data dictionary from LLM
            resume_id: Resume ID (UUID) to use (optional, will generate if not provided)
            
        Returns:
            Resume ID (UUID) as string
        """
        if not ENABLE_DYNAMODB:
            # Return resume_id or generate new UUID
            return resume_id or str(uuid.uuid4())
            
        try:
            # Make sure table exists
            self.ensure_table_exists()
            
            # Use provided resume_id or generate a new one
            if not resume_id:
                resume_id = str(uuid.uuid4())
            
            # Add timestamps
            timestamp = datetime.now().isoformat()
            
            # Prepare item for DynamoDB
            item = {
                'resume_id': resume_id,
                'created_at': timestamp,
                'updated_at': timestamp
            }
            
            # List of fields to keep in DynamoDB (based on JSON schema)
            # Only store these fields plus timestamps and resume_id
            fields_to_keep = [
                'summary', 'total_experience', 'skills', 'positions', 'companies', 
                'education', 'certifications', 'achievements', 'industries', 'projects'
            ]
            
            # Create a filtered copy with only the fields we want to keep
            filtered_data = {}
            for field in fields_to_keep:
                if field in resume_data:
                    filtered_data[field] = resume_data[field]
            
            logger.info(f"Storing resume data in DynamoDB with fields: {', '.join(fields_to_keep)}")
            
            # Process the data to ensure it's compatible with DynamoDB
            processed_data = self._process_data_for_dynamodb(filtered_data)
            
            # Add processed data to item
            item.update(processed_data)
            
            # Store in DynamoDB with retries
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Putting item in DynamoDB table {self.table_name} with ID: {resume_id} (attempt {attempt+1}/{max_retries})")
                    self.table.put_item(Item=item)
                    logger.info(f"Successfully stored resume data in DynamoDB with ID: {resume_id}")
                    break
                except Exception as e:
                    logger.error(f"Error storing resume data in DynamoDB (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        import time
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to store resume data in DynamoDB after {max_retries} attempts")
                        raise
            
            return resume_id
            
        except Exception as e:
            logger.error(f"Error storing resume data in DynamoDB: {str(e)}")
            raise
            
    def batch_store_resume_data(self, resume_batch: List[Dict[str, Any]]) -> List[str]:
        """
        Store multiple resume data items in DynamoDB using batch operations
        
        Args:
            resume_batch: List of dictionaries containing 'resume_data' and optional 'resume_id'
            
        Returns:
            List of resume IDs (UUIDs)
        """
        if not ENABLE_DYNAMODB or not resume_batch:
            return [item.get('resume_id', str(uuid.uuid4())) for item in resume_batch]
            
        try:
            # Make sure table exists
            self.ensure_table_exists()
            
            # List of fields to keep in DynamoDB (based on JSON schema)
            # Only store these fields plus timestamps and resume_id
            fields_to_keep = [
                'summary', 'total_experience', 'skills', 'positions', 'companies', 
                'education', 'certifications', 'achievements', 'industries', 'projects'
            ]
            
            # Prepare items
            items = []
            resume_ids = []
            timestamp = datetime.now().isoformat()
            
            for batch_item in resume_batch:
                resume_data = batch_item['resume_data']
                resume_id = batch_item.get('resume_id')
                
                # Generate resume_id if not provided
                if not resume_id:
                    resume_id = str(uuid.uuid4())
                
                resume_ids.append(resume_id)
                
                # Create filtered data with only the fields we want to keep
                filtered_data = {}
                for field in fields_to_keep:
                    if field in resume_data:
                        filtered_data[field] = resume_data[field]
                
                # Prepare item for DynamoDB
                item = {
                    'resume_id': resume_id,
                    'created_at': timestamp,
                    'updated_at': timestamp
                }
                
                # Process the data to ensure it's compatible with DynamoDB
                processed_data = self._process_data_for_dynamodb(filtered_data)
                
                # Add processed data to item
                item.update(processed_data)
                
                items.append(item)
            
            logger.info(f"Batch storing {len(items)} items in DynamoDB with fields: {', '.join(fields_to_keep)}")
            
            # DynamoDB batch write has a limit of 25 items per request
            batch_size = 25
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                
                # Prepare batch write request
                request_items = {
                    self.table_name: [
                        {'PutRequest': {'Item': item}} for item in batch_items
                    ]
                }
                
                # Batch write with retries
                max_retries = 3
                retry_delay = 1  # seconds
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Batch writing {len(batch_items)} items to DynamoDB (attempt {attempt+1}/{max_retries})")
                        response = self.dynamodb.batch_write_item(RequestItems=request_items)
                        
                        # Check for unprocessed items
                        unprocessed = response.get('UnprocessedItems', {}).get(self.table_name, [])
                        if unprocessed:
                            logger.warning(f"{len(unprocessed)} items were not processed in the batch")
                            
                            # Retry unprocessed items
                            request_items = {
                                self.table_name: unprocessed
                            }
                            continue
                        
                        break
                    except Exception as e:
                        logger.error(f"Error batch writing to DynamoDB (attempt {attempt+1}/{max_retries}): {str(e)}")
                        if attempt < max_retries - 1:
                            import time
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            logger.error(f"Failed to batch write to DynamoDB after {max_retries} attempts")
                            raise
            
            logger.info(f"Successfully batch stored {len(items)} items in DynamoDB")
            return resume_ids
            
        except Exception as e:
            logger.error(f"Error batch storing resume data in DynamoDB: {str(e)}")
            raise
    
    def _process_data_for_dynamodb(self, data: Any) -> Any:
        """
        Process data to ensure compatibility with DynamoDB
        - Handles None values
        - Converts Python objects to DynamoDB-compatible types
        - Processes nested lists and dictionaries
        
        Args:
            data: Data to process (can be any type)
            
        Returns:
            DynamoDB-compatible data
        """
        import decimal
        from decimal import Decimal
        
        # Handle None
        if data is None:
            return "Not provided"
            
        # Handle dictionaries (recursively process values)
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # Skip null keys
                if key is None:
                    continue
                # Process value recursively
                processed_value = self._process_data_for_dynamodb(value)
                result[key] = processed_value
            return result
            
        # Handle lists (recursively process items)
        if isinstance(data, list):
            # DynamoDB doesn't accept empty lists
            if not data:
                return ["Not provided"]
            return [self._process_data_for_dynamodb(item) for item in data]
            
        # Handle sets by converting to lists
        if isinstance(data, set):
            # DynamoDB doesn't accept empty lists
            if not data:
                return ["Not provided"]
            return [self._process_data_for_dynamodb(item) for item in data]
        
        # Convert floats to Decimal
        if isinstance(data, float):
            # Use Decimal for float values as DynamoDB doesn't support floats
            try:
                return Decimal(str(data))
            except (decimal.InvalidOperation, TypeError):
                # If conversion fails, convert to string
                return str(data)
            
        # Handle numbers (integers)
        if isinstance(data, int):
            return data
            
        # Handle booleans
        if isinstance(data, bool):
            return data
            
        # Handle strings
        if isinstance(data, str):
            return data if data else "Not provided"
            
        # Handle other types by converting to string
        return str(data)
    
    def get_resume(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """
        Get resume data from DynamoDB by ID
        
        Args:
            resume_id: Resume ID (UUID)
            
        Returns:
            Resume data dictionary or None if not found
        """
        if not ENABLE_DYNAMODB:
            return None
            
        try:
            response = self.table.get_item(
                Key={
                    'resume_id': resume_id
                }
            )
            
            if 'Item' in response:
                return response['Item']
            else:
                logger.warning(f"Resume with ID {resume_id} not found in DynamoDB")
                return None
                
        except Exception as e:
            logger.error(f"Error getting resume data from DynamoDB: {str(e)}")
            raise
    
    def update_resume(self, resume_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing resume in DynamoDB
        
        Args:
            resume_id: Resume ID (UUID)
            update_data: Data to update
            
        Returns:
            True if successful, False otherwise
        """
        if not ENABLE_DYNAMODB:
            return False
            
        try:
            # Add updated timestamp
            timestamp = datetime.now().isoformat()
            update_data['updated_at'] = timestamp
            
            # Build update expression and attribute values
            update_expression = "SET updated_at = :updated_at"
            expression_attr_values = {
                ':updated_at': timestamp
            }
            
            # Add each field to update expression
            for key, value in update_data.items():
                if key != 'resume_id' and key != 'updated_at':
                    update_expression += f", {key} = :{key}"
                    expression_attr_values[f':{key}'] = value if value is not None else "Not provided"
            
            # Update item in DynamoDB
            response = self.table.update_item(
                Key={
                    'resume_id': resume_id
                },
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attr_values,
                ReturnValues="UPDATED_NEW"
            )
            
            logger.info(f"Updated resume in DynamoDB with ID: {resume_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating resume in DynamoDB: {str(e)}")
            return False 