#!/usr/bin/env python3
import os
import sys
import boto3
from botocore.exceptions import ClientError

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import configuration from the central config file
try:
    from config.config import (
        DYNAMODB_TABLE_NAME, 
        DYNAMODB_REGION, 
        DYNAMODB_ENDPOINT, 
        ENABLE_DYNAMODB, 
        DYNAMODB_READ_CAPACITY_UNITS, 
        DYNAMODB_WRITE_CAPACITY_UNITS
    )
except ImportError as e:
    print(f"Error importing configuration: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Only proceed if explicitly enabled
if not ENABLE_DYNAMODB:
    print("DynamoDB table creation is disabled (ENABLE_DYNAMODB != true). Exiting.")
    sys.exit(0)

# Use the configuration values
TABLE_NAME = DYNAMODB_TABLE_NAME
REGION_NAME = DYNAMODB_REGION
ENDPOINT_URL = DYNAMODB_ENDPOINT
RCU = DYNAMODB_READ_CAPACITY_UNITS
WCU = DYNAMODB_WRITE_CAPACITY_UNITS

print(f"Using DynamoDB table name: {TABLE_NAME}")

# Initialize client
dynamodb = boto3.client(
    "dynamodb",
    region_name=REGION_NAME,
    endpoint_url=ENDPOINT_URL
)

def create_resume_table():
    try:
        # Check if table already exists
        dynamodb.describe_table(TableName=TABLE_NAME)
        print(f"Table '{TABLE_NAME}' already exists – skipping creation.")
    except dynamodb.exceptions.ResourceNotFoundException:
        print(f"Creating table '{TABLE_NAME}'...")
        resp = dynamodb.create_table(
            TableName=TABLE_NAME,
            AttributeDefinitions=[
                # Define only the key attributes here
                {"AttributeName": "resume_id", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "resume_id", "KeyType": "HASH"},
            ],
            BillingMode="PROVISIONED",
            ProvisionedThroughput={
                "ReadCapacityUnits": RCU,
                "WriteCapacityUnits": WCU
            }
        )
        # Wait until the table is active
        waiter = dynamodb.get_waiter("table_exists")
        waiter.wait(TableName=TABLE_NAME)
        print(f"Table '{TABLE_NAME}' is now ACTIVE.")

if __name__ == "__main__":
    try:
        create_resume_table()
    except ClientError as e:
        print("Error interacting with DynamoDB:", e)
        sys.exit(1)
