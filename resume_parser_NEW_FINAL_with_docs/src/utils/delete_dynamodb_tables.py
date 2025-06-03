#!/usr/bin/env python3
"""
Script to delete the DynamoDB table named by DYNAMODB_TABLE_NAME in your .env.
It loads AWS and DynamoDB parameters via python-dotenv, then deletes and waits
for the table to be fully removed.
"""

import os
import sys
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

# Load environment variables from .env in working directory
load_dotenv()  # uses find_dotenv() and load_dotenv() by default :contentReference[oaicite:6]{index=6}

# Ensure deletion is explicitly enabled
if os.getenv("ENABLE_DYNAMODB", "").lower() != "true":
    print("DynamoDB deletion disabled (ENABLE_DYNAMODB != true). Exiting.")
    sys.exit(0)

# Read essential configuration
TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME")
REGION     = os.getenv("DYNAMODB_REGION")
ENDPOINT   = os.getenv("DYNAMODB_ENDPOINT")

# Initialize DynamoDB client
dynamodb = boto3.client(
    "dynamodb",
    region_name=REGION,
    endpoint_url=ENDPOINT
)

def delete_resume_table():
    """
    Attempt to delete the table, handle common exceptions,
    and wait until deletion completes.
    """
    try:
        print(f"Initiating deletion of table '{TABLE_NAME}'...")
        dynamodb.delete_table(TableName=TABLE_NAME)  # starts DELETING state :contentReference[oaicite:7]{index=7}
    except dynamodb.exceptions.ResourceNotFoundException:
        print(f"Table '{TABLE_NAME}' does not exist; nothing to delete.")  # ResourceNotFoundException :contentReference[oaicite:8]{index=8}
        return
    except ClientError as err:
        # Handle cases like table in CREATING/UPDATING state → ResourceInUseException
        print(f"Failed to delete table: {err.response['Error']['Message']}")
        sys.exit(1)

    # Wait until table no longer exists
    waiter = dynamodb.get_waiter("table_not_exists")  # waiter name from boto3 docs :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}
    print(f"Waiting for table '{TABLE_NAME}' to be fully deleted...")
    try:
        waiter.wait(TableName=TABLE_NAME)
        print(f"Table '{TABLE_NAME}' has been deleted successfully.")
    except ClientError as err:
        print(f"Error while waiting for deletion: {err}")
        sys.exit(1)

if __name__ == "__main__":
    delete_resume_table()
