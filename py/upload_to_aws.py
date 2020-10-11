import boto3
from botocore.exceptions import NoCredentialsError
import json


## AWS Authentication Settings
with open("/home/pi/Projects/Check-Your-Tone/amazon_tokens.json") as f:
    keys = json.load(f)

AWS_ACCESS_KEY = keys["ACCESS"]
AWS_SECRET_ACCESS_KEY = keys["ACCESS_SECRET"]


## Function to upload files to AWS bucket using Boto3 client
def upload_to_aws(file_path, bucket_name, s3_file_name):
    
    s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    
    try:
        s3.upload_file(file_path, bucket_name, s3_file_name)
        print("Upload complete.\n")
        return True
    
    except FileNotFoundError:
        print("No file was found.")
        return False
    
    except NoCredentialsError:
        print("Error establishing credentials with AWS.")
        return False


