import boto3
import os
from loadyaml import load_yaml

#Function to download trained model from s3
def s3_download():
    config = load_yaml('config.yaml')
    aws_access_key_id = config["dev"]["aws_access_key_id"]
    aws_secret_access_key = config["dev"]["aws_secret_access_key"]
    bucket_name = config["dev"]["bucket"]
    prefix = config["dev"]["s3_download"]["prefix"]

    s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-2',
    aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    # select bucket
    local_dir = '.'


    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=prefix):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, prefix))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)