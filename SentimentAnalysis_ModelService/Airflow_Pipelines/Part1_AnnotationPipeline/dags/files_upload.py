import boto3
#from configparser import ConfigParser
import os
import glob
#config = ConfigParser()
import yaml


def load_yaml(yaml_path):
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print("Failed to load a yaml file due to {e}")
    return config

def files_upload_edgar():
	yaml_path='/Users/prathyusha/Desktop/pranathi/Assignment2/airflow_edgar/dags/config.yaml'
	config= load_yaml(yaml_path)
	bucket = config["dev"]["bucket"]
	filepath = config["dev"]['filepath']
	 #upload to s3

	arr = os.listdir(filepath)
	s3 = boto3.resource('s3')

	for file in arr:
		s3.Bucket(bucket).upload_file('{}/{}'.format(filepath,str(file)),'{}/{}'.format('files',str(file)))

