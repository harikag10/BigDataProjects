import urllib.request, json 
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
# Import Custom Modules
import boto3
import pandas as pd
import numpy as np
import json
import re
import sys
import yaml
import http.client, urllib.request, urllib.parse, urllib.error, base64
from smart_open import smart_open
import requests
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#from flask import jsonify

yaml_path='/Users/prathyusha/Desktop/pranathi/CSYE7245-Spring2021-Labs/transcript-simulated-api/airflow/dags/config.yaml'

def load_yaml(yaml_path):
    try:
        with open(yaml_path, "r") as f:
            configs = yaml.safe_load(f)
    except Exception as e:
        print("Failed to load a yaml file due to {e}")
    return configs

config= load_yaml(yaml_path)

def getlist():
	aws_access_key_id = config["dev"]["aws_access_key_id"]
	aws_secret_access_key = config["dev"]["aws_secret_access_key"]
	bucket_name = config["dev"]["bucket"]
	object_key = config["dev"]["fileinput"]["object_key"]
	prefix = config["dev"]["fileinput"]["prefix"]
	path = 's3://{}:{}@{}/{}/{}'.format(aws_access_key_id, aws_secret_access_key, bucket_name, prefix, object_key)
	df = pd.read_csv(smart_open(path))
	print(df)
	df.to_csv("./companies_list.csv",index=False)


def download():
	df=pd.read_csv("./companies_list.csv")
	for company in list(df['company']):
		with urllib.request.urlopen("http://127.0.0.1:8000/call-transcripts/{}/2021".format(company)) as url:
			data = json.loads(url.read().decode())
			#print(data)
		parent_dir = config["dev"]["downloadpath"]
		directory='edgar_download' 
		localpath = os.path.join(parent_dir, directory) 
		print(localpath)
		if not os.path.exists(localpath):
			print("inside if")
			os.mkdir(localpath) 
		
		with open('{}/{}'.format(localpath,data["company"]), 'w') as f:
			f.write(data['transcript'])
		f.close()



def preprocess():

  #DIR = "/Users/prathyusha/Desktop/pranathi/Assignment2/airflow_edgar/edgar_download"

  DIR= config["dev"]["fetchpath"]
  final_df=pd.DataFrame({'Text':[],'company':[]})
  final_df=final_df[1:10]
# if you want to list all the contents in DIR
  entries = [entry for entry in os.listdir(DIR) if '.' not in entry]
  print(entries)
  #if entries in ('.DS_Store')
  for i in entries:
      #company=i
      #print(i)
      index_value=entries.index(i)
      #print(os.path.join(DIR,'{}'.format(i)))
      string = open(os.path.join(DIR,'{}'.format(i)),encoding='utf-8').read()
      n=string.replace('\n\nCompany Representatives\n\n', '\n\nCompany Participants\n\n')
      n=n.replace('–', '-')
      n = n[n.find('Company Participants')+len('Company Participants'):]

      a = []
      start=''
      for line in n.split("\n"):
          if line != '' and '-' in line:
              a.append(line[0:line.find('-')])
          elif line != '' and '-' not in line and 'Conference Call Participants' not in line:
              start=line
              break
      if 'Conference Call Participant' in a:
          a.remove('Conference Call Participant')
      a=[re.sub('[^a-zA-Z0-9\s\n\.]', '', _) for _ in a]
      a=[i.strip() for i in a]
      a.append('Operator')
      a.append('Unidentified Analyst')
      split_list=string.split('\n\n')
      #print(company)
    #print(start)
      if start!='Operator':
          splitlist_new=split_list[split_list.index('{}'.format(start)):]
      else:
          splitlist_new=split_list[split_list.index('Operator'):]
      mainlist=[]
      company=[]

      for j in splitlist_new:
          if j not in a:
              mainlist.append(j)
              company.append(i)
      new_df=pd.DataFrame({'Text':mainlist})

      copy_df=new_df.copy()
      copy_df['company']=company
      def clean(row):
          return re.sub('[^a-zA-Z0-9\s\n\.]', '', row['Text'])
      copy_df['Text']=copy_df.apply(lambda row: clean(row),axis=1)
      copy_df=copy_df.dropna()
      copy_df['Text']=copy_df['Text'].str.strip()
      copy_df=copy_df[copy_df['Text']!='']
      
      final_df=pd.concat([final_df,copy_df])
  aws_access_key_id = config["dev"]["aws_access_key_id"]
  aws_secret_access_key = config["dev"]["aws_secret_access_key"]
  bucket_name = config["dev"]["bucket"]
  object_key = config["dev"]["stage"]["object_key"]
  prefix = config["dev"]["stage"]["prefix"]
  downloadpath=config["dev"]["downloadpath"]
  path = 's3://{}/{}/{}'.format(bucket_name, prefix, object_key)
  filename="CompaniesCallData.csv"

  final_df.to_csv(os.path.join(downloadpath,filename),index=False)
  s3 = boto3.resource('s3')
  s3.Bucket(bucket_name).upload_file('{}/{}'.format(downloadpath,filename),'{}/{}'.format(prefix,str(object_key)))
  #final_df.reset_index(inplace=True)
  #final_df.to_json("./companiesdata.json",orient = 'records', compression = 'infer', index = 'true')
  
  #temp=final_df.to_json(orient = 'records', compression = 'infer', index = 'true')
  #print(temp)

def prediction():

  aws_access_key_id = config["dev"]["aws_access_key_id"]
  aws_secret_access_key = config["dev"]["aws_secret_access_key"]
  bucket_name = config["dev"]["bucket"]
  object_key = config["dev"]["stage"]["object_key"]
  prefix = config["dev"]["stage"]["prefix"]
  path = 's3://{}:{}@{}/{}/{}'.format(aws_access_key_id, aws_secret_access_key, bucket_name, prefix, object_key)
  df = pd.read_csv(smart_open(path))
  df=df.dropna()
  listtext=list(df["Text"])
  def chunks(lst,n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

  final_df=pd.DataFrame({"Text":[],"metric":[]})


  for lis in chunks(listtext,200):
    r = requests.post(config["dev"]["FlaskUrl"],json={"data":lis})
    a = json.loads(r.text)
    copy_df=final_df[final_df["Text"]=='~~~']
    copy_df["Text"]=lis
    copy_df["metric"]=a["pred"]
    final_df=pd.concat([final_df,copy_df])

  final_df['Sentiment'] = np.where(final_df.eval("metric > 0"), "Positive", "Negative")
  sentiment_df=final_df[["Text","Sentiment"]]
  filename="Sentiments.csv"
  sentiment_df.to_csv(os.path.join(config["dev"]["downloadpath"],filename),index=False)


def upload():
  filepath=config["dev"]["downloadpath"]
  file="Sentiments.csv"
  aws_access_key_id = config["dev"]["aws_access_key_id"]
  aws_secret_access_key = config["dev"]["aws_secret_access_key"]
  bucket_name = config["dev"]["bucket"]
  object_key = config["dev"]["final"]["sentiment_key"]
  prefix = config["dev"]["final"]["prefix"]
  s3 = boto3.resource('s3')
  s3.Bucket(bucket_name).upload_file('{}/{}'.format(filepath,file),'{}/{}'.format(prefix,str(file)))

  
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(0),
    'concurrency': 1,
    'retries': 0,
    'depends_on_past': False,
}

with DAG('EDGAR-Inference-Pipeline',
         catchup=False,
         default_args=default_args,
         schedule_interval='@once',
         ) as dag:
    t0_getList = PythonOperator(task_id='FetchDataFromAPI',
                              python_callable=getlist)
    t1_download = PythonOperator(task_id='DownloadDataFromCloud',
                                python_callable=download)
    t2_preprocess = PythonOperator(task_id='preprocess',
                                python_callable=preprocess)
    t3_prediction = PythonOperator(task_id='Prediction',
                                python_callable=prediction)
    t4_upload= PythonOperator(task_id='upload',
                                python_callable=upload)

t0_getList >> t1_download >> t2_preprocess >> t3_prediction >> t4_upload




