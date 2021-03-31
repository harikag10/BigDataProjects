# Imports the modules
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

# Import Custom Modules
import boto3
import pandas as pd
import numpy as np
import http.client, urllib.request, urllib.parse, urllib.error, base64
import json
import re
import sys
import yaml
#sys.path.insert(0,os.path.abspath(os.path.dirname('/Users/prathyusha/Desktop/pranathi/Assignment2/airflow_edgar/s3_uploader/files_upload.py')))
#from s3_uploader import files_upload
from  files_upload import files_upload_edgar
from files_upload import load_yaml


yaml_path='/Users/prathyusha/Desktop/pranathi/Assignment2/airflow_edgar/dags/config.yaml'
config= load_yaml(yaml_path)


def s3_upload():
  #sys.path.insert(0,os.path.abspath(os.path.dirname('/Users/prathyusha/Desktop/pranathi/Assignment2/airflow_edgar/s3_uploader/files_upload.py')))
  execute=files_upload_edgar()
  print('executed')

def download():
  #initiate s3 resource
  s3 = boto3.resource('s3')


  # select bucket
  #my_bucket = s3.Bucket('edgarfile-storage')
  my_bucket = s3.Bucket(config["dev"]["bucket"])

  #parent_dir = "/Users/prathyusha/Desktop/pranathi/Assignment2/airflow_edgar"
  parent_dir = config["dev"]["downloadpath"]
  directory='edgar_download'
  # Path 
  localpath = os.path.join(parent_dir, directory) 
  #print(localpath)

  if not os.path.isdir(localpath):
    #print("inside if")
    os.mkdir(localpath) 

  # download file into current directory
  for s3_object in my_bucket.objects.all():
      #print(s3_object.key)
      # Need to split s3_object.key into path and file name, else it will give error file not found.
      path, filename = os.path.split(s3_object.key)
      
      if filename in ('.DS_Store','','annotation.csv'):
          #print(filename)
          continue
      elif path in ('files'):
        my_bucket.download_file(s3_object.key, os.path.join(localpath,filename) )


def preprocess():

  #DIR = "/Users/prathyusha/Desktop/pranathi/Assignment2/airflow_edgar/edgar_download"
  DIR= config["dev"]["fetchpath"]
  final_df=pd.DataFrame({'Text':[],'company':[]})

# if you want to list all the contents in DIR
  entries = [entry for entry in os.listdir(DIR)]
  #print(entries)
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
  final_df.to_csv('/tmp/final_df.csv',index=False)

def sentiment():
  final_df=pd.read_csv('/tmp/final_df.csv')
  #final_df=final_df[1:10]
  headers = {
      # Request headers
      'Content-Type': 'application/json',
      'Ocp-Apim-Subscription-Key': config["dev"]["APIKey"],
  }

  params = urllib.parse.urlencode({
      # Request parameters
      'showStats': '{boolean}',
  })
  def positivity(row):
        api_body={
          "documents": [
            {
              "language": "en",
              "id": "1",
              "text": row['Text']
            }
          ]
        }
        
        conn.request("POST", "/text/analytics/v2.1/sentiment?%s" % params, """{}""".format(api_body) , headers)
        response = conn.getresponse()
        data = response.read()
        #data_decoded=data.decode("utf-8") 
        data_decoded=json.loads(data)
        #print(data_decoded) 
        
        row['metric']=data_decoded['documents'][0]['score']

        return row
  conn = http.client.HTTPSConnection('eastus.api.cognitive.microsoft.com')
  copy_df=final_df.apply(lambda row: positivity(row),axis=1)
  copy_df.to_csv('/tmp/copy_df.csv',index=False)
  conn.close()


def normalize():
  copy_df=pd.read_csv('/tmp/copy_df.csv')
  DIR=config["dev"]["downloadpath"]
  #DIR = "/Users/prathyusha/Desktop/pranathi/Assignment2/airflow_edgar"
  filename='annotation.csv'
  df1=copy_df.copy()
  a=np.array(df1['metric'])
  d = 2.*(a - np.min(a))/np.ptp(a)-1
  df1['scaled_metric']=d
  pd.set_option('display.float_format', '{:.6f}'.format)
  df2=df1.copy()
  df2['sentiment']=np.where(df1['scaled_metric']>0, 'Positive', 'Negative')
  df2.to_csv(os.path.join(DIR,filename),index=False) 


def files_upload():
  #config.read('config.ini')
  #bucket = config.get('main1', 'bucket')
  #filepath = config.get('main1', 'filepath')
  #bucket='edgarfile-storage'
  bucket=config["dev"]["bucket"]
  #filepath = '/Users/prathyusha/Desktop/pranathi/Assignment2/airflow_edgar'
  filepath = config["dev"]["downloadpath"]
  file='annotation.csv'
  s3 = boto3.resource('s3')
  s3.Bucket(bucket).upload_file('{}/{}'.format(filepath,file),'{}/{}'.format('annotation',str(file)))
  print('annotation file uploaded')


default_args = {
    'owner': 'airflow',
    'start_date': days_ago(0),
    'concurrency': 1,
    'retries': 0,
    'depends_on_past': False,
}

with DAG('EDGAR-Annotation-Pipeline',
         catchup=False,
         default_args=default_args,
         schedule_interval='@once',
         ) as dag:
    t0_start = PythonOperator(task_id='PushDataToCloud',
                              python_callable=s3_upload)
    t1_getdata = PythonOperator(task_id='DownloadDataFromCloud',
                                python_callable=download)
    t2_preprocess = PythonOperator(task_id='preprocess',
                                python_callable=preprocess)
    t3_annotation = PythonOperator(task_id='Annotation',
                                python_callable=sentiment)
    t4_normalize = PythonOperator(task_id='Normalize',
                                python_callable=normalize)
    t5_files_upload = PythonOperator(task_id='UploadLabledData',
                                python_callable=files_upload)

t0_start >> t1_getdata >> t2_preprocess >> t3_annotation >> t4_normalize >> t5_files_upload
#t1_getdata >> t2_preprocess >> t3_annotation >> t4_normalize >> t5_files_upload
