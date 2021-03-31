# -*- coding: utf-8 -*-
"""GCP_DATAFLOW_SEVIR_STORM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_-emD4AcT5JjDWyCZ-m4yrmVPQzEiF34
"""

#Commented out IPython magic to ensure Python compatibility.
!pip install boto3
pip install fsspec
pip install s3fs
pip install gcsfs
# %%bash
pip install --upgrade tensorflow==1.13.1
pip install --ignore-installed --upgrade pytz==2018.4
pip uninstall -y google-cloud-dataflow
pip install --upgrade apache-beam[gcp]==2.6
!pip install google-cloud-core==1.4.1
!pip install --upgrade apache-beam[gcp]==2.23.0
!pip install six==1.12.0

from google.cloud import storage
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/content/bigdata-7245-ae3b0618cd6c.json"
storage_client = storage.Client()

import apache_beam as beam
import apache_beam as beam
from apache_beam import pipeline
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam import runners
#from apache_beam.runners.interactive import interactive_environment
#from apache_beam import pipeline
#from apache_beam.runners import runner
import datetime


#Class to handle Storm related data

class Storm(beam.DoFn):



  def process(self, element):
    import os
    os.system('pip install boto3')
    os.system('pip install s3fs')
    os.system('pip install gcsfs')
    os.system('pip install --upgrade google-cloud-storage')
    os.system('pip install pyOpenSSL')
    os.system('pip install pandas-gbq')
    ##yield word
    import boto3
    import pandas as pd
    import gcsfs
    import numpy as np
    import s3fs
    import h5py
    from google.cloud import storage
    import gzip
    import OpenSSL
    from OpenSSL import SSL
    import pandas_gbq
    storage_client = storage.Client()


    Details_PATH='gs://bigdata-7245/storm_raw/StormEvents_details-ftp_v1.0_d2019_c20210223.csv.gz'
    Fatalities_PATH='gs://bigdata-7245/storm_raw/StormEvents_fatalities-ftp_v1.0_d2019_c20210223.csv.gz'
    Locations_PATH='gs://bigdata-7245/storm_raw/StormEvents_locations-ftp_v1.0_d2019_c20210223.csv.gz'

    # Read and Transform
    # Storm Details
    details = pd.read_csv(Details_PATH, compression='gzip', header=0, sep=',', quotechar='"')
    details_new=details.loc[(details['BEGIN_YEARMONTH']==201909) & (details['BEGIN_DAY']>8) & (details['BEGIN_DAY']<24)].copy()
    details_new.reset_index(drop=True,inplace=True)
    details_new['eventid_new']='S'+details_new['EVENT_ID'].astype('str')

    details_new.DAMAGE_PROPERTY = (details_new.DAMAGE_PROPERTY.replace(r'[KM]+$', '', regex=True).astype(float) * \
              details_new.DAMAGE_PROPERTY.str.extract(r'[\d\.]+([KM]+)', expand=False)
                .fillna(1)
               .replace(['K','M'], [10**3, 10**6]).astype(int))



    details_new.DAMAGE_CROPS = (details_new.DAMAGE_CROPS.replace(r'[KM]+$', '', regex=True).astype(float) * \
              details_new.DAMAGE_CROPS.str.extract(r'[\d\.]+([KM]+)', expand=False)
                .fillna(1)
               .replace(['K','M'], [10**3, 10**6]).astype(int))

    details_filename ='sevir/storm_data/details.csv'
    bucket_name = 'bigdata-7245'
    details_new.to_csv('gs://{}/{}'.format(bucket_name,details_filename),index=False)
    pandas_gbq.to_gbq(details_new,'Storm.storm_details',if_exists='replace',project_id='bigdata-7245')


    # Fatalities
    fatalities= pd.read_csv(Fatalities_PATH, compression='gzip', header=0, sep=',', quotechar='"')
    fatalities_new=fatalities.loc[(fatalities['FAT_YEARMONTH']==201909) & (fatalities['FAT_DAY']>8) & (fatalities['FAT_DAY']<24)]
    fatalities_new.reset_index(drop=True, inplace=True)
    pandas_gbq.to_gbq(fatalities_new,'Storm.storm_fatalities',if_exists='replace',project_id='bigdata-7245')

    fatalities_filename ='sevir/storm_data/fatalities.csv'
    fatalities_new.to_csv('gs://{}/{}'.format(bucket_name,fatalities_filename),index=False)

    # Locations
    locations = pd.read_csv(Locations_PATH, compression='gzip', header=0, sep=',', quotechar='"')
    # Filter Locations
    events_list=list(details_new['EVENT_ID'])
    locations_new=locations[locations['EVENT_ID'].isin(events_list)]
    locations_new.reset_index(drop=True,inplace=True)

    locations_filename ='sevir/storm_data/locations.csv'
    locations_new.to_csv('gs://{}/{}'.format(bucket_name,locations_filename),index=False)
    pandas_gbq.to_gbq(locations_new,'Storm.storm_locations',if_exists='replace',project_id='bigdata-7245')



# Class to handle sampling of sevir catalog data

class Sample(beam.DoFn):

  def process(self, element):
    import os
    os.system('pip install boto3')
    os.system('pip install s3fs')
    os.system('pip install gcsfs')
    os.system('pip install --upgrade google-cloud-storage')
    os.system('pip install pyOpenSSL')
    os.system('pip install pandas-gbq')
    ##yield word
    import boto3
    import pandas as pd
    import gcsfs
    import numpy as np
    import s3fs
    import h5py
    from google.cloud import storage
    import gzip
    import OpenSSL
    from OpenSSL import SSL
    import pandas_gbq
    storage_client = storage.Client()
    # import os
    #os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/content/sapient-flare-303301-72808613b8d5.json"
    

    s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-2')

    CATALOG_PATH='s3://sevir/CATALOG.csv'

    # Read catalog
    catalog = pd.read_csv(CATALOG_PATH,parse_dates=['time_utc'],low_memory=False)
    fitered_catalog=catalog[(catalog['time_utc']>='2019-09-09') & (catalog['time_utc']<='2019-09-22')]

    # Desired image types
    img_types = set(['vis','ir069','ir107','vil','lght'])

    # Group by event id, and filter to only events that have all desired img_types
    events = fitered_catalog.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id')
    event_ids = list(events.groups.keys())
    print('Found %d events matching' % len(event_ids),img_types)

    filtering_df=events.apply(lambda x: x)
    filtering_df=filtering_df.sort_values('img_type')

    filename ='sevir/catalog_data/filtered_catalog.csv'
    bucket_name = 'bigdata-7245'

    filtering_df.to_csv('gs://{}/{}'.format(bucket_name,filename),index=False)
    pandas_gbq.to_gbq(filtering_df,'sevir.sevir_catalog',if_exists='replace',project_id='bigdata-7245')

#Dataflow Pipeline

def pipeline_exec():
  job_name = 'preprocess-catalog' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')
  import os
  OUTPUT_DIR = 'gs://{0}/dataflow/preproc/'.format('bigdata-7245')
  options = {
  'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
  'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
  'job_name': job_name,
  'project': 'bigdata-7245',
  'teardown_policy': 'TEARDOWN_ALWAYS',
  'max_num_workers': 3, # CHANGE THIS IF YOU HAVE MORE QUOTA
  'no_save_main_session': True,
  'region':'us-central1'
  }
  #opts = beam.pipeline.PipelineOptions(flags=[], **options)
  opts = PipelineOptions(flags=[], **options)
  # if in_test_mode:
  # RUNNER = 'DirectRunner'
  # else:
  RUNNER = 'DataflowRunner'
  p = beam.Pipeline(RUNNER, options=opts)
  classes = {
      'storm': Storm,
      'sample': Sample
  }
  j='Storm'
  for i in [Storm,Sample]:
      
    
    (p
    
    | 'Pipeline Start {}'.format(j) >> beam.Create([
    'Start'
    ])
    #| 'Storm' >> beam.ParDo(Storm())
    | '{}'.format(j) >> beam.ParDo(i())
    
    )
    j='Catalog'

  job = p.run()

pipeline_exec()
