#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import pandas as pd
import boto3

#Fetching .gz files from local S3

StormEvents_details='s3://prudhvis7245/storm_all/StormEvents_details-ftp_v1.0_d2019_c20210223.csv.gz'
StormEvents_fatalities='s3://prudhvis7245/storm_all/StormEvents_fatalities-ftp_v1.0_d2019_c20210223.csv.gz'
StormEvents_locations='s3://prudhvis7245/storm_all/StormEvents_locations-ftp_v1.0_d2019_c20210223.csv.gz'
print("Access to files ovtained")

df_details = pd.read_csv(StormEvents_details, compression='gzip', header=0, sep=',', quotechar='"')
df_fatalities= pd.read_csv(StormEvents_fatalities, compression='gzip', header=0, sep=',', quotechar='"')
df_locations=pd.read_csv(StormEvents_locations, compression='gzip', header=0, sep=',', quotechar='"')


#Filtering Data
df_details_new=df_details.loc[(df_details['BEGIN_YEARMONTH']==201909) & (df_details['BEGIN_DAY']>8) & (df_details['BEGIN_DAY']<24)].copy()
df_fatalities_new=df_fatalities.loc[(df_fatalities['FAT_YEARMONTH']==201909) & (df_fatalities['FAT_DAY']>8) & (df_fatalities['FAT_DAY']<24)]
df_details_new.reset_index(drop=True,inplace=True)
df_fatalities_new.reset_index(drop=True,inplace=True)


df_details_new['eventid_new']='S'+df_details_new['EVENT_ID'].astype('str')

events_list=list(df_details_new['EVENT_ID'])
df_locations_new=df_locations[df_locations['EVENT_ID'].isin(events_list)]

df_locations_new.reset_index(drop=True,inplace=True)


from io import StringIO # python3; python2: BytesIO 

#Data Loading as CSV Files

bucket = 'prudhvis7245' # already created on S3
csv_buffer = StringIO()
df_details_new.to_csv(csv_buffer,index=0)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, 'storm/df_details_new.csv').put(Body=csv_buffer.getvalue())


csv_buffer = StringIO()
df_fatalities_new.to_csv(csv_buffer,index=0)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, 'storm/df_fatalities_new.csv').put(Body=csv_buffer.getvalue())


csv_buffer = StringIO()
df_locations_new.to_csv(csv_buffer,index=0)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, 'storm/df_locations_new.csv').put(Body=csv_buffer.getvalue())

print("Success")


