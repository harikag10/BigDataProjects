#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import s3fs
import pandas as pd

#Reading Data from CSV Files
storm_details=pd.read_csv('s3://prudhvis7245/storm/df_details_new.csv')
storm_fatalaties = pd.read_csv('s3://prudhvis7245/storm/df_fatalities_new.csv')
storm_locations = pd.read_csv('s3://prudhvis7245/storm/df_locations_new.csv')
sevir = pd.read_csv('s3://prudhvis7245/sevir/catalog_data/filtered_catalog.csv')

#Joining Data
sevir_storm_details=pd.merge(sevir,storm_details,how='left',left_on='id',right_on='eventid_new')
storm_fatalities_details=pd.merge(storm_fatalaties,storm_details,how='left',left_on='EVENT_ID',right_on='EVENT_ID')
storm_locations_details=pd.merge(storm_locations,storm_details,how='left',left_on='EVENT_ID',right_on='EVENT_ID')

#Loading Back to S3
bucket_name='prudhvis7245'

sevir_storm_details.to_csv('s3://{}/{}'.format(bucket_name,'storm_sevir_joined/sevir_storm_details'),index=False)
storm_fatalities_details.to_csv('s3://{}/{}'.format(bucket_name,'storm_fatalities_details/storm_fatalities_details'),index=False)
storm_locations_details.to_csv('s3://{}/{}'.format(bucket_name,'storm_sevir_joined/storm_locations_details'),index=False)
print('Files Loaded Successfully')

