#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import boto3
import pandas as pd 
import os
import numpy as np
import s3fs
import h5py

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

#Target Filename and bucket
filename ='sevir/catalog_data/filtered_catalog.csv'

bucket_name='prudhvis7245'


filtering_df.to_csv('s3://{}/{}'.format(bucket_name,filename),index=False)

s3 = s3fs.S3FileSystem()

for filename in list(set(filtering_df['file_name'])):
    types= filtering_df[filtering_df['file_name']==filename]['img_type'].iloc[0]
    print(types)
    events_list=list(filtering_df[filtering_df['file_name']==filename]['id']) 
    index_list= list(filtering_df[(filtering_df['id'].isin(events_list)) & (filtering_df['file_name']==filename)]['file_index'])
    print("before with")
    with s3.open('s3://sevir/data/{}'.format(filename), 'rb') as s3file:
        print('after with')
        with h5py.File(s3file, 'r') as hf:
            
            if types=='lght':

                        f1 = h5py.File('/tmp/tmpfile.h5', "w")

                        a=list(hf.keys())
                        for i in range(0,len(a)):

                                if a[i] in events_list:
                                    f1.create_dataset(a[i],hf["{}".format(a[i])].shape ,'<f4', hf["{}".format(a[i])][:])
                        s3 = boto3.resource(
                                service_name='s3',
                                region_name='us-east-2')
                        s3.Bucket(bucket_name).upload_file(Filename='/tmp/tmpfile.h5', Key= 'sevir/' + filename)
                        f1.close()
                        os.remove('/tmp/tmpfile.h5')
                
            else:    
                ids=np.array([hf['id'][i] for i in index_list])
                images=np.array([hf['{}'.format(types)][i] for i in index_list])

                f1 = h5py.File('/tmp/tmpfile.h5', "w")

                dset1 = f1.create_dataset('id',(len(ids),) ,str(hf['id'].dtype), ids)
                dset2 = f1.create_dataset('{}'.format(types),(len(ids),int(hf['{}'.format(types)].shape[1]),int(hf['{}'.format(types)].shape[2]),int(hf['{}'.format(types)].shape[3])) ,["|u1" if types=='vil' else "i2"][0], images)
                s3 = boto3.resource(
                                service_name='s3',
                                region_name='us-east-2')
                s3.Bucket(bucket_name).upload_file(Filename='/tmp/tmpfile.h5', Key= 'sevir/' + filename)
                f1.close()
                os.remove('/tmp/tmpfile.h5')
                
    print('{} loaded successfully'.format(filename))
    s3 = s3fs.S3FileSystem()
    hf.close()
    s3file.close()
print('Success')

