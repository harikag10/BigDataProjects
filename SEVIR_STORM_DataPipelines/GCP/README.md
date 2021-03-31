## Experiment with Google
Here, we are leveraging Google Cloud Platform for developing data pipeline using Google Dataflow, query and visualize the data using Google BigQuery and Google Data studio respectively

## Contents
* GCP_DATAFLOW_SEVIR_STORM.py -> Includes the code to run the data pipeline to sample the data from s3 and cloud storage and load back into storage and into BigQuery
* SEVIR_STORM_JOIN_BQUERY.sql -> SQL Scripts for joining Sevir and Storm data and creating views 

## Instructions to run 
- Download the storm data from https://www.ncdc.noaa.gov/stormevents/ftp.jsp and store it in your cloud storage buckets
- Modify the Bucket_name in the code 
- Run the 'GCP_DATAFLOW_SEVIR_STORM.py' to start the Dataflow Pipeline Job
- Once th job is completed and data loaded into Big Query tables, run the SEVIR_STORM_JOIN_BQUERY.sql script to create the views, which can be used in visualizations through Google Data Studio


## Report 
Refer to https://codelabs-preview.appspot.com/?file_id=1jBm19uLwkIqiLjWorO7ClJ5UcWRHkf_WgXqZcVmBPM0#3 for detailed description of the pipeline illustration 
