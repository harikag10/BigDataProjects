Snowflake Pipeline:<br/>

![](images/flow.png)



**Getting Started**<br/>
Download & Configure Apache Superset from [here](https://superset.apache.org/docs/installation/installing-superset-from-scratch)<br/>
Connect to Apache Superset with connection string :<br/>
              **snowflake://{username}:{password}@{account}/{database}/{schema}?warehouse={warehouse}&role={role}**<br/><br/>
*Python Libraries*<br/>
- ConfigParser<br/>
- sqlalchemy<br/>
- apache-superset<br/><br/>

**Usage**<br/><br/>

Using **snowflake_scripts**<br/>

Run this file in snowflake database to create tables/views and import data from S3 <br/><br/>

Using **config_file.ipynb** <br/>

Add the database connection details and run the file for setting up database connection parameters<br/><br/>

Using **SQLAlchemy.ipynb**<br/>

Run this file for querying the snowflake database<br/>

**Apache Superset Dashboard**

![](images/F55E5692-8AD0-43A5-964B-A860F0E5E2F0.jpeg)

![](images/1F48ED21-77A1-4960-9923-9B1A45AB33A7_4_5005_c.jpeg)











