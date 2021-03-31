# Sentiment Analysis Flask API:

The main aim of this module is to download the trained tensorflow model from s3 bucket and expose it as Flask API for inference.

***Technical Requirements:***
- Docker pre-installed
- s3 bucket access

***Folder Structure:***
```
SentimentAPI-Flask-Part3/
├── api/
│   ├── app.py - Initialises the Flask app based on a blueprint
│   └── ml_app.py - The entire blueprint of the flask app withn POST and GET methods
├── config.yaml - Contains the dynamic parameters for the app execution
├── Dockerfile - Contains the steps required for dockerizing the application
├── loadyaml.py - Contains the code to load a specified yaml file
├── predict.py - Contains the code to get loaded model do the predictions and return the results
├── requirements.txt - Contains all the package requirements for app execution
├── run.py - The main file where the app execution starts
├── s3_download.py - Contains the code to download model from s3 bucket
└── saved_models/
    └── load_model.py - Contains the code which loads the model
```

***Steps to follow:***
- git clone the repository
- create a python3.7 environment using

    `pip3.7 install virtualenv`
    
    `python3.7 -m virtualenv venv`
    
    `source venv/bin/activate`
    
 - `pip install requirement.txt`
 
 - Dockerizing Application

    `docker build -t <imagename> .`
  
    `docker run -d -p 5000:5000 imagename`
  
Now the sentiment API is up and running and can be connected on http://0.0.0.0:5000/predict

***Deploying the app on Google Cloud Run:***<br>

- Install Google Cloud SDK 
- Authenticate the SDK using<br>
  `google auth login`<br>
- Authenicate docker to push images to Google Cloud Container Repositories<br>
  `google auth configure-docker`<br>
- Create a tag for your image<br>
  `docker tag imagename us.gcr.io/project-id/imagename` <br>
- Push the docker image<br>
  `docker push us.gcr.io/project-id/imagename` <br>
- Now to go to Cloud run and create a service with following parameters: <br>
  - select the image from the container repository which we pushed<br>
  - Memory: min 4GB<br>
  - Create<br>
- Once Cloud Run service is created we will get the service url and our API is up and running<br>
- To access the API use `SERVICE URL/predict`<br>

    
***Testing the API using Postman:***<br><br>
***Input:*** <br><br>
Should be in json format with a list of strings/sentences.<br><br>
`{"data": ["this is the best. It is a good watch", "this is worst!"]}`<br><br>
***Method:***<br><br>
POST<br><br>
***URL:***<br><br>
The service URL we get from Google Cloud Run and concatenating it with /predict <br><br>
Here in our case it is: `https://sentimentapi-zi7kg63pga-ue.a.run.app/predict`<br><br>
***Output:***<br><br>
JSON having to list predictions<br>
```
{
    "input": 
        "data": [
            "this is the best. It is a good watch",
            "this is worst!"
        ]
    },
    "pred": [
        0.9969875812530518,
        -0.5278245210647583
    ]
}
```


***Claat Document:*** https://codelabs-preview.appspot.com/?file_id=1jCLBg9N-M6sL1eEP3I5kE4cvZVNoPEeiTT1aiGq8qdY#0
