import logging
from pathlib import Path
import os
import tensorflow as tf
import click
import numpy as np
import boto3
import pandas as pd
from smart_open import smart_open
from loadyaml import load_yaml
from saved_models.load_model import load_tf_hub_model


def predict_online(data, config=None):
    """Predict from in-memory data on the fly.
    """

    model_path = './EDGARModel_bert'
    checkpoint = load_tf_hub_model(model_path)
    reloaded_results = tf.sigmoid(checkpoint(tf.constant(data)))

    config = load_yaml('config.yaml')
    aws_access_key_id = config["dev"]["aws_access_key_id"]
    aws_secret_access_key = config["dev"]["aws_secret_access_key"]
    bucket_name = config["dev"]["bucket"]
    object_key = config["dev"]["annotation"]["object_key"]
    prefix = config["dev"]["annotation"]["prefix"]

    path = 's3://{}:{}@{}/{}/{}'.format(aws_access_key_id, aws_secret_access_key, bucket_name, prefix, object_key)

    df = pd.read_csv(smart_open(path))

    a=df['metric'].to_numpy()

    # Data Sclaing in range -1 to 1
    b = reloaded_results.numpy()
    d = 2. * (b - np.min(a)) / np.ptp(a) - 1
    final_result = np.concatenate(d).ravel().tolist()
    return final_result