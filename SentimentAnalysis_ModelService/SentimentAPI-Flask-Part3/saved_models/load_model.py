import tensorflow as tf
import tensorflow_text

def load_tf_hub_model(save_path):

    reloaded_model = tf.saved_model.load(save_path)
    return reloaded_model





