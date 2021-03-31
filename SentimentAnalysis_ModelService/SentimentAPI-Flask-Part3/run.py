import logging
from api.app import initialize_app
from s3_download import s3_download

logger = logging.getLogger(__name__)

app = initialize_app()

# Model download from s3
model_download = s3_download()

if __name__ == "__main__":
    logger.info("Initializing a Flask app...")
    app.run(debug=True,host='0.0.0.0')