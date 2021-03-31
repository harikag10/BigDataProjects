FROM python:3.7-slim



COPY . /flask

WORKDIR /flask

RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [  "run.py" ]

