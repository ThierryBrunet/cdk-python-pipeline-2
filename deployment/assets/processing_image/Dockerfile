FROM python:3.7-slim-buster

RUN apt-get update
RUN apt-get install -y build-essential

# Install scikit-learn and pandas
ADD processing_image/requirements.txt .
RUN pip3 install -r requirements.txt

# Add a Python script and configure Docker to run it
ADD processing_image/match.py /
ADD processing_image/build_index.py /
ADD lib/utils.py /
# ENTRYPOINT ["python3", "/match.py"]
ENTRYPOINT ["python3"]
