FROM python:3.9-slim-buster

# Install scikit-learn and pandas
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /
ENTRYPOINT ["python", "/main.py"]