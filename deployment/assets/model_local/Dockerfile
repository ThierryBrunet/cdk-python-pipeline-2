FROM python:3.9-slim-buster

# Install scikit-learn and pandas
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN mkdir /modeldata
RUN python --version
COPY model_download.py /
RUN ls -la
ENTRYPOINT ["python", "/model_download.py"]