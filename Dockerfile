FROM python:3.9-slim-buster

# Install scikit-learn and pandas
RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get -y install g++
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
COPY requirements.txt .
RUN python --version
RUN pip install -r requirements.txt

COPY src/ /
RUN ls -la /resources
ENTRYPOINT ["python", "/main.py"]

# FROM continuumio/miniconda3:latest

# RUN apt-get -y update &&\
#     apt-get install -y  --no-install-recommends \
#     libpq-dev \
#     libffi-dev \
#     libssl-dev \
#     bash-completion &&\
#     apt-get autoremove &&\
#     apt-get clean &&\
#     echo "[ -f /etc/bash_completion ] && . /etc/bash_completion" >> ~/.bashrc &&\
#     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# # initialize your conda env (this can be expensive, so we cache it in a lower layer here)
# COPY environment.yml .
# RUN conda env update --name base --file environment.yml &&\
#     conda clean --all -y

# # Copy your project contents into the Docker image
# RUN python --version
# COPY src/ /

# ENTRYPOINT ["python", "/main.py"]