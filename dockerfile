FROM python:3.11
#FROM python:latest
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y python3-pip git


RUN apt install -y libgl1
RUN pip install opencv-python
RUN pip install numpy==1.23 matplotlib
#RUN pip install scipy


## RUN pip install gpyopt
#RUN pip install GPy

WORKDIR /GPy
COPY SheffieldML-GPy-e91799a /GPy
RUN python setup.py install
