FROM python:3.11
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update
RUN apt install -y python3-pip git
RUN pip install gpyopt
RUN pip install numpy==1.23 matplotlib