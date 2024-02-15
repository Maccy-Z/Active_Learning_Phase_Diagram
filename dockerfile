FROM python:3.11
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get clean all
RUN apt update --fix-missing

# Install local modified version of GPy
RUN pip install numpy==1.23 matplotlib
WORKDIR /GPy
COPY SheffieldML-GPy-e91799a /GPy
RUN python setup.py install

## Install GUI dependency
RUN apt install libgl1-mesa-glx -y
RUN apt install libxcb-cursor0 -y

RUN apt install -y python3-pyqt6
RUN pip install pyqt6

RUN pip install dill
RUN pip install scikit-learn
RUN pip install scikit-image

