FROM python:3.11
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get clean all
RUN apt update

# Install GPy
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
# docker run -it --net=host --entrypoint /bin/bash -v /mnt/storage_ssd/PDSp:/opt/project -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix phase_sample:py311
# docker run -it -e DISPLAY -v /home/maccyz/Documents/active_phase:/opt/project --net host phase_sample:gui bash
