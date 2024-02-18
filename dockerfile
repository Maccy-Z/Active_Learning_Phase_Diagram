FROM python:3.11
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get clean all
RUN apt update
RUN apt update --fix-missing

RUN pip install numpy matplotlib
RUN pip install numba
RUN pip install scikit-learn

## Install GUI dependency
RUN apt install libgl1-mesa-glx -y
RUN apt install libxcb-cursor0 -y
RUN apt install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
RUN apt install libdbus-1-dev -y
RUN pip install pyqt5

# Pip mayavi is out of date currently.
RUN pip install git+https://github.com/enthought/mayavi.git@8ae6e5cc4e95607541da210070066e79f2de3e7e

# RUN pip install mayavi

# docker run -it --net=host --entrypoint /bin/bash -v /mnt/storage_ssd/PDSp:/opt/project -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix phase_sample:py311
# docker run -it -e DISPLAY -v /home/maccyz/Documents/active_phase:/opt/project --net host phase_sample:gui bash
