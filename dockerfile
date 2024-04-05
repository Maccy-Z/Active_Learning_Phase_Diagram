FROM python:3.11
RUN apt update

RUN pip install numpy==1.26
RUN pip install matplotlib==3.8
RUN pip install numba==0.59
RUN pip install scikit-learn==1.4
RUN pip install torch==2.2 --index-url https://download.pytorch.org/whl/cpu


### Install GUI dependency
RUN apt install python3-pyqt5 -y
RUN pip install pyqt5


# Pip mayavi is out of date currently.
RUN pip install git+https://github.com/enthought/mayavi.git@8ae6e5cc4e95607541da210070066e79f2de3e7e
RUN pip install vtk==9.2.*
