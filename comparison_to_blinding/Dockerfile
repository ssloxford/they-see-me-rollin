FROM ubuntu:18.04

RUN apt -y update && apt -y install python3-pip wget locales libxrender-dev libsm6 git sudo libboost-all-dev libusb-1.0-0-dev python-mako doxygen python-docutils cmake build-essential ffmpeg
ENV LANG C.UTF-8

RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

ADD ./code /home/code
WORKDIR /home/code

### Install Python Requirements and Jupyter ###

RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade jupyter
RUN pip3 install jupyterlab

WORKDIR /home/code/scripts

### Spawn a Shell 
CMD [ "/bin/bash" ]
