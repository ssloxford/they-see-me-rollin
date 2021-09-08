FROM tensorflow/tensorflow:2.1.0-py3
WORKDIR /home

# expose this port for jupyter notebook
EXPOSE 7331

# avoid questions when installing stuff in apt-get
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y install python3-tk
RUN apt-get -y install git
RUN apt-get -y install wget
RUN apt-get -y install ffmpeg  # needed for scikit-video

COPY requirements.txt /home/requirements.txt
RUN pip install -r requirements.txt
RUN rm /home/requirements.txt

# ======= Set up tensorflow models ========== #
WORKDIR /home
ENV MODELS_SHA1 ea61bbf06c25068dd8f8e130668e36186187863b
RUN git clone https://github.com/tensorflow/models.git
WORKDIR /home/models
RUN git reset --hard $MODELS_SHA1

# ======= Set up Coco API ========== #
WORKDIR /home
ENV COCOAPI_SHA1 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR /home/cocoapi/
RUN git reset --hard $COCOAPI_SHA1
WORKDIR /home/cocoapi/PythonAPI
RUN make
RUN cp -r pycocotools /home/models/research/
RUN apt -y install protobuf-compiler python-pil python-lxml python-tk
WORKDIR /home/models/research/
RUN protoc object_detection/protos/*.proto --python_out=.

# Environment
ENV PYTHONPATH "/home/models/research/:/home/models/research/slim:$PYTHONPATH"

WORKDIR /home
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.password='sha1:88cab7e51516:4dbe3e130dc43b83ac19bd1bd0756c1276b85d9c'">>/root/.jupyter/jupyter_notebook_config.py
CMD if [ ! -d "/home/data/datasets" ]; then mv /home/datasets /home/data; fi && nohup jupyter lab --port=7331 --allow-root --ip=0.0.0.0 & bash

