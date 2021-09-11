# Evaluation on Cameras and Object Detection

The source code in this folder was used for the evaluation of Section 6 and 7.
This folder includes everything you need to run the evaluation.


## Structure of the Directory

```
.     
├── docker-compose.yml     
├── Dockerfile     
├── requirements.txt                                   
├── rsa                                                  
│   ├── attack_object_detection.py       # script to run object detectors on images
│   ├── config.yaml                      # configuration data
│   ├── make_figures.py                  # generates Figure 6, 7, 9 and 10
│   ├── profile.py                       # script to perform analysis of RS affected rows
│   └── utils.py                         # utilities
├── data                                 
│   ├── datasets                         # contains video datasets
│   │   └── bdd100k/videos/val/          # bdd100k video dataset subset
│   ├── models                           # contains tensorflow models
│   ├── profiling                        # surveillance video dataset
│   │   └── Logitech/30.1Hz/Exposure 1/  # examples of collected videos in dark environment
│   └── results                          # results
│       ├── extracted_patterns           # all patterns extracted for the Axis camera
│       ├── object_detection             # results for attack on object detection 
│       └── profiling                    # number of rows affected results
└── README.md                            # this README file
```

## Using the Docker Container

To facilitate deployment, we provide a Dockerfile to build a container with all the required dependencies.
<br>**Please note**, to use this repository, you will need `docker` and `docker-compose`.
<br>You can build the container by running:

```sh
docker-compose build
```
and run it with:

```sh
docker-compose up -d
```
As soon as the container is up and running, you can attach to it directly by running:
```sh
docker attach ra_object_detection
```
This is necessary to have the correct environment and folder structure to run the evaluation.

The two folders `rsa` and `data` are automatically mounted into the Docker container. 
Note that if you want to execute on GPU it is sufficient to add the tag `gpu` to the image
the dockerfile is pulling from (`FROM tensorflow/tensorflow:2.1.0-py3-gpu`).

## Running no. of Affected Row Analysis

The following script execution will print the camera-extracted information about
the number of illuminated pixel rows to console for a given video:
```sh
python profile.py -f /home/data/profiling/Logitech/30.1Hz/Exposure\ 1/freq_30.1_exp_1_dc_10.mkv
```

The same script can be run for an entire folder with the approriate subdirectories:
```sh
python profile.py --camera_id Logitech 
```
in this case the output is saved into a `.csv` file in `/home/data/results/profiling/`.

We provide the final csv files extracted by this script in `/home/data/results/profiling/`,
named `Axis_final.csv` and `Logitech_final.csv`. These contain the information 
used for Figure 6.

## Running Attack on Object Detection Evaluation

This part uses the `attack_object_detection.py` script.

It requires tensorflow models that can be downloaded from the [release file](https://github.com/ssloxford/they-see-me-rollin/releases/download/v1.0/models.zip) found in this
repository. These should be available into `repository/object_detection/data/models/` (which is mapped to
`/home/data/models/` in the container).

When evaluating the effect of the rolling shutter attack, we compare inferences between
rolling-shutter corrupted frames with inferences on legitimate frames.
To run inferences and store helpers results for inference on legit frames,
for a certain input video and a certain tensoflow model, run the following script:

```sh
python attack_object_detection.py \
            --pattern_filepath baseline \
            --video_name b1c9c847-3bda4659.mov \
            --model_name ssd_inception_v2_coco_2018_01_28
```
This saves results in `/home/data/results/object_detection/<DATASET>/baseline/<video_name>/<model_name>/`.

The same script can be run to store the same results for inference on corrupted frames. 
In this case a rolling-shutter pattern has to be specified as the corrupting-pattern.
This script will throw an error if the baseline results for the same combination of 
parameters is not present (`model_name` and `video_name`), as this is necessary to
compute the rolling-shutter effect.
```
python attack_object_detection.py \
            --pattern_filepath /home/data/results/extracted_patterns/Axis/259Hz/Exposure\ 75/5/40.png \
            --video_name b1c9c847-3bda4659.mov \
            --model_name ssd_inception_v2_coco_2018_01_28
```
This saves results in `/home/data/results/object_detection/<DATASET>/baseline/<video_name>/<model_name>/`.

We provide the final `.csv`s extracted by executing the above script and concatenating
all single-video `.csv` results in `/home/data/results/object_detection/`, named 
`bdd100k_final.csv` and `virat_final.csv`. These contain the information necessary to plot Figure 9 and 10.


## Training the Rolling Shutter Detection Defense

This part uses the `defense.py` script.

As above, it requires tensorflow models that which be downloaded from the [release file](https://github.com/ssloxford/they-see-me-rollin/releases/download/v1.0/models.zip) found in this
repository. These should be available into `repository/object_detection/data/models/` (which is mapped to
`/home/data/models/` in the container).

It also requires the extracted Axis camera rolling shutter patterns which can be downloaded from the [release file](https://github.com/ssloxford/they-see-me-rollin/releases/download/v1.0/Axis_patterns.zip) found in this
repository. These should be available into `repository/object_detection/data/results/extracted_patterns/Axis/` (which is mounted inside the container).

The script adds a network head to an object detector architecture which is in charge of predicting whether an input frame is corrupted by a rolling shutter attack.
To train the head, the script takes frames from BDD100K videos and the rolling shutter attack patterns and creates a dataset to train on. 

One can run training on top of Single-Shot Detector as follows:

```sh
python defense.py --model_name ssd_inception_v2_coco_2018_01_28 \
                  --num_videos_train 1 \
                  --num_videos_val 1 \
                  --num_videos_test 1
```
This saves classification, execution time and trained models in `/home/data/results/defense/<model_name>/`.

N.B.: To train on several videos, more videos have to be present in `/home/data/datasets/bdd100k/videos/val` (see [BDD100K](https://github.com/bdd100k/bdd100k)) or you have to write your own video file finder. At the moment only three of these are included in this repository.

## Generating figures

Figures 6, 7, 9 and 10 can be generated by running:
```sh
python make_figures.py
```
which stores figures in `/home/rsa/tmp_figures/`.

# Recommended Hardware

The scripts presented in this readme take hours to run
for the entirety of the collected data, so we provide intermediate/final results files.
The execution as presented in the paper was carried out on an NVIDIA TITAN RTX gpu.

