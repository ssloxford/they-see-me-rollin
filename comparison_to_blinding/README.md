# Comparision with Blinding Attack

The source code in this folder was used for our evaluation presented in Section 7.2 - Comparision with Blinding Attack.
We calculated three different image quality metrics between consecutive video frames, namely SSIM, MS-SSIM and UQI, to evaluate the amount of distortion caused by the Rolling Shutter Attack in comparison to fully blinding.
We found that for short exposure values, the Rolling Shutter Attack causes interference that is similar across all metrics to the expected level of perturbation seen in consecutive legitimate video frames.
This folder includes everything you need to run the evaluation yourself.

## Structure of the Directory

```
.                                   # root directory of the repository
├── code                            # contains the evaluation source code
│   ├── lib                         # various Python classes required for the evaluation
│   │   ├── AggregatedResults.py    # Python class that contains the aggregated results
│   │   ├── EvaluationUtils.py      # Python class that implements various utils required for the evaluation
│   │   ├── SimulationSetting.py    # Python class that defines the different simulation settings
│   │   ├── SimulationTask.py       # Python class that defines a simulation task
│   │   └── SimulationTaskResult.py # Python class that stores the results of a simulation
│   ├── requirements.txt            # text file that contains all the Python requirements
│   └── scripts                     # the scripts required to run the evaluation
│       ├── aggregate_results.py    # Python script that aggregates the results
│       ├── calculate_roc.py        # Python script to calculate ROC-AUC
│       └── run_evaluation.py       # Python script that runs the evaluation
├── data                            # persistent directory in the patterns and videos are stored
│   ├── patterns                    # different patterns used in our evaluation
│   │   ├── 750                     # variety of example patterns
│   │   └── blinding                # example frames, showing blinding
│   └── videos                      # video datasets used in our evaluation
│       ├── BDD100K                 # driving dataset
│       └── VIRAT                   # surveillance video dataset
├── docker-compose.yml              # configuration file of the Docker container
├── Dockerfile                      # build instructions for the Docker container
└── README.md                       # this README file
```

## Flowchart

The following flowchart illustrates the flow of the evaluation.

<p align="center"><img src="https://github.com/ssloxford/they-see-me-rollin/blob/main/comparison_to_blinding/doc/flowchart_comparison_with_blinding.jpg" width="70%"><br><em style="color: grey">Fig. 1: Flowchart illustrating the flow of the run_evaluation.py script.</em></p>

## Running the Evaluation

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
As soon as the container is up and running, you can either attach to it directly by running:
```sh
docker attach comparison_to_blinding
```

or by spawning a new shell:
```sh
docker exec -it comparison_to_blinding /bin/bash
```
The two folders `code` and `data` are automatically mounted into the Docker container.
To make it easier for you to get started, we provide three scripts that help you to run the evaluation.
All scripts are located in `/home/code/scripts`.
The `run_evaluation.py` is an interactive Python script that helps you to set the required evaluation parameters and once done, runs the evaluation.
You can run the script with the following command:
```sh
python3 /home/code/scripts/run_evaluation.py
```

The script will store SSIM, MS-SSIM and UQI values for each frame pair, i.e., previous and current frame, in a CSV file.
<br>
Since the evaluation stores the results for each frame pair, the second script, `aggregate_results.py`, can be used to aggregate these results.
The image metrics results are grouped by `frequency`, `exposure` and `duty cycle` before being aggregated. 
<br>**Please note:** A frequency, exposure and duty cycle of 0 indicates a baseline simulation.
<br>The script requires two parameters - the path to the CSV file with the unaggregated results and the path to the CSV file in which the aggregated results should be stored.
You can run the script with the following command:

```sh
python3 /home/code/scripts/aggregate_results.py -i <PATH/TO/RESULTS> -o <PATH/TO/AGGREGATED/RESULTS>
```

We also provide a script to calculate the ROC-AUC.
<br>The script requires the path to the CSV file with the unaggregated results.
You can run the script with the following command:

```sh
python3 /home/code/scripts/calculate_roc.py -i <PATH/TO/UNAGGREGATED/RESULTS>
```

## Recommended Hardware

Running the simulations is very resource intesitiv. 
In particular, the RAM usage is very high, since the videos are loaded into the RAM before being processed.
Therefore, we recommend to run the evaluation on a system with at least 32GB of RAM.
The source code is optimized to run as a multicore application and does not require a GPU.
We recommend to use a PC/server with a high number of CPU cores.
