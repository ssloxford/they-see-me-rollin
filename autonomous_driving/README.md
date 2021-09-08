# They See Me Rollin' - Effect on Autonomous Driving

This folder contains the evaluation source code for Appendix C - Effect on Autonomous Driving.
We used the open-source autonomous driving simulator [CARLA](https://github.com/carla-simulator/carla) in combination with the autonomous vehicle platform [Pylot](https://github.com/erdos-project/pylot) to measure how the system as a whole is affected by the Rolling Shutter Attack.
We leveraged the [CARLA Autonomous Driving Leaderboard](https://github.com/carla-simulator/leaderboard), to analyze and benchmark the behaviour of the autonomous agent under two realistic traffic situations.
In Scenario 1, a pedestrian and in Scenario 2 a cyclist suddenly crosses the road.
The two scenarios are depicted in Figure 1.

<table align="center"><tr>
<td> 
  <p align="center">
    <img src="https://github.com/OXSKCS/theyseemerollin_artifiact/blob/main/pylot/doc/scenario_1.gif" width="400">
    <br>
    <em style="color: grey">a) Scenario 1</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img src="https://github.com/OXSKCS/theyseemerollin_artifiact/blob/main/pylot/doc/scenario_2.gif" width="400">
    <br>
    <em style="color: grey">b) Scenario 2</em>
  </p> 

</td>
</tr></table>
 <p align="center"><em align="center">Fig. 1: The two scenarios used to analyze the Rolling Shutter Attack in a dynamic setting.</em></p>

## Table of Contents
<details open>
<summary><b>(click to expand or hide)</b></summary>
<!-- MarkdownTOC -->

1. [Structure of the Directroy](#directory_structure)
1. [Rolling Shutter Attack Implementation](#rolling_shutter_implementation)
1. [Running the Simulation](#run_simulation)
1. [Analyzing the Results](#analyze_results)
1. [Known Issues](#known_issues)
    1. [Wrong Permissions](#wrong_permissions)
    1. [Zombie Process](#zombie_process)
1. [Recommended Hardware](#recommended_equipment)

<!-- /MarkdownTOC -->
</details>

<a id="directory_structure"></a>
## Structure of the Directroy
This directory is organized as follows:

```
.                                         # root directory of the Pylot evaluation
├── docker-compose.yml                    # configuration file of the Docker container
├── Dockerfile                            # build instructions for the Docker container
├── README.md                             # this README file
└── rsa                                   # directory that contains all necessary files to run the simulation
    ├── agents                            # directory that contains the modified agent
    │   └── ERDOSAgentUnderRSAAttack.py   # modified agent used to test the effect of the Rolling Shutter Attack
    ├── configs                           # directory with config files for the two object detectors
    │   ├── frcnn.conf                    # config file that specifies the use of FRCNN for object detection
    │   └── ssd.conf                      # config file that specifies the use of SSD for object detection
    ├── leaderboard                       # CARLA Autonomous Driving Leaderboard
    ├── patterns                          # directory that contains patterns for different attack settings (exposure, duty cycle)
    ├── recordings                        # directory where simulation recordings are stored
    ├── results                           # directory where simulation results are stored
    ├── routes                            # directory that contains the routes used for the simulation
    │   └── custom_routes.xml             # custom routes, copied and modified from CARLA Autonomous Driving Leaderboard
    ├── scenarios                         # directory that contains the scenarios used for the simulation
    │   └── custom_scenarios.json         # custom scenarios, copied and modified from CARLA Autonomous Driving Leaderboard
    ├── scripts                           # scripts that help to run the simulation
    │   ├── cleanup.sh                    # script that ensures that all the processes spawned by the simulation are killed
    │   └── pylot_evaluation.sh           # main script that walks you through all the required settings and executes the simulation
    └── utils                             # additional scripts

```

<a id="rolling_shutter_implementation"></a>
## Rolling Shutter Attack Implementation

We copied and slightly modified the `ERDOSAgent.py` to enable Rolling Shutter Attacks.
We added the following Python code that intercepts the camera frames sent from CARLA to Pylot and applies the Rolling Shutter pattern, before they are fed into the object detectors.

```python
# Check if the current stream is from the center camera, which is used for the object detection.
# If it is, apply the Rolling Shutter pattern.
if key == "center_camera":
    val[1][:, :, :3] = apply_pattern_v2(val[1][:, :, :3], pattern_BGR)
```

The modified version can be found in `rsa/agents`.

<a id="run_simulation"></a>
## Running the Simulation

To ensure a quick and easy setup, we provide a Dockerfile to build a Docker container with Pylot as a base image and all the required dependencies.
<br>**Please note**, to run the simulations, you will need the newest version of `docker` and `docker-compose`, as well as a GPU with updated drivers.
For more information about the required hardware, please refer to Section [Recommended Equipment](#recommended_equipment)
<br>
We provide an interactive shell script `pylot_evaluation.sh` that allows you to run the simulation in an end-2-end manner. 
This means, the script will walk you through all the necessary configuration steps and in the end execute the simulation.
There are two different simulations you can run: `baseline` and `rolling shutter attack`.
As the name indicates, the `baseline` just runs the two predefined scenarios without any attack.
In contrast, the setting `rolling shutter attack`, intecepts the camera frames and applies the Rolling Shutter pattern to the frames, before they are fed into Pylot.  

<a id="analyze_results"></a>
## Analyzing the Results

Analyzing the results is a two-step process. 
In the first step, the results from the CARLA Leaderboard simulation are converted from JSON to CSV format by running the following script:
```sh
python3 /home/erdos/rsa/utils/json2csv.py -p <PATH/TO/CALRA/RESULTS>
```
This change of format helps to analyze the data.
Each row in the CSV represents the results of one repetition.
In the second step, the rows are aggregated by calculating the mean of the infractions. 
```sh
python3 /home/erdos/rsa/utils/aggregate.py -p <PATH/TO/PREVIOUS/GENERATED/CSV>
```
<a id="known_issues"></a>
## Known Issues

<a id="wrong_permissions"></a>
#### Wrong Permissions
Depending on your setup and the environment Docker is running, it is possible that the Docker container does not have permissions to write to the mounted folder `rsa`.
If this is the case, we recommend to run the following command within the docker container:
```sh
sudo chmod -R o+rw /home/erdos/rsa
```
This will add read and write permission to the directory `rsa` and its subfolders.

<a id="zombie_process"></a>
#### Zombie Process

It can happen that the simulation does not exit as expected and continues to run as a zombie process.
This can happen, for example, due to limited resources or when the simulation is interrupted (CTRL + c).
Since the zombie process can cause a resource leak, it should be ensured that no such process is running before executing the simulation.
To ensure a clean simulation environment, we recommend to run the `cleanup.sh` script in `/home/erdos/rsa/scripts` before starting the simulation.

<a id="recommended_equipment"></a>
## Recommended Hardware

**Please note:** running CARLA and Pylot requires powerful hardware.
According to the [CARLA GitHub repository](https://github.com/carla-simulator/carla), the following hardware is a minimum requirement:

* Intel i7 gen 9th - 11th / Intel i9 gen 9th - 11th / AMD ryzen 7 / AMD ryzen 9
* +16 GB RAM memory
* NVIDIA RTX 2070 / NVIDIA RTX 2080 / NVIDIA RTX 3070, NVIDIA RTX 3080
* Ubuntu 18.04
