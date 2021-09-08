import glob
import multiprocessing
import cv2
import sys
sys.path.append("../lib/")
import skimage
import pandas as pd
import os
from skimage import io, transform
from EvaluationUtils import *
from SimulationSetting import *
from SimulationTask import *
from SimulationTaskResult import *
import argparse
import random
import time
import skvideo.io
import numpy as np
import warnings
warnings.filterwarnings("ignore")

path_to_videos = "/home/data/videos/"
result_path = "/home/data/image_metrics_results.csv"

# These arrays contain the possible simulation settings
# Available simulation types
simulation_types = ["baseline", "attack"]
# Available laser frequencies
frequencies = [750, "blinding"]
# Available exposure times
exposure_times = [32, 200]
# Available duty cycles
duty_cycles = [1.0, 20, 40]
# Available video datasets
video_datasets = ["BDD100K", "VIRAT", "both"]

# Get the maximum number of CPU cores
max_number_of_cores = multiprocessing.cpu_count()

def read_input(info_text, expected_input=None):
    while True:
        print(info_text)
        if expected_input:
            for index, value in enumerate(expected_input):
                print(f"{value} [{index + 1}]")
            user_selection = int(input()) - 1
            # Try to access the selected option
            # If the option does not exist, an exception is raised
            try:
                return expected_input[user_selection]
            # Catch the exception and ask user again for the input
            except:
                print("Sorry, you have to pick one of the available options!")
                print("-----------------------------------------")
                continue
        else:
            # Read user input, if the input is not an integer or negativ,
            # raise an exception
            try:
                user_input = int(input())
                if user_input < 1:
                    raise Exception()
                return user_input
            # Catch the exception and ask user again for the input
            except:
                print("Please make sure the provided number is a positiv integer!")
                print("-----------------------------------------")
                continue

# Loads the patterns based on the parameters specified by the user
# Since the number of patterns can be chosen by the user, the patterns are shuffled 
# to ensure that the same patterns are not always tested
# The method returns two arrays - one with the pattern IDs and one with the patterns themselves
def load_patterns(simulation_setting):
    if simulation_setting.frequency == "blinding":
        # Path that contains all patterns for given frequency, exposure and duty cycle
        path = f"/home/data/patterns/blinding/*.png"
    else:
        # Path that contains all patterns for given frequency, exposure and duty cycle
        path = f"/home/data/patterns/{simulation_setting.frequency}/{simulation_setting.exposure}/{simulation_setting.duty_cycle}/*.png"
    # Get paths to all the patterns
    patterns_paths = glob.glob(path)
    # Shuffle the array to add randomness 
    random.shuffle(patterns_paths)
    
    pattern_ids = []
    patterns = []
    # Iterate over the pattern paths, extract the ID from the path and read the pattern
    # Note: The array is truncated based on the number of patterns that the user chose
    for pattern_path in patterns_paths[:simulation_setting.number_of_patterns]:
        pattern_ids.append(EvaluationUtils.get_file_name_from_path(pattern_path))
        patterns.append(skimage.io.imread(pattern_path))
    
    return pattern_ids, patterns

# Loads the videos from the selected video dataset
# Since the number of videos can be chosen, the videos are shuffled 
# to ensure that the same videos are not always tested
def load_videos(dataset):
    path = f"/home/data/videos/{dataset}/*"
    videos_paths = glob.glob(path)
    random.shuffle(videos_paths)
    return videos_paths

# This method executes the simulation
# Since the calculation is running in separate processes,
# the data is passed to the function via an array
def execute_simulation(parameters):
    # extract the different parameters from the array
    simulation_task = parameters[0]
    previous_frame = parameters[1]
    current_frame = parameters[2]
    pattern = parameters[3]
    
    # The pattern_id also indicates if it is the baseline simulation
    # If the pattern_id is not -1, the attack pattern will be applied to the current frame
    if simulation_task.pattern_id != -1: 
        # apply the pattern to the current_frame
        current_frame = EvaluationUtils.apply_pattern_v2(current_frame, pattern)
    
    return EvaluationUtils.calculate_metrics_for_simulation_task(simulation_task, previous_frame, current_frame)

def run_evaluation(simulation_setting):
    results = []
    simulation_tasks = []
    pattern_ids, patterns = load_patterns(simulation_setting)
    
    # For each dataset specified, run the simulation
    for video_dataset in simulation_setting.video_datasets:
        # Load the paths to all videos from the current dataset
        videos_paths = load_videos(video_dataset)
        
        # For the specified number of videos run the simulation
        for video_index, video_path in enumerate(videos_paths[:simulation_setting.number_of_videos]):
            # Extract the video name from the path
            video_name = EvaluationUtils.get_file_name_from_path(video_path, with_file_extension=True)
            print("Loading video now into memory, please wait...")
            # Read the video into the memory
            video = skvideo.io.vreader(video_path)
            # Convert the generator object to an actual list of frames
            video_frames = list(video)
            del video
            # Iterate over the frames of the video with the step size (sample_rate) specified
            for current_frame_index in range(1, len(video_frames) - 1, simulation_setting.sample_rate):
                previous_frame_index = current_frame_index - 1
                
                previous_frame = video_frames[previous_frame_index]
                current_frame = video_frames[current_frame_index]
                
                # Check if the simulation_type is attack
                # If it is, we need to iterate over the patterns
                # If not, we can just run the simulation without any pattern
                if simulation_setting.simulation_type == "attack":
                    
                    for pattern_index, pattern in enumerate(patterns):
                        pattern_id = pattern_ids[pattern_index]
                        simulation_task = SimulationTask(video_name, video_dataset, previous_frame_index, current_frame_index, simulation_setting.frequency, simulation_setting.exposure, simulation_setting.duty_cycle, pattern_id)
                        simulation_tasks.append([simulation_task, previous_frame, current_frame, pattern])
                else:
                    simulation_task = SimulationTask(video_name, video_dataset, previous_frame_index, current_frame_index, simulation_setting.frequency, simulation_setting.exposure, simulation_setting.duty_cycle, -1)
                    simulation_tasks.append([simulation_task, previous_frame, current_frame, 0])
    
            print(f"Running simulation {video_index} out of {simulation_setting.number_of_videos}!")
            # Create a multiprocessing pool 
            p = multiprocessing.Pool(simulation_setting.number_of_cores)
            # Map the list of simulation tasks to the pool
            results = p.map(execute_simulation, simulation_tasks)
            p.close()
            p.join()
    
    # Once all the simulations finished save the results to a CSV files
    EvaluationUtils.save_results(result_path, results)
    print(f"Simulation completed, please find the results in {result_path}")

def get_user_input():
    
    # Initialize the attack parameters
    # Please note, a frequency, exposure and duty cycle of 0 indicates a baseline simulation
    frequency = 0
    exposure = 0
    duty_cycle = 0
    number_of_patterns = 0
    
    print("This script runs the evaluation described in Section 7.2 - Comparision with Blinding Attack.")
    print("It will walk you through all the configurations required to calculate different similarity metrics between legitimate and malicious video frames.")
    print("Please note: Depending on the number of videos and the attack parameters, the evaluation might run for multiple days!")

    simulation_type = read_input("What type of simulation would you like to run?", simulation_types)
    print("-----------------------------------------")
    
    if simulation_type == "attack": 
        frequency = read_input("Please select the laser frequency or if you want to evaluate blinding:", frequencies)
        print("-----------------------------------------")
        if frequency != "blinding":    
            exposure = read_input("Please select the exposure time you want to evaluate:", exposure_times)
            print("-----------------------------------------")
            duty_cycle = read_input("Please select the duty cycle (in %) you want to evaluate:", duty_cycles)
            print("-----------------------------------------")
        number_of_patterns = read_input("Please input the number of patterns:")
        print("-----------------------------------------")
        
    video_dataset = read_input("For which video dataset do you want to run the evaluation?", video_datasets)
    print("-----------------------------------------")
    number_of_videos = read_input("For how many videos per dataset do you want to run the simulation?")
    print("-----------------------------------------")
    sample_rate = read_input("Please input the sample rate for extracting frames from the video. The value determines the interval between the frames:")
    print("-----------------------------------------")
    number_of_cores = read_input(f"Please input the number of cores you want to use (max. {max_number_of_cores}):")
    print("-----------------------------------------")
    print("Perfect, all required parameters have been set. The simulation will start now.")
    print(f"The results will be written to {result_path}")
    
    return SimulationSetting(simulation_type, frequency, exposure, duty_cycle, number_of_patterns, video_dataset, number_of_videos, sample_rate, number_of_cores)
    
if __name__ == "__main__":  
    simulation_setting = get_user_input()
    run_evaluation(simulation_setting)    
