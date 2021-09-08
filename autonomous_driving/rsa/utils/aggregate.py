import argparse
import glob
import os
import pathlib
from pathlib import Path
import pandas as pd

def get_path_and_name(filepath):
    file_path_splitted = filepath.split("/")
    file_name = file_path_splitted[len(file_path_splitted)-1]
    path = filepath[:len(filepath)-len(file_name)]
    return path, file_name[:len(file_name)-5]

def aggregate_data(file, collisions_pedestrian, collisions_vehicle):
        simulation = pathlib.Path(file).stem
        df = pd.read_csv(file)

        collisions_pedestrian[simulation] = df.groupby("route_id")["collisions_pedestrian"].mean()
        collisions_vehicle[simulation] = df.groupby("route_id")["collisions_vehicle"].mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate the results from the CARLA Leaderboard. You can either provide a signle csv file, or a path to a folder that contains all results.")
    parser.add_argument("-p", "--input_path", type=str, help="Path to folder with all CSV files")
    parser.add_argument("-f", "--input_file", type=str, help="Path to CSV file")
    args = parser.parse_args()

    collisions_pedestrian = pd.DataFrame()
    collisions_vehicle = pd.DataFrame()

    path = ""

    if args.input_file != None:
        aggregate_data(args.input_file, collisions_pedestrian, collisions_vehicle)
        path, file_name = get_path_and_name(file)
    else:
        files = glob.glob(f"{args.input_path}/*.csv")

        for file in files:
            aggregate_data(file, collisions_pedestrian, collisions_vehicle)
            path = args.input_path

    Path(f"{path}/aggregated/").mkdir(parents=True, exist_ok=True)

    collisions_pedestrian.to_csv(f"{path}/aggregated/collisions_pedestrian.csv", index=True)
    collisions_vehicle.to_csv(f"{path}/aggregated/collisions_vehicle.csv", index=True)
    print(f"Finished aggregating the data. The results are stored at {path}aggregated")
