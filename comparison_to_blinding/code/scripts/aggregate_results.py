import argparse
import pandas as pd
import sys
sys.path.append("../lib/")
from AggregatedResults import *
from EvaluationUtils import *

def aggregate_simulation_task_results(simulation_task_results_dataframe, output_file):
    # Array that will contain all the aggregated results
    aggregated_results = []
    
    # Iterate over the rows of the aggregated data
    for index, row in simulation_task_results_dataframe.iterrows():
        # Since we grouped the data based on Frequency, Exposure and the Duty Cycle
        # these parameters build the index/key for each row
        # Here we extract the parameters from the index
        frequency = index[0]
        exposure = index[1]
        duty_cycle = index[2]
        
        aggregated_result = AggregatedResults(frequency, exposure, duty_cycle)
        
        # SSIM
        aggregated_result.ssim_mean = row["ssim"]["mean"]
        aggregated_result.ssim_min = row["ssim"]["min"]
        aggregated_result.ssim_max = row["ssim"]["max"]
        aggregated_result.ssim_median = row["ssim"]["median"]
        aggregated_result.ssim_std = row["ssim"]["std"]

        # MS-SSIM
        aggregated_result.ms_ssim_mean = row["ms_ssim"]["mean"]
        aggregated_result.ms_ssim_min = row["ms_ssim"]["min"]
        aggregated_result.ms_ssim_max = row["ms_ssim"]["max"]
        aggregated_result.ms_ssim_median = row["ms_ssim"]["median"]
        aggregated_result.ms_ssim_std = row["ms_ssim"]["std"]

        # UQI
        aggregated_result.uqi_mean = row["uqi"]["mean"]
        aggregated_result.uqi_min = row["uqi"]["min"]
        aggregated_result.uqi_max = row["uqi"]["max"]
        aggregated_result.uqi_median = row["uqi"]["median"]
        aggregated_result.uqi_std = row["uqi"]["std"]
        
        aggregated_results.append(aggregated_result)

    # Save all the aggregated results to a CSV file
    EvaluationUtils.save_results(output_file, aggregated_results)
        
def run_aggregation(input_file, output_file):  
    # Read the CSV file into a Pandas DataFrame
    simulation_task_results = pd.read_csv(input_file) 
    # Group the data based on the Frequency, Exposure and the Duty Cycle
    grouped_data = simulation_task_results.groupby(['frequency', 'exposure', 'duty_cycle'])
    # Aggregate the different image metrics
    simulation_task_results_dataframe = grouped_data.agg({'ssim':['min', 'max', 'mean', 'median', 'std'], 
                                                          'ms_ssim':['min', 'max', 'mean', 'median', 'std'], 
                                                          'uqi':['min', 'max', 'mean', 'median', 'std']})
    
    aggregate_simulation_task_results(simulation_task_results_dataframe, output_file)
    
    
    
if __name__ == "__main__":  
    
    parser = argparse.ArgumentParser(description='Aggregate the results from the simulation.')
    parser.add_argument('--input_file', '-i', type=str, help='Path to the unaggregated CSV.', required=True)
    parser.add_argument('--output_file', '-o', type=str, help='Path to the aggregated CSV.', required=True)
    args = parser.parse_args()
    
    run_aggregation(args.input_file, args.output_file)
    print(f"Aggregated the data successfully. Please find the results in {args.output_file}")
    