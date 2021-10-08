import sys
sys.path.append("./")
from SimulationTask import *
from SimulationTaskResult import *

from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import uqi as uqi
from sewar.full_ref import msssim as msssim
import numpy as np
import pandas as pd
import os

class EvaluationUtils:
    
    # This method takes an object, converts it into a dict and stores it into a CSV file
    def save_results(file_name, results, overwrite=False):
        if len(results) > 0:
            data_frame = pd.DataFrame(columns=results[0].to_dict().keys())
            for result in results:
                data_frame = data_frame.append(result.to_dict(), ignore_index=True)

            # Check if file exists, if so, append the file.
            # If not, create a new one.
            if os.path.isfile(file_name):
                # If overwrite is True, the result file will be overwritten
                if overwrite:
                    data_frame.to_csv(file_name, index=False)
                else:
                    data_frame.to_csv(file_name, mode='a', header=False, index=False)
            else:
                data_frame.to_csv(file_name, index=False)

    # This method calculates different image metrics between two consecutive frames
    def calculate_metrics_for_simulation_task(simulation_task, previous_frame, current_frame):
        # The WimulationTask that contains all the important information about the current simulation
        # is parsed to a SimulationTaskResult object
        simulation_task_result = SimulationTaskResult(simulation_task)

        # Calculate SSIM
        simulation_task_result.ssim = ssim(previous_frame, current_frame, multichannel=True)

        # Calculate MS-SSIM
        simulation_task_result.ms_ssim = float(msssim(previous_frame, current_frame))

        # Calculate UQI            
        simulation_task_result.uqi = uqi(previous_frame, current_frame)

        # return the SimulationTaskResult object
        return simulation_task_result

    def convert_to_rgb(img):
        if img.shape[2] == 4:
            img = img[:,:,:3]
        return img
        
    
    # This method takes a base image as input and overlays the pattern
    # Important: the base image and the pattern must have the same resolution!
    def apply_pattern_v2(base, pattern, a1=1, a2=2):
        assert base.dtype == np.uint8
        assert pattern.dtype == np.uint8      
        # Make sure both images have the same "shape"
        # i.e., are RGB rather than RGBA
        base = EvaluationUtils.convert_to_rgb(base)
        pattern = EvaluationUtils.convert_to_rgb(pattern)
        
        b = base.astype(float)
        p = pattern.astype(float)
        r = np.clip(b + p, 0, 255).astype(np.uint8)
        return r
    
    # This method splits a file path and only returns the filename 
    # The filename can either be returned with or without the file extenstion
    def get_file_name_from_path(file_path, with_file_extension=False):
        file_name = file_path.split("/")
        file_name = file_name[len(file_name) - 1]
        if not with_file_extension:
            file_name = file_name.split(".")[0]
        return file_name