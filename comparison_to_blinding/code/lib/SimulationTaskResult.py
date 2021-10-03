import sys
sys.path.append("./lib/")
from SimulationTask import *

class SimulationTaskResult(SimulationTask):
       
    ssim = 0.0
    ms_ssim = 0.0
    uqi = 0.0
        
    def __init__(self, simulation_task):
        super().__init__(simulation_task.video_name, simulation_task.video_dataset, simulation_task.previous_frame, simulation_task.current_frame, simulation_task.frequency, simulation_task.exposure, simulation_task.duty_cycle, simulation_task.pattern_id)
        
    def to_dict(self):
        return self.__dict__