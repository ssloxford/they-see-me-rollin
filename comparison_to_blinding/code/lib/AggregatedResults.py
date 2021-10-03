import sys

class AggregatedResults:
       
    ssim_min = 0
    ssim_mean = 0
    ssim_max = 0
    ssim_median = 0
    ssim_std = 0
    
    ms_ssim_min = 0
    ms_ssim_mean = 0
    ms_ssim_max = 0
    ms_ssim_median = 0
    ms_ssim_std = 0
    
    uqi_min = 0
    uqi_mean = 0
    uqi_max = 0
    uqi_median = 0
    uqi_std = 0
    
    values = 0
        
    def __init__(self, frequency, exposure, duty_cycle):
        self.frequency = frequency
        self.exposure = exposure
        self.duty_cycle = duty_cycle
        
    def to_dict(self):
        return self.__dict__