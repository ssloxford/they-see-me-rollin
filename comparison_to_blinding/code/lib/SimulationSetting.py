class SimulationSetting():
       
    def __init__(self, simulation_type, frequency, exposure, duty_cycle, number_of_patterns, video_dataset, number_of_videos, sample_rate, number_of_cores):
        self.simulation_type = simulation_type
        self.frequency = frequency
        self.exposure = exposure
        self.duty_cycle = duty_cycle
        self.number_of_patterns = number_of_patterns
        
        if video_dataset == "both":
            self.video_datasets = ["BDD100K", "VIRAT"]
        else:
            self.video_datasets = [video_dataset]
        
        self.number_of_videos = number_of_videos
        self.sample_rate = sample_rate
        self.number_of_cores = number_of_cores