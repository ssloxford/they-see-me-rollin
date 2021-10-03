class SimulationTask():
       
    def __init__(self, video_name, video_dataset, previous_frame, current_frame, frequency, exposure, duty_cycle, pattern_id):
        self.video_name = video_name
        self.video_dataset = video_dataset
        self.previous_frame = previous_frame
        self.current_frame = current_frame     
        self.frequency = frequency
        self.exposure = exposure
        self.duty_cycle = duty_cycle
        self.pattern_id = pattern_id