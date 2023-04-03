import numpy as np
from habitat_sim.sensors.noise_models.redwood_depth_noise_model import RedwoodDepthNoiseModel
from habitat_sim.sensors.noise_models.gaussian_noise_model import GaussianNoiseModel

class RGBNoise:
    def __init__(self, intensity_constant):
        self.rgb_noise = GaussianNoiseModel(intensity_constant=intensity_constant)
    
    def __call__(self, img_noiseless: np.array) -> np.array:
        return self.rgb_noise.apply(img_noiseless) 
        
class DepthNoise:
    def __init__(self, noise_multiplier):
        self.depth_noise = RedwoodDepthNoiseModel(noise_multiplier=noise_multiplier)
    
    def __call__(self, img_noiseless: np.array) -> np.array:
        return self.depth_noise.apply(img_noiseless) 
        