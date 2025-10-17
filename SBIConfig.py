import numpy as np

class SBIConfig():
    def __init__(self):
        self.t_final = 0.8 # was 0.5
        self.n_steps = 101  # was 10
        
        self.gamma = 1.4
        self.x0 = 0.5
        self.y0 = 0.
        self.r0 = 0.2
        
        self.pinf = 5.
        
        self.rhoout = 1.
        self.rhoin = 0.1
        self.pout = 1.
        self.pin = 1.
        self.xshock = 0.2
        self.density = False
        self.x_domain = [0., 2.]
        self.y_domain = [0., 0.5]
        self.nx = 100
        self.ny = 25
        
        self.threshold = 0.2
        self.obs_start_idx = 2
        
        self.state_param_list = ['x0', 'xshock', 'r0', 'rhoin', 'pin']
        
        self.diff_cost_coeff = 1.0
        self.gaussian_filter_sigma = [0, 2, 2]
        
    def set_density(self,new_density):
        self.density = new_density
        
class SBIWarmStartDataConfig(SBIConfig):
    def __init__(self):
        super().__init__()
        self.state_param_list = [
            'x0',
            'r0',
            'rhoin',
            'rhoout',
            'pin'
        ]
        self.state_param_bounds = np.array([
            [0.3, 1.], # x0
            [0.1, 0.4], # r0
            [0.05, 0.2], # rhoin
            [1.5, 3], # rhoout
            [0.5, 2] # pin
        ])
        
        

        


        
        
