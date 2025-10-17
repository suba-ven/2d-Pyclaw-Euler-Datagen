import numpy as np
import os
from pathos.multiprocessing import ProcessingPool as Pool
import h5py
from tqdm import trange
from tensorboardX import SummaryWriter
from tqdm.auto import trange, tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


from Euler import ShockBubbleInteractionVarEst
from SBIConfig import SBIWarmStartDataConfig
from Utils import generate_description

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WarmStartDataGen():
    def __init__(self, config: SBIWarmStartDataConfig):
        """ Data generator for the shock bubble interaction problem.

        Args:
            config (SBIWarmStartDataConfig): Data Generator configuration.
        """
        self.state_param_list = config.state_param_list
        self.state_param_bounds = config.state_param_bounds
        self.model = ShockBubbleInteractionVarEst(config, config.state_param_list)
        
        self.n_obs_frames = config.n_steps - config.obs_start_idx + 1
        self.nx = config.nx
        self.ny = config.ny
        self.density = config.density
        self.description = generate_description(config)
        
        
    def generate_data(self, run_number, num_samples, multiprocessing):
        print('density = ', self.density)
        data_dir = f"Data/WarmStart/run_{run_number}"
        os.makedirs(data_dir, exist_ok=False)
        
        sample_states = np.random.uniform(
            low=self.state_param_bounds[:, 0],
            high=self.state_param_bounds[:, 1],
            size=(num_samples, len(self.state_param_list))
        )
        
        if multiprocessing:
            print("cpu count:", os.cpu_count())
            def generate_data_helper(i_par):
                claw_frames = self.model.forward_model(sample_states[i_par])
                return self.model.observation_operator(claw_frames, density=self.density)

            with Pool(nodes=os.cpu_count()) as pool:
                observations = list(tqdm(
                    pool.imap(generate_data_helper, range(num_samples)),
                    total=num_samples,
                    desc="Generating data (mp)"
                ))
            observations = np.array(observations)
 
        else:
            observations = np.zeros((num_samples, self.n_obs_frames, self.nx, self.ny))
            for i_par in trange(num_samples):
                claw_frames = self.model.forward_model(sample_states[i_par])
                observations[i_par] = self.model.observation_operator(claw_frames, density=self.density)
                
        filename = data_dir + f"/warm_start_data_{num_samples:.0e}_run_{run_number}.h5"
        filename = filename.replace("+0", "")
        
        description = self.description + "\n" + f"  Run number: {run_number}\n" + f"Num samples: {num_samples}\n"
        
        with h5py.File(filename, "w") as f:
            f.create_dataset("observations", data=observations)
            f.create_dataset("input_state", data=sample_states)
            f.attrs['description'] = description
            f.attrs['state_param_list'] = self.state_param_list
            f.attrs['state_param_bounds'] = self.state_param_bounds
            f.attrs['n_obs_frames'] = self.n_obs_frames
            f.attrs['nx'] = self.nx
            f.attrs['ny'] = self.ny
            
        # Saving the description to a text file
        description_file = filename.replace(".h5", "_description.txt")
        with open(description_file, "w") as f:
            f.write(description)
        
        print(f"Data saved to {filename}")


class WarmStartDataSet(Dataset):
    def __init__(self, filename):
        super().__init__()
        with h5py.File(filename, "r") as f:
            self.observations = f["observations"][:]
            self.input_state = f["input_state"][:]
            self.description = f.attrs['description']
            self.state_param_list = f.attrs['state_param_list']
            
        self.observations = torch.tensor(self.observations, dtype=torch.float32)
        self.input_state = torch.tensor(self.input_state, dtype=torch.float32)
        
        self.state_param_list = self.state_param_list.tolist()
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return {
            "observations": self.observations[idx],
            "input_state": self.input_state[idx]
        }



