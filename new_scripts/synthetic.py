import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import math
import random


class SyntheticDataSampler:

    def __init__(self, dataset_name, chkpt_paths, schedule, total_iterations, current_iteration=0):
        self.dataset_name = dataset_name
        self.chkpt_paths = chkpt_paths
        self.chkpts = [self.load_checkpoint(i) for i in chkpt_paths]
        self.schedule = schedule
        self.current_iteration = current_iteration
        self.total_iterations = len(schedule)
    
    def load_checkpoint_from_path(self, chkpt_path, learn):
        self.chkpt_paths.append(chkpt_path)
        chkpt = learn(0, self, chkpt_path, epochs=0, resume=True)
        self.chkpts.append(chkpt)
    
    def load_checkpoint_from_partial(self, chkpt_path, partial):
        self.chkpts.append(partial)
        self.chkpt_paths.append(chkpt_path)
        
    def increment_iteration(self):
        self.current_iteration += 1
        if self.current_iteration >= self.total_iterations:
            return Exception("All iterations have been sampled, there is no specified scedule for the latest iteration")

    def sample(self, n):
        # sample a singular batch of synthetic data (that is what I am going to use this for)
        if len(self.schedule[self.current_iteration]) > len(self.chkpts):
            return Exception("The number of models that were scheduled to sample synthetic data have exceeded the number of models that are available")

        total = sum(self.schedule[self.current_iteration])
        data = []

        for i, chkpt in zip(self.schedule[self.current_iteration], self.chkpts):
            if i == 1:
                data.append(chkpt(int(math.floor(n / total))))

        left_over = total - len(data)
        if left_over > 0:
            chkpt = random.choice(self.chkpts)
            data.append(chkpt(n - len(data) * math.floor(n / total)))

        return np.concatenate(data, axis=0) if data != [] else np.empty((0, 2), dtype=np.float32)