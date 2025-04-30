import math
import numpy as np
import torch
import pandas as pd
from sklearn.datasets import make_swiss_roll, make_moons
from torch.utils.data import Dataset


__all__ = ["Gaussian8", "PoissonGLM", "CreditCardData", "DataStreamer"]


class ToyDataset(Dataset):
    def __init__(self, size: int, random_state: int = None, synth = []):
        self.size = size
        self.random_state = random_state
        
        if len(synth) == 0: self.data = self._sample()
        else: 
            len_synth = len(synth)
            len_sample = self.size

            choices = np.concatenate([np.zeros(len_synth, dtype=int), np.ones(len_sample, dtype=int)])
            np.random.shuffle(choices)
            
            synth_idx = np.where(choices == 0)[0]
            sample_idx = np.where(choices == 1)[0]
            output = np.empty((len_synth + len_sample, 2)) if self.name != "creditcard" else np.empty((len_synth + len_sample, 31))

            output[synth_idx] = np.array(synth)
            output[sample_idx] = self._sample()
            
            self.data = np.array(output, dtype=np.float32)
    
    def _sample(self):
        pass

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])       
    

class Gaussian8(ToyDataset):
    scale = 2
    modes = [
        (math.cos(0.25 * t * math.pi), math.sin(0.25 * t * math.pi))
        for t in range(8)
    ]  # scale x (8 roots of z^8 = 1)

    def __init__(self, size, random_state=1234, synth = [], accumulate=False):
        self.name = "gaussian8"
        self.modes = self.scale * np.array(self.modes, dtype=np.float32)
        self.noise = 0.1
        if len(synth) == 0: super(Gaussian8, self).__init__(size, random_state, synth)
        elif accumulate: super(Gaussian8, self).__init__(size, random_state, synth)
        else: super(Gaussian8, self).__init__(size - len(synth), random_state, synth)
    
    def _sample(self):
        rng = np.random.default_rng(seed=self.random_state)
        data = self.noise * rng.standard_normal((self.size, 2), dtype=np.float32)
        data += np.array(self.modes)[
            np.random.choice(np.arange(8), size=self.size, replace=True)]
        stdev = math.sqrt(self.noise ** 2 + (self.scale ** 2) * 0.5)
        data /= stdev
        return data

class PoissonGLM(ToyDataset):

    def __init__(self, size, random_state=1234, synth=[], accumulate=False):
        self.name = "poisson_glm"
        self.a = 0.5 * (np.log(10) - np.log(5))
        self.b = 0.5 * (np.log(10) + np.log(5))
        if len(synth) == 0: super(PoissonGLM, self).__init__(size, random_state, synth)
        elif accumulate: super(PoissonGLM, self).__init__(size, random_state, synth)
        else: super(PoissonGLM, self).__init__(size - len(synth), random_state, synth)
        

    def _sample(self):
        if self.size == 0: return np.empty((0, 2), dtype=np.float32)
        
        self.coef = np.asarray([self.a for _ in range(self.size)])
        self.intercept = np.asarray([self.b for _ in range(self.size)])

        rng = np.random.default_rng(seed=self.random_state)
        values = np.arange(-1.0, 1.1, 0.1, dtype=np.float32)
        X = rng.choice(values, size=(self.size,), replace=True)
        
        eta = self.intercept + (self.coef * X)
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-5, 1e5)
        
        y = rng.poisson(mu, size=(self.size, )).astype(np.float32)
        return np.column_stack((X, y))


class CreditCardData(ToyDataset):
    path = "ddpm_torch/toy/creditcard.csv"

    def __init__(self, size, random_state=1234, synth=[], accumulate=False):
        self.name = "creditcard"
        df = pd.read_csv(self.path, on_bad_lines='skip')
        self.real_data = df.to_numpy()
        if len(synth) == 0: super(CreditCardData, self).__init__(size, random_state, synth)
        elif accumulate: super(CreditCardData, self).__init__(size, random_state, synth)
        else: super(CreditCardData, self).__init__(size - len(synth), random_state, synth)

    def _sample(self):
        if self.size == 0: return np.empty((0, 31), dtype=np.float32)
        if self.size >= len(self.real_data): # sample with replacement
            idx = np.random.choice(len(self.real_data), size=self.size, replace=True)
        else: # sample without replacement
            idx = np.random.choice(len(self.real_data), size=self.size, replace=False)
        self.data = self.real_data[idx]
        return self.data.astype(np.float32)



class DataStreamer:

    def __init__(self, dataset: ToyDataset, batch_size: int, num_batches: int, synth = [], accumulate=False):
        dataset = self.dataset_map(dataset)
        self.batch_size = batch_size
        self.num_batches = num_batches
        
        if len(synth) == 0:
            self.dataset = dataset(batch_size * num_batches, synth = synth, random_state=None, accumulate=accumulate)
        else:
            self.dataset = dataset(batch_size * num_batches, random_state=None, synth = synth, accumulate=accumulate)

    def __iter__(self):
        cnt = 0
        while True:
            start = cnt * self.batch_size
            end = start + self.batch_size
            yield torch.from_numpy(self.dataset.data[start:end])
            cnt += 1
            if cnt >= self.num_batches:
                break


    def __len__(self):
        return self.num_batches
        
    @staticmethod
    def dataset_map(dataset):
        return {
            "gaussian8": Gaussian8,
            "poisson_glm": PoissonGLM,
            "creditcard": CreditCardData
        }.get(dataset, None)



if __name__ == "__main__":
    import os
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    mpl.rcParams["figure.dpi"] = 144

    fig_dir = "./figs"
    if not os.path.exists(fig_dir): os.makedirs(fig_dir)

    size = 100000

    DATASET = {
            "gaussian8": Gaussian8,
            "poisson_glm": PoissonGLM
    }

    for name, dataset in DATASET.items():
        data = dataset(size)
        plt.figure(figsize=(6, 6))
        print(f"Dataset: {name}")
        print(data.data[0:10])
        plt.hist2d(data.data.T[0], data.data.T[1], bins=250, cmap="magma")
        plt.xlim([-2, 2])
        # plt.ylim([-2, 2])
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{name}.jpg"))
        dataloader = DataLoader(data)
        x = next(iter(dataloader))

    # change path to csv in CreditCardData before running commented out code
    # print("Credit Card Data Test")
    # data = CreditCardData(size)
    # print(data.data[0:10])
    # data = DataLoader(data)
    # x = next(iter(data))
    # print(x)
