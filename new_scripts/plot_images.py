from new_scripts.precision_recall import knn_precision_recall_features
from ddpm_torch.toy.toy_data import Gaussian8, PoissonGLM, CreditCardData
import numpy as np
from scipy.stats import wasserstein_distance_nd
import os
from matplotlib import pyplot as plt
from tqdm import tqdm


def main():
    top_most_path = "synth"
    name_class = {
        "gaussian8": Gaussian8,
        "poisson_glm": PoissonGLM,
        "creditcard": CreditCardData
    }
    for folder in os.listdir(top_most_path):
        experiment_path = os.path.join(top_most_path, folder)
        for dataset in tqdm(os.listdir(experiment_path)):
            if dataset == "creditcard": continue
            iteration_path = os.path.join(experiment_path, dataset)
            for iteration in tqdm(os.listdir(iteration_path)):
                data = np.loadtxt(os.path.join(iteration_path, iteration), delimiter=",", dtype=np.float32)

                plt.figure(figsize=(6, 6))
                plt.hist2d(data.data.T[0], data.data.T[1], bins=250, cmap="magma")
                plt.xlim([-2, 2])
                plt.tight_layout()
                plt.savefig(f"images/{folder}_{dataset}_{iteration}.jpg")