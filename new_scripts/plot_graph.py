from new_scripts.precision_recall import knn_precision_recall_features
from ddpm_torch.toy.toy_data import Gaussian8, PoissonGLM, CreditCardData
import numpy as np
from scipy.stats import wasserstein_distance_nd
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

# wasserstein_distance = wasserstein_distance_nd(data.data, synth)

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
            real_data = name_class[dataset](2000).data
            file_path = os.path.join(experiment_path, dataset)
            
            precision_list = []
            recall_list = []
            wasserstein_distance_list = []
            iteration_list = []
            try:
                for file in tqdm(os.listdir(file_path)):
                    generated_data = np.loadtxt(os.path.join(file_path, file), delimiter=",", dtype=np.float32)[:2000]
                    state = knn_precision_recall_features(real_data, generated_data)
                    wasserstein_distance = wasserstein_distance_nd(real_data[:500], generated_data[:500])
                    precision, recall = state["precision"], state["recall"]
                    iteration = file[0]

                    precision_list.append(precision)
                    recall_list.append(recall)
                    wasserstein_distance_list.append(wasserstein_distance)
                    iteration_list.append(iteration)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
            
            temp = sorted(zip(iteration_list, precision_list, recall_list, wasserstein_distance_list), key=lambda x: int(x[0]))
            iteration_list = [temp[i][0] for i in range(len(temp))]
            precision_list = [temp[i][1] for i in range(len(temp))]
            recall_list = [temp[i][2] for i in range(len(temp))]
            wasserstein_distance_list = [temp[i][3] for i in range(len(temp))]

            plt.scatter(iteration_list, wasserstein_distance_list)
            plt.xlabel("Iterations")
            plt.ylabel("Wasserstein Distance")
            plt.title(f"Wasserstein Distance vs Iterations")
            plt.savefig(rf"plots/{folder}_{dataset}_wasser.png")
            plt.close()

            plt.scatter(iteration_list, precision_list,)
            plt.xlabel("Iterations") 
            plt.ylabel("Precision")
            plt.title(f"Precision vs Iterations")
            plt.savefig(rf"plots/{folder}_{dataset}_precision.png")
            plt.close()

            
            plt.scatter(iteration_list, recall_list)
            plt.xlabel("Iterations")
            plt.ylabel("Recall")
            plt.title(f"Recall vs Iterations")
            plt.savefig(rf"plots/{folder}_{dataset}_recall.png")
            plt.close()

