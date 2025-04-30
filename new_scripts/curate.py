from new_scripts.train_toy import TrainToy, learn
import os
import numpy as np    
from ddpm_torch.toy.toy_data import Gaussian8, PoissonGLM
import random
import math
from scipy.stats import norm, poisson

def prune(dataset, cumulative_probability_function):
    if dataset == "credit_card": raise Exception("Credit card dataset is not supported for correction.")
    
    save_folder = f"synth/prune/{dataset}"
    if not os.path.exists(save_folder): os.makedirs(save_folder)

    synth_func = learn(synth=[], dataset=dataset, size=30000)
    data_to_save = synth_func(n=10000)
    save_path = f"{save_folder}/{0}.txt"
    np.savetxt(save_path, data_to_save, delimiter=',')

    for i in range(1, 10):
        
        synth = synth_func(n=30000)
        synth = list(filter(lambda x: cumulative_probability_function(x) <= 0.95, synth))
        synth = np.asarray(synth, dtype=np.float32)
        synth_func = learn(synth=synth, dataset=dataset)

        data_to_save = synth_func(n=10000)
        save_path = f"{save_folder}/{i}.txt"
        np.savetxt(save_path, data_to_save, delimiter=',')

def correct(dataset, cumulative_probability_function):
    if dataset == "credit_card": raise Exception("Credit card dataset is not supported for correction.")
    elif dataset == "gaussian8": samples = Gaussian8(5000).data
    elif dataset == "poisson_glm": samples = PoissonGLM(5000).data

    def correction_function(x):
        if cumulative_probability_function(x) > 0.95: 
            return samples[random.randint(0, 4999)]
        return x

    save_folder = f"synth/correct/{dataset}"
    if not os.path.exists(save_folder): os.makedirs(save_folder)

    synth_func = learn(synth=[], dataset=dataset, size=30000)

    data_to_save = synth_func(n=10000)
    save_path = f"{save_folder}/{0}.txt"
    np.savetxt(save_path, data_to_save, delimiter=',')

    for i in range(1, 10):
        
        synth = synth_func(n=30000)
        synth = list(map(correction_function, synth))
        synth = np.asarray(synth, dtype=np.float32)
        synth_func = learn(synth=synth, dataset=dataset)

        data_to_save = synth_func(n=10000)
        save_path = f"{save_folder}/{i}.txt"
        np.savetxt(save_path, data_to_save, delimiter=',')



# for any dataset whose distribution is known, the cumulative probability function can be used as the sorting function
def sort(dataset, cumulative_probability_function):
    if dataset == "credit_card": raise Exception("Credit card dataset is not supported for correction.")
    
    save_folder = f"synth/sort/{dataset}"
    if not os.path.exists(save_folder): os.makedirs(save_folder)

    synth_func = learn(synth=[], dataset=dataset, size=30000)

    data_to_save = synth_func(n=10000)
    save_path = f"{save_folder}/{0}.txt"
    np.savetxt(save_path, data_to_save, delimiter=',')

    for i in range(1, 10):
        
        synth = synth_func(n=30000)
        synth = sorted(synth, key=cumulative_probability_function)
        synth = np.asarray(synth, dtype=np.float32)
        synth_func = learn(synth=synth, dataset=dataset)

        data_to_save = synth_func(n=10000)
        save_path = f"{save_folder}/{i}.txt"
        np.savetxt(save_path, data_to_save, delimiter=',')



def centre_gaussian(x, y):
    modes = [(math.cos(0.25 * t * math.pi), math.sin(0.25 * t * math.pi)) for t in range(8)]
    dist = [(x - a) ** 2 + (y - b) ** 2 for (a, b) in modes]
    index = dist.index(min(dist))
    return modes[index]

def cumulative_gaussian(x):
    mean = centre_gaussian(x[0], x[1])
    stdev = 0.1
    return norm.cdf(x[0], loc=mean[0], scale=stdev) * norm.cdf(x[1], loc=mean[1], scale=stdev)
    
def points_poisson(x):
    points = [-1.0 + x * 0.1 for x in range(21)]
    dist = [abs(x-a) for a in points]
    index = dist.index(min(dist))
    return points[index]

def cumulative_poisson(x):
    true_x = points_poisson(x[0])
    coef = 0.5 * (np.log(10) - np.log(5))
    intercept = 0.5 * (np.log(10) + np.log(5))
    mu = np.exp(coef * true_x + intercept)
    _lambda = np.exp(mu)
    return poisson.cdf(x[1], mu=_lambda)


def main():
    # prune("gaussian8", cumulative_gaussian)
    # correct("gaussian8", cumulative_gaussian)
    # sort("gaussian8", cumulative_gaussian)

    # prune("poisson_glm", cumulative_poisson)
    # correct("poisson_glm", cumulative_poisson)
    # sort("poisson_glm", cumulative_poisson)
    pass


if __name__ == "__main__":
    main()


