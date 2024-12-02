from datetime import datetime
import os
from train_toy import TrainToy
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import math
from scipy.stats import wasserstein_distance_nd
import traceback


def one_iteration(l, synth, model_var_type, dataset, batch_size = 100, timesteps = 100):
    BINS = 250
    CMAP = "magma"
    train_toy = TrainToy(batch_size=batch_size, timesteps=timesteps, model_var_type=model_var_type, dataset=dataset)
    train_toy.set_data(l, synth)
    train_toy.set_parameters()
    train_toy.set_checkpoint_path(l, dataset)
    train_toy.set_image_directory()
    train_toy.set_scheduler()
    synth = train_toy.train(l, BINS, CMAP)
    return synth

def many_iterations(training_iterations, l, path_to_save, path_to_image, model_var_type, dataset, batch_size=200, timesteps=200):
    if not os.path.exists(path_to_save): 
        os.makedirs(path_to_save)
    synth_path = os.path.join("synth/", path_to_save)
    if not os.path.exists(synth_path):
        os.makedirs(synth_path)
    size = 30000 # should match up with value in train_toy.py
    synth = one_iteration(l, [], model_var_type, dataset, batch_size, timesteps)
    print(len(synth))
    np.savetxt(rf"{synth_path}/0.txt", synth)
    synth = synth[:int(np.floor(l * size))]
    Image.open(path_to_image).save(rf"{path_to_save}/0.jpg")    
    for t in tqdm(range(training_iterations)):
        print(f"iteration {t} for dataset {dataset} and l value {l} at time {datetime.now()}")
        synth = one_iteration(l, synth, model_var_type, dataset, batch_size, timesteps)
        np.savetxt(rf"{synth_path}/{str(t+1)}.txt", synth)
        synth = synth[:int(np.floor(l * size))]
        Image.open(path_to_image).save(rf"{path_to_save}/{str(t+1)}.jpg")

def main():
    datasets = ["gaussian8", "gaussian25", "swissroll", "gaussian2", "gaussian1", "twomoons"]
    model_var_type = "learned"
    l = [0.0, 0.25, 0.5, 0.75, 1.0]
    training_iterations = 10
    path_to_save = lambda dataset, l_val: f"images/train/experiment12/{dataset}_{l_val}/"
    path_to_image = lambda dataset: f"/dcs/22/u2211900/ddpm-torch/images/train/{dataset}/50.jpg" #this should match with number of epochs in train_toy.py
    for dataset in tqdm(datasets):
        for l_val in tqdm(l):
            try:
                many_iterations(training_iterations, l_val, path_to_save(dataset, l_val), path_to_image(dataset), model_var_type, dataset)    
            except Exception as e:
                print(f"error in dataset {dataset} and l value {l_val}")
                traceback.print_exc()

# compare wasserstein distance between real and synthetic data

if __name__ == "__main__":
    # one_iteration(0.5, [], "learned", "gaussian8")
    main()

