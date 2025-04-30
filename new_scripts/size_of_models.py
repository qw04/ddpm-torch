from new_scripts.train_toy import TrainToy, learn
import os
import numpy as np


def timestep(dataset, timesteps, timestep_increment):
    save_folder = f"synth/model_{timesteps}_{timestep_increment}/{dataset}"
    if not os.path.exists(save_folder): os.makedirs(save_folder)

    synth_func = learn(synth=[], dataset=dataset, size=30000, timesteps=timesteps)

    data_to_save = synth_func(n=10000)
    save_path = f"{save_folder}/{0}.txt"
    np.savetxt(save_path, data_to_save, delimiter=',')

    for i in range(1, 10):
        
        timesteps += timestep_increment

        synth = synth_func(n=30000)
        synth_func = learn(synth=synth, dataset=dataset, timesteps=timesteps)

        data_to_save = synth_func(n=10000)
        save_path = f"{save_folder}/{i}.txt"
        np.savetxt(save_path, data_to_save, delimiter=',')

def main():
    # timestep("gaussian8", 200, 0)
    # timestep("gaussian8", 1000, 0)
    # timestep("gaussian8", 200, 90)

    # timestep("poisson_glm", 200, 0)
    # timestep("poisson_glm", 1000, 0)
    # timestep("poisson_glm", 200, 90)

    print("creditcard")
    print("200 timesteps")
    timestep("creditcard", 200, 0)
    print("1000 timesteps")
    timestep("creditcard", 1000, 0)
    print("200 timesteps with increment of 90")
    timestep("creditcard", 200, 90)