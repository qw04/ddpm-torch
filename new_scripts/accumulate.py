from new_scripts.train_toy import TrainToy, learn
import numpy as np
from random import randint
import os

def accumulate(dataset):

    save_folder = f"synth/accumulate/{dataset}"
    if not os.path.exists(save_folder): os.makedirs(save_folder)

    synth_func = learn(synth=[], dataset=dataset, size=30000)
    synth_funcs = [synth_func]

    data_to_save = synth_func(n=10000)
    save_path = f"{save_folder}/{0}.txt"
    np.savetxt(save_path, data_to_save, delimiter=',')

    for i in range(1, 10):
        
        synth = np.empty(shape=(0, 2)) if dataset != "creditcard" else np.empty(shape=(0, 31))

        for func in synth_funcs:
            synth_data = func(n=30000)
            synth = np.concatenate((synth, synth_data), axis=0)
        synth = synth.astype(np.float32)
        
        synth_func = learn(synth=synth, accumulate=True, dataset=dataset)
        synth_funcs.append(synth_func)

        data_to_save = synth_func(n=10000)
        save_path = f"{save_folder}/{i}.txt"
        np.savetxt(save_path, data_to_save, delimiter=',')


def replace(dataset):
    save_folder = f"synth/replace/{dataset}"
    if not os.path.exists(save_folder): os.makedirs(save_folder)

    synth_func = learn(synth=[], dataset=dataset, size=30000)

    data_to_save = synth_func(n=10000)
    save_path = f"{save_folder}/{0}.txt"
    np.savetxt(save_path, data_to_save, delimiter=',')

    for i in range(1, 10):
        
        synth = synth_func(n=30000)
        synth_func = learn(synth=synth, dataset=dataset)

        data_to_save = synth_func(n=10000)
        save_path = f"{save_folder}/{i}.txt"
        np.savetxt(save_path, data_to_save, delimiter=',')

def accumulate_fixed(dataset):
    save_folder = f"synth/accumulate_fixed/{dataset}"
    if not os.path.exists(save_folder): os.makedirs(save_folder)

    synth_func = learn(synth=[], dataset=dataset, size=30000)
    synth_funcs = [synth_func]

    data_to_save = synth_func(n=10000)
    save_path = f"{save_folder}/{0}.txt"
    np.savetxt(save_path, data_to_save, delimiter=',')

    for i in range(1, 10):
        
        total_samples_per_model = 30000 // (len(synth_funcs) + 1)
        
        synth = np.empty(shape=(0, 2)) if dataset != "creditcard" else np.empty(shape=(0, 31))
        for func in synth_funcs:
            synth_data = func(n=total_samples_per_model)
            synth = np.concatenate((synth, synth_data), axis=0)
        synth = synth.astype(np.float32)
        
        synth_func = learn(synth=synth, dataset=dataset)
        synth_funcs.append(synth_func)

        data_to_save = synth_func(n=10000)
        save_path = f"{save_folder}/{i}.txt"
        np.savetxt(save_path, data_to_save, delimiter=',')


def main():
    # accumulate("gaussian8")
    # accumulate("poisson_glm")
    # accumulate("creditcard")

    # replace("gaussian8")
    # replace("poisson_glm")
    # replace("creditcard")

    # accumulate_fixed("gaussian8")
    # accumulate_fixed("poisson_glm")
    # accumulate_fixed("creditcard")
    pass

if __name__ == "__main__":
    main()