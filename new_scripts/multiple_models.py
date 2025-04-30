from new_scripts.train_toy import TrainToy, learn
import os
import numpy as np

def multiple_models(dataset):
    save_folder_one = f"synth/multiple_models_model_small/{dataset}"
    save_folder_two = f"synth/multiple_models_model_large/{dataset}"

    if not os.path.exists(save_folder_one): os.makedirs(save_folder_one)
    if not os.path.exists(save_folder_two): os.makedirs(save_folder_two)

    synth_func_1 = learn(synth=[], dataset=dataset, size=30000, model_var_type="fixed-small")
    synth_func_2 = learn(synth=[], dataset=dataset, size=30000, model_var_type="fixed-large")

    data_to_save_1 = synth_func_1(n=10000)
    data_to_save_2 = synth_func_2(n=10000)

    save_path_1 = f"{save_folder_one}/{0}.txt"
    save_path_2 = f"{save_folder_two}/{0}.txt"

    np.savetxt(save_path_1, data_to_save_1, delimiter=',')
    np.savetxt(save_path_2, data_to_save_2, delimiter=',')

    for i in range(1, 10):

        synth = np.concatenate((synth_func_1(n=15000), synth_func_2(n=15000)), axis=0)
        synth_func_1 = learn(synth=synth, dataset=dataset, model_var_type="fixed-small")
        synth_func_2 = learn(synth=synth, dataset=dataset, model_var_type="fixed-large")

        data_to_save_1 = synth_func_1(n=10000)
        data_to_save_2 = synth_func_2(n=10000)

        save_path_1 = f"{save_folder_one}/{i}.txt"
        save_path_2 = f"{save_folder_two}/{i}.txt"

        np.savetxt(save_path_1, data_to_save_1, delimiter=',')
        np.savetxt(save_path_2, data_to_save_2, delimiter=',')

def single_model(dataset, model_var_type):
    save_folder = f"synth/single_model_{model_var_type}/{dataset}"
    if not os.path.exists(save_folder): os.makedirs(save_folder)

    synth_func = learn(synth=[], dataset=dataset, size=30000, model_var_type=model_var_type)

    data_to_save = synth_func(n=10000)
    save_path = f"{save_folder}/{0}.txt"
    np.savetxt(save_path, data_to_save, delimiter=',')

    for i in range(1, 10):
        
        synth = synth_func(n=30000)
        synth_func = learn(synth=synth, dataset=dataset, model_var_type=model_var_type)

        data_to_save = synth_func(n=10000)
        save_path = f"{save_folder}/{i}.txt"
        np.savetxt(save_path, data_to_save, delimiter=',')

    

def main():
    # multiple_models("gaussian8")
    # single_model("gaussian8", "fixed-small")
    # single_model("gaussian8", "fixed-large")

    print("poisson_glm_multiple_models")
    multiple_models("poisson_glm")
    print("poisson_glm_single_model fixed-small")
    single_model("poisson_glm", "fixed-small")
    print("poisson_glm_single_model fixed-large")
    single_model("poisson_glm", "fixed-large")

    print("creditcard_multiple_models")
    multiple_models("creditcard")
    print("creditcard_single_model fixed-small")
    single_model("creditcard", "fixed-small")
    print("creditcard_single_model fixed-large")
    single_model("creditcard", "fixed-large")