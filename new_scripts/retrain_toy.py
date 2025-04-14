import os
from new_scripts.train_toy import TrainToy
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import math
from scipy.stats import wasserstein_distance_nd
from new_scripts.synthetic import SyntheticDataSampler


def learn(l, synth, chkpt_path_load, training_iterations, **kwargs):
    train_toy = TrainToy(**kwargs)
    train_toy.set_data(synth, l)
    train_toy.set_parameters()
    chkpt_path_save = train_toy.set_checkpoint_path(synthetic_ratio = l, training_iterations = training_iterations)
    train_toy.set_image_directory()
    train_toy.set_scheduler()
    synth = train_toy.train(l, checkpoint_path_load=chkpt_path_load, bins=250, cmap="magma")
    return synth, chkpt_path_save


def Algorithm(training_iterations, l, path):
    
    path_to_image = r"/dcs/22/u2211900/ddpm-torch/images/train/gaussian2/200.jpg"
    synth_paths = [f"synth\\{l}\\{i}.txt" for i in range(training_iterations+1)]


    if not os.path.exists(path):  os.makedirs(path)
    print(os.path.exists(path))

    print("Training on Real Data")
    synthetic_data_sampler = SyntheticDataSampler("gaussian8", [], [i*[1] for i in range(training_iterations+1)], training_iterations)
    synth_sampler, chkpt_path_save = learn(0.0, synthetic_data_sampler, None, 0, epochs=500)
    
    # save sampled data into synthpaths
    # file = open(synth_paths[0], "w")
    # file.close()
    # np.savetxt(synth_paths[0], synth_sampler(2000))

    # synthetic_data_sampler.load_checkpoint_from_partial(synth_paths[0], synth_sampler)
    
    # Image.open(path_to_image).save(rf"{path}/0.jpg")    
    
    # for t in tqdm(range(1, training_iterations)):
    #     synth_sampler, chkpt_path_save = learn(l, synthetic_data_sampler, chkpt_path_save, t, epochs=10)
    #     synthetic_data_sampler.increment_iteration()
    #     file = open(synth_paths[t], "w")
    #     file.close()
    #     np.savetxt(synth_paths[t], synth_sampler(2000))

    #     synthetic_data_sampler.load_checkpoint_from_partial(synth_paths[t+1], synth_sampler)
    #     Image.open(path_to_image).save(rf"{path}/{str(t+1)}.jpg")
        

def main():
    Algorithm(training_iterations=10, l=1.0, path="images/train/experiment11/") # shouldn't collapse


    # model_mean_type = eps
    # Algorithm(training_iterations=10, l=0.5, stdev=0.25, path="images/train/experiment1/", batch_size=200, timesteps=200, model_var_type="fixed-small") # shouldn't collapse
    # Algorithm(training_iterations=10, l=1.0, stdev=0.25, path="images/train/experiment2/", batch_size=200, timesteps=200, model_var_type="fixed-small") # shouldn't collapse

    # model_mean_type = eps
    # Algorithm(training_iterations=10, l=0.5, stdev=0.25, path="images/train/experiment3/", batch_size=200, timesteps=200, model_var_type="fixed-large") # shouldn't collapse
    # Algorithm(training_iterations=10, l=1.0, stdev=0.25, path="images/train/experiment4/", batch_size=200, timesteps=200, model_var_type="fixed-large") # should collapse

    # Algorithm(10, 0.5, "images/train/experiment5/")
    # Algorithm(10, 1, "images/train/experiment6/")

    # Algorithm(training_iterations=10, l=0.5, stdev=1.0, path="images/train/experiment5/", batch_size=200, timesteps=200, model_var_type="fixed-small") # shouldn't collapse
    # Algorithm(training_iterations=10, l=1.0, stdev=1.0, path="images/train/experiment6/", batch_size=200, timesteps=200, model_var_type="fixed-small") # shouldn't collapse

    # model_mean_type = eps
    # Algorithm(training_iterations=10, l=0.5, stdev=1.0, path="images/train/experiment7/", batch_size=200, timesteps=200, model_var_type="fixed-large") # shouldn't collapse
    # Algorithm(training_iterations=10, l=1.0, stdev=1.0, path="images/train/experiment8/", batch_size=200, timesteps=200, model_var_type="fixed-large") # should collapse
    
    # TRAINING_ITERATIONS = 10
    
    # for l in tqdm(range(0, 101, 25)):
    #     l = l/100
    #     for stdev in tqdm(range(25, 101, 25)):
    #         stdev = stdev/100
    #         path_small = f"images/train/experiment9/fixed_small/{l}_{stdev}"
    #         path_large = f"images/train/experiment9/fixed_large/{l}_{stdev}"
            # path_learned = f"images/train/experiment9/learned/{l}_{stdev}"
            # Algorithm(training_iterations=TRAINING_ITERATIONS, l=l, stdev=stdev, path=path_small, batch_size=200, timesteps=200, model_var_type="fixed-small")
            # Algorithm(training_iterations=TRAINING_ITERATIONS, l=l, stdev=stdev, path=path_large, batch_size=200, timesteps=200, model_var_type="fixed-large")
            # Algorithm(training_iterations=TRAINING_ITERATIONS, l=l, stdev=stdev, path=path_learned, batch_size=200, timesteps=200, model_var_type="learned")


    # path_small = lambda x: f"images/train/experiment10/fixed_small/{x}"
    # path_large = lambda x: f"images/train/experiment10/fixed_large/{x}"
    # Algorithm(training_iterations=25, l=1.0, stdev=1.0, path=path_small(1.0), batch_size=200, timesteps=200, model_var_type="fixed-small")
    # Algorithm(training_iterations=25, l=0.5, stdev=1.0, path=path_small(0.5), batch_size=200, timesteps=200, model_var_type="fixed-small")
    # Algorithm(training_iterations=25, l=1.0, stdev=1.0, path=path_large(1.0), batch_size=200, timesteps=200, model_var_type="fixed-large")
    # Algorithm(training_iterations=25, l=0.5, stdev=1.0, path=path_large(0.5), batch_size=200, timesteps=200, model_var_type="fixed-large")
    


if __name__ == "__main__":
    main()



# synth = synth[:int(np.floor(l * len(synth)))]