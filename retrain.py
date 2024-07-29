import os
from train_toy import TrainToy
from pathlib import Path
import numpy as np
from tqdm import tqdm

def learn(training_iterations, l, synth, model):
    train_toy = TrainToy()
    train_toy.set_data(l=l, retraining_iteration=training_iterations, synth=synth)
    train_toy.set_parameters()
    train_toy.set_checkpoint_path()
    train_toy.set_image_directory()
    train_toy.set_scheduler()
    synth, model = train_toy.train(model)
    return synth, model

def Algorithm(training_iterations, l):
    path_to_image = r"images\train\gaussian8\10.jpg"
    path_to_folder = r"images\train\experiment"
    print("Training on Real Data")
    synth, model = learn(0, l, [], None)
    print(len(synth))
    synth = synth[:int(np.floor(l * len(synth)))]
    for t in tqdm(range(training_iterations)):
        synth, model = learn(t, l, synth, None)
        Path(path_to_image).rename(os.path.join(path_to_folder, str(t) + ".jpg"))
        

def main():
    training_iterations = 20
    l = 1
    Algorithm(training_iterations, l)

if __name__ == "__main__":
    main()



