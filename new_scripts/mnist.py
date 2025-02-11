from datetime import datetime
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import math
from scipy.stats import wasserstein_distance_nd
import traceback
from old_scripts.train import main as learn


def Algorithm(training_iterations, l, path):
    path_to_image = lambda x: rf"/dcs/22/u2211900/ddpm-torch/images/train/mnist/{x}.jpg"
    

    if not os.path.exists(path):
        os.makedirs(path)

    EPOCHS = 200
    print(os.path.exists(path))
    print("Training on Real Data")
    synth = learn(None, 0, epochs=EPOCHS, chkpt_intv=EPOCHS, resume=True, chkpt_name=f"mnist_{EPOCHS}_{l}.pt", chkpt_path=f"chkpts/mnist/mnist_{EPOCHS}_{l}.pt")
    Image.open(path_to_image(EPOCHS)).save(rf"{path}/0.jpg")

    for t in tqdm(range(training_iterations)):
        EPOCHS += 50
        if os.path.exists(f"chkpts/mnist/mnist_{EPOCHS}_{l}.pt"): chkpt_path = f"chkpts/mnist/mnist_{EPOCHS}_{l}.pt"
        else: chkpt_path = f"chkpts/mnist/mnist_{EPOCHS-50}_{l}.pt"

        synth = learn(synth, l, epochs=EPOCHS, chkpt_intv=EPOCHS, resume=True, chkpt_path=chkpt_path, chkpt_name=f"mnist_{EPOCHS}_{l}.pt")
        Image.open(path_to_image(EPOCHS)).save(rf"{path}/{str(t + 1)}.jpg")

def main():
    Algorithm(training_iterations=20, l=1.0, path="images/train/experiment13/full/")
    Algorithm(training_iterations=20, l=0.5, path="images/train/experiment13/half/")
