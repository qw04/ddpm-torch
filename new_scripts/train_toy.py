import math
import numpy as np
import os
import torch
from ddpm_torch.toy import *
from ddpm_torch.utils import seed_all, infer_range
from torch.optim import Adam, lr_scheduler
from random import randint
from argparse import ArgumentParser
from functools import partial

class TrainToy:
    def __init__(self, **kwargs):
        parser = ArgumentParser()
        
        # neural network parameters
        parser.add_argument("--dataset", choices=["gaussian8", "poisson_glm", "creditcard"], default="gaussian8") # changes here
        parser.add_argument("--size", default=30000, type=int)
        parser.add_argument("--root", default="~/datasets", type=str, help="root directory of datasets")
        parser.add_argument("--epochs", default=100, type=int, help="total number of training epochs")
        parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
        parser.add_argument("--beta1", default=0.9, type=float, help="beta_1 in Adam")
        parser.add_argument("--beta2", default=0.999, type=float, help="beta_2 in Adam")
        parser.add_argument("--lr-warmup", default=0, type=int, help="number of warming-up epochs")
        parser.add_argument("--batch-size", default=200, type=int)
        
        # ddpm parameters
        parser.add_argument("--timesteps", default=200, type=int, help="number of diffusion steps")
        parser.add_argument("--beta-schedule", choices=["quad", "linear", "warmup10", "warmup50", "jsd"], default="linear")
        parser.add_argument("--beta-start", default=0.001, type=float)
        parser.add_argument("--beta-end", default=0.2, type=float)
        parser.add_argument("--model-mean-type", choices=["mean", "x_0", "eps"], default="mean", type=str)
        parser.add_argument("--model-var-type", choices=["learned", "fixed-small", "fixed-large"], default="fixed-small", type=str)  # noqa
        parser.add_argument("--loss-type", choices=["kl", "mse"], default="kl", type=str)

        # saving parameters
        parser.add_argument("--image-dir", default="./images/train", type=str)
        parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
        parser.add_argument("--chkpt-intv", default=10000, type=int, help="frequency of saving a checkpoint")
        parser.add_argument("--eval-intv", default=10000, type=int)
        parser.add_argument("--seed", default=1234, type=int, help="random seed")
        parser.add_argument("--resume", action="store_true", default=False, help="to resume training from a checkpoint")
        parser.add_argument("--device", default="cpu", type=str) #default="cuda:0"
        parser.add_argument("--mid-features", default=128, type=int)
        parser.add_argument("--num-temporal-layers", default=3, type=int)

        self.args = parser.parse_args()
        for key, value in kwargs.items():
            setattr(self.args, key, value)

        if self.args.model_var_type == "learned": self.args.loss_type = "kl"    

        seed_all(self.args.seed)
        self.in_features = 2 if self.args.dataset != "creditcard" else 31
        self.root = os.path.expanduser(self.args.root)
    
    def set_data(self, synth, accumulate=False):
        if accumulate:
            self.num_batches = (self.args.size + len(synth)) // self.args.batch_size
        else:
            self.num_batches = self.args.size // self.args.batch_size
        # synth has correct size according to the synthetic ratio
        self.trainloader = DataStreamer(
            dataset=self.args.dataset, batch_size=self.args.batch_size, num_batches=self.num_batches, 
            synth=synth, accumulate=accumulate
            )
            
    def set_parameters(self, model = None):
        self.device = torch.device(self.args.device)
        
        # Gaussian Diffusion Parameters
        self.betas = get_beta_schedule(self.args.beta_schedule, beta_start=self.args.beta_start, beta_end=self.args.beta_end, timesteps=self.args.timesteps)
        self.diffusion = GaussianDiffusion(betas=self.betas, model_mean_type=self.args.model_mean_type, model_var_type=self.args.model_var_type, loss_type=self.args.loss_type)

        # neural network parameters
        self.out_features = 2 * self.in_features if self.args.model_var_type == "learned" else self.in_features
        if model is None: self.model = Decoder(self.in_features, self.args.mid_features, self.out_features, self.args.num_temporal_layers)
        else: self.model = model
        self.model.to(self.device)

        # optimizer
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))

    def set_checkpoint_path(self, **kwargs):
        # checkpoint path
        if not os.path.exists(self.args.chkpt_dir):
            os.makedirs(self.args.chkpt_dir)
        self.chkpt_path = os.path.join(self.args.chkpt_dir, "ddpm__" + "__".join([f"{key}_{kwargs[key]}" for key in kwargs.keys()]) + ".pt")
        return self.chkpt_path

    def set_image_directory(self):
        # set up image directory
        self.image_dir = os.path.join(self.args.image_dir, f"{self.args.dataset}")
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def set_scheduler(self): # scheduler
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda t: min((t + 1) /self.args.lr_warmup, 1.0)) if self.args.lr_warmup > 0 else None

    def train(self, checkpoint_path_load, bins=100, cmap="Blues"):
        # load trainer
        grad_norm = 0  # gradient global clipping is disabled
        eval_intv = self.args.eval_intv
        
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            diffusion=self.diffusion,
            epochs=self.args.epochs,
            trainloader=self.trainloader,
            scheduler=self.scheduler,
            grad_norm=grad_norm,
            device=self.device,
            eval_intv=self.args.eval_intv,
            chkpt_intv=self.args.chkpt_intv
        )

        # load evaluator
        max_eval_count = min(self.args.size, 30000)
        eval_batch_size = min(max_eval_count, 30000)
        xlim, ylim = infer_range(self.trainloader.dataset)
        value_range = (xlim, ylim)
        true_data = iter(self.trainloader)
        
        # evaluator = Evaluator(
        #     true_data=np.concatenate([next(true_data) for _ in range(max_eval_count // eval_batch_size)]),
        #     eval_batch_size=eval_batch_size, 
        #     max_eval_count=max_eval_count, 
        #     value_range=value_range
        # )

        if self.args.resume:
            try:
                if checkpoint_path_load is None: raise FileNotFoundError
                trainer.load_checkpoint(checkpoint_path_load)
                print(f"Resume traing : {self.args.resume} from {checkpoint_path_load}")
            except FileNotFoundError:
                print("Checkpoint file does not exist!")
                print("Starting from scratch...")

        trainer.train(evaluator=None, chkpt_path=self.chkpt_path, image_dir=self.image_dir, bins=bins, cmap=cmap)
        
        
        def sample_fn(n, diffusion, model, shape_, device):
            if isinstance(n, float): n = int(n// 1) 
            shape = (n, ) + shape_
            sample = diffusion.p_sample(denoise_fn=model, shape=shape, device=device, seed = randint(0, 1000))
            synth = sample.cpu().numpy()
            return synth

        return partial(sample_fn, diffusion=self.diffusion, model=self.model, shape_ = trainer.shape, device=self.device)

def learn(synth=[], accumulate=False, **kwards):
    BINS = 250
    CMAP = "magma"

    train_toy = TrainToy(**kwards)
    train_toy.set_data(synth=synth, accumulate=accumulate)
    train_toy.set_parameters()
    train_toy.set_checkpoint_path()
    train_toy.set_image_directory()
    train_toy.set_scheduler()
    synth = train_toy.train(BINS, CMAP)
    return synth

if __name__ == "__main__":
    learn()