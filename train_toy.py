import numpy as np
import os
import torch
from ddpm_torch.toy import *
from ddpm_torch.utils import seed_all, infer_range
from torch.optim import Adam, lr_scheduler

from argparse import ArgumentParser

class TrainToy:
    def __init__(self):
        parser = ArgumentParser()

        parser.add_argument("--dataset", choices=["gaussian8", "gaussian25", "swissroll"], default="gaussian8")
        parser.add_argument("--size", default=100000, type=int)
        parser.add_argument("--root", default="~/datasets", type=str, help="root directory of datasets")
        parser.add_argument("--epochs", default=10, type=int, help="total number of training epochs")
        parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
        parser.add_argument("--beta1", default=0.9, type=float, help="beta_1 in Adam")
        parser.add_argument("--beta2", default=0.999, type=float, help="beta_2 in Adam")
        parser.add_argument("--lr-warmup", default=0, type=int, help="number of warming-up epochs")
        parser.add_argument("--batch-size", default=1000, type=int)
        parser.add_argument("--timesteps", default=100, type=int, help="number of diffusion steps")
        parser.add_argument("--beta-schedule", choices=["quad", "linear", "warmup10", "warmup50", "jsd"], default="linear")
        parser.add_argument("--beta-start", default=0.001, type=float)
        parser.add_argument("--beta-end", default=0.2, type=float)
        parser.add_argument("--model-mean-type", choices=["mean", "x_0", "eps"], default="eps", type=str)
        parser.add_argument("--model-var-type", choices=["learned", "fixed-small", "fixed-large"], default="fixed-large", type=str)  # noqa
        parser.add_argument("--loss-type", choices=["kl", "mse"], default="mse", type=str)
        parser.add_argument("--image-dir", default="./images/train", type=str)
        parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
        parser.add_argument("--chkpt-intv", default=100, type=int, help="frequency of saving a checkpoint")
        parser.add_argument("--eval-intv", default=1, type=int)
        parser.add_argument("--seed", default=1234, type=int, help="random seed")
        parser.add_argument("--resume", action="store_true", help="to resume training from a checkpoint")
        parser.add_argument("--device", default="cuda:0", type=str)
        parser.add_argument("--mid-features", default=128, type=int)
        parser.add_argument("--num-temporal-layers", default=3, type=int)

        self.args = parser.parse_args()

    def set_data(self, l=0, retraining_iteration=0, synth = []):
        
        seed_all(self.args.seed)
        self.in_features = 2
        self.root = os.path.expanduser(self.args.root)
        self.num_batches = self.args.size // self.args.batch_size
        
        if retraining_iteration == 0:
            self.trainloader = DataStreamer(self.args.dataset, batch_size=self.args.batch_size, num_batches=self.num_batches)

        else:
            self.trainloader = DataStreamer(self.args.dataset, batch_size=self.args.batch_size, num_batches=self.num_batches, synth = synth)
            

    def set_parameters(self):
        # training parameters
        self.device = torch.device(self.args.device)
        
        self.betas = get_beta_schedule(
            self.args.beta_schedule, beta_start=self.args.beta_start, beta_end=self.args.beta_end, timesteps=self.args.timesteps
        )
        
        self.diffusion = GaussianDiffusion(
            betas=self.betas, model_mean_type=self.args.model_mean_type, model_var_type=self.args.model_var_type, loss_type=self.args.loss_type
        )

        # model parameters
        self.out_features = 2 * self.in_features if self.args.model_var_type == "learned" else self.in_features
        self.model = Decoder(self.in_features, self.args.mid_features, self.args.num_temporal_layers)
        self.model.to(self.device)

        # training parameters
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))

    def set_checkpoint_path(self):
        # checkpoint path
        if not os.path.exists(self.args.chkpt_dir):
            os.makedirs(self.args.chkpt_dir)
        self.chkpt_path = os.path.join(self.args.chkpt_dir, f"ddpm_{self.args.dataset}.pt")

    def set_image_directory(self):
        # set up image directory
        self.image_dir = os.path.join(self.args.image_dir, f"{self.args.dataset}")
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def set_scheduler(self):
        # scheduler
        self.scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda t: min((t + 1) /self.args.lr_warmup, 1.0)) if self.args.lr_warmup > 0 else None

    def train(self, model):
        # load trainer
        grad_norm = 0  # gradient global clipping is disabled
        eval_intv = self.args.eval_intv
        
        trainer = Trainer(
            model=self.model if model is None else model,
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
        evaluator = Evaluator(
            true_data=np.concatenate([
                next(true_data) for _ in range(max_eval_count // eval_batch_size)
            ]), eval_batch_size=eval_batch_size, max_eval_count=max_eval_count, value_range=value_range)

        if self.args.resume:
            try:
                trainer.load_checkpoint(self.chkpt_path)
            except FileNotFoundError:
                print("Checkpoint file does not exist!")
                print("Starting from scratch...")

        trainer.train(evaluator, chkpt_path=self.chkpt_path, image_dir=self.image_dir, xlim=xlim, ylim=ylim)
        
        shape = (self.args.size,) + trainer.shape
        sample = trainer.diffusion.p_sample(denoise_fn=trainer.model, shape=shape, device=trainer.device, noise=None)
        
        return sample.cpu().numpy(), trainer.model


if __name__ == "__main__":
    train_toy = TrainToy()
    train_toy.set_data()
    train_toy.set_parameters()
    train_toy.set_checkpoint_path()
    train_toy.set_image_directory()
    train_toy.set_scheduler()
    train_toy.train()
