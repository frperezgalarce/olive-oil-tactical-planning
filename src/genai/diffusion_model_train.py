
from utils import standardize, unstandardize
import torch
import torch.nn.functional as F
from diffusion_model import DiffusionSchedule, TinyCondUNet1D
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import math

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="min"):
        """
        mode: "min" for loss, "max" for metrics like accuracy.
        """
        assert mode in ("min", "max")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode

        self.best = math.inf if mode == "min" else -math.inf
        self.num_bad = 0
        self.best_state = None
        self.best_epoch = None

    def step(self, value, model, epoch):
        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)

        if improved:
            self.best = float(value)
            self.num_bad = 0
            self.best_state = copy.deepcopy(model.state_dict())  # safe checkpoint
            self.best_epoch = epoch
            return False  # do not stop
        else:
            self.num_bad += 1
            return self.num_bad >= self.patience  # stop if patience exceeded

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(opt, model, sched, device, train_loader, mean_t, std_t):
    model.train()
    losses = []
    for ctx, tgt in train_loader:
        ctx = ctx.to(device)
        tgt = tgt.to(device)

        ctx_s = standardize(ctx, mean_t, std_t)
        tgt_s = standardize(tgt, mean_t, std_t)

        B = tgt_s.shape[0]
        t = torch.randint(0, sched.T, (B,), device=device)
        noise = torch.randn_like(tgt_s)
        x_t = sched.q_sample(tgt_s, t, noise=noise)

        pred = model(x_t, ctx_s, t)
        loss = F.mse_loss(pred, noise)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def eval_epoch(model, device, val_loader, sched, mean_t, std_t):
    model.eval()
    losses = []
    for ctx, tgt in val_loader:
        ctx = ctx.to(device)
        tgt = tgt.to(device)

        ctx_s = standardize(ctx, mean_t, std_t)
        tgt_s = standardize(tgt, mean_t, std_t)

        B = tgt_s.shape[0]
        t = torch.randint(0, sched.T, (B,), device=device)
        noise = torch.randn_like(tgt_s)
        x_t = sched.q_sample(tgt_s, t, noise=noise)

        pred = model(x_t, ctx_s, t)
        loss = F.mse_loss(pred, noise)
        losses.append(loss.item())
    return float(np.mean(losses))
