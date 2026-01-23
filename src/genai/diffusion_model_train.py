
from utils import standardize, unstandardize
import torch
import torch.nn.functional as F
from diffusion_model import DiffusionSchedule, TinyCondUNet1D
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

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
