from utils import standardize, unstandardize
import torch

@torch.no_grad()
def p_sample_loop(ctx, model, device, sched, HORIZON, VARS, mean_t, std_t, n_scenarios=10):
    """
    ctx: (1, 60, 4) raw scale tensor
    returns: (n_scenarios, 60, 4) raw scale scenarios
    """
    model.eval()

    ctx = ctx.to(device)
    ctx_s = standardize(ctx, mean_t, std_t)

    # Start from pure noise
    x = torch.randn((n_scenarios, HORIZON, VARS), device=device)

    for ti in reversed(range(sched.T)):
        t = torch.full((n_scenarios,), ti, device=device, dtype=torch.long)
        eps = model(x, ctx_s.repeat(n_scenarios,1,1), t)

        beta = sched.betas[ti]
        alpha = sched.alphas[ti]
        abar = sched.alpha_bar[ti]

        # DDPM mean (using eps-pred parameterization)
        # x0_hat = (x - sqrt(1-abar)*eps) / sqrt(abar)
        x0_hat = (x - sched.sqrt_one_minus_alpha_bar[ti]*eps) / (sched.sqrt_alpha_bar[ti] + 1e-8)

        # mean = 1/sqrt(alpha) * (x - beta/sqrt(1-abar) * eps)
        mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / (sched.sqrt_one_minus_alpha_bar[ti] + 1e-8)) * eps)

        if ti > 0:
            z = torch.randn_like(x)
            # variance (beta_t) is common in toy setups; other choices exist
            x = mean + torch.sqrt(beta) * z
        else:
            x = mean

    samples = unstandardize(x, mean_t, std_t)
    # Keep precipitation non-negative
    samples[..., 3] = torch.clamp(samples[..., 3], min=0.0)
    return samples.detach().cpu()
