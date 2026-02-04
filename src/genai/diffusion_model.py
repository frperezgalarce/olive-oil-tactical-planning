import torch
import math
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        
        # self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)

        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))

        # FiLM conditioning
        emb_out = self.emb_proj(F.silu(emb))          # (B, 2*out_ch)
        gamma, beta = emb_out.chunk(2, dim=1)         # each (B, out_ch)

        gamma = gamma.unsqueeze(-1)                   # (B, out_ch, 1)
        beta  = beta.unsqueeze(-1)

        h = h * (1 + gamma) + beta                    # FiLM

        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class TinyCondUNet1D(nn.Module):
    def __init__(self, in_vars=4, ctx_vars=4, base=128, emb_dim=256):
        """
        Model predicts noise eps for target sequence.
        Inputs:
          x_t: (B, L, in_vars) noisy future
          ctx: (B, Lctx, ctx_vars) conditioning past
          t:   (B,)
        Strategy:
          - Encode ctx with Conv1D -> get a pooled context vector
          - Concatenate ctx vector into timestep embedding
          - U-Net over x_t along time dimension
        """
        super().__init__()
        self.emb_dim = emb_dim

        # ctx encoder (simple)
        self.ctx_conv1 = nn.Conv1d(ctx_vars, base, 5, padding=2)
        self.ctx_conv2 = nn.Conv1d(base, base, 5, padding=2)
        
        self.ctx_attn = nn.Sequential(
            nn.Conv1d(base, base, 1),
            nn.SiLU(),
            nn.Conv1d(base, 1, 1)  # attention logits over time
        )


        self.ctx_proj = nn.Linear(base, emb_dim)

        self.pos_proj = nn.Conv1d(emb_dim, base, 1)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.in_conv = nn.Conv1d(in_vars, base, 3, padding=1)

        # Down
        self.rb1 = ResBlock1D(base, base, emb_dim)
        self.down = nn.Conv1d(base, base*2, 4, stride=2, padding=1)  # L/2

        self.rb2 = ResBlock1D(base*2, base*2, emb_dim)
        self.mid = ResBlock1D(base*2, base*2, emb_dim)

        # Up
        self.up = nn.ConvTranspose1d(base*2, base, 4, stride=2, padding=1)  # back to L
        self.rb3 = ResBlock1D(base*2, base, emb_dim)  # skip concat

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv1d(base, in_vars, 3, padding=1)

    def forward(self, x_t, ctx, t):
        # x_t: (B, L, C)
        # ctx: (B, Lctx, Cctx)
        B, L, C = x_t.shape

        # ctx encoding
        c = ctx.transpose(1,2)  # (B, Cctx, Lctx)
        c = F.silu(self.ctx_conv1(c))
        c = F.silu(self.ctx_conv2(c))
        
        
        w = self.ctx_attn(c)                # (B,1,Lctx)
        w = torch.softmax(w, dim=-1)
        c = (c * w).sum(dim=-1)             # (B, base)

        c = self.ctx_proj(c)              # (B, emb_dim)

        # time embedding + context
        te = timestep_embedding(t, self.emb_dim)
        #te = timestep_embedding_with_seasonality(t, self.emb_dim)
        emb = self.time_mlp(te + c)

        x = x_t.transpose(1,2)  # (B, C, L)
        x0 = self.in_conv(x)

        # Lead-time / position embedding (0..L-1)
        pos = torch.arange(L, device=x_t.device).long().unsqueeze(0).repeat(B, 1)   # (B, L)
        pos_emb = timestep_embedding(pos.reshape(-1), self.emb_dim).view(B, L, self.emb_dim)  # (B,L,emb_dim)
        pos_emb = pos_emb.transpose(1, 2)  # (B, emb_dim, L)

        x0 = x0 + self.pos_proj(pos_emb)   # (B, base, L)

        h1 = self.rb1(x0, emb)
        h2 = self.down(h1)
        h2 = self.rb2(h2, emb)
        hmid = self.mid(h2, emb)

        u = self.up(hmid)
        # handle any odd length mismatch (shouldn't happen for L=60 with stride 2, but safe)
        if u.shape[-1] != h1.shape[-1]:
            u = F.pad(u, (0, h1.shape[-1] - u.shape[-1]))

        u = torch.cat([u, h1], dim=1)
        u = self.rb3(u, emb)

        out = self.out_conv(F.silu(self.out_norm(u)))

        return out.transpose(1,2)  # (B, L, C)
    
class DiffusionSchedule:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bar = alpha_bar.to(device)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

        # After computing: betas, alphas, alpha_bar
        alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]], dim=0)

        self.alpha_bar_prev = alpha_bar_prev

        # Posterior variance: beta_tilde
        self.posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

        # Posterior mean coefficients:
        self.posterior_mean_coef1 = betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        self.posterior_mean_coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(alphas) / (1.0 - alpha_bar)


    def q_sample(self, x0, t, noise=None):
        """
        x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1-alpha_bar[t]) * eps
        x0 shape: (B, L, C)
        t shape:  (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return a * x0 + b * noise

def timestep_embedding_with_seasonality(t, dim, period=365.0):
    half = dim // 2 - 1  # reserve space
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1)
    )

    ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    # yearly cycle
    omega_year = 2 * math.pi / period
    yearly = torch.stack([
        torch.sin(omega_year * t),
        torch.cos(omega_year * t)
    ], dim=1)

    emb = torch.cat([emb, yearly], dim=1)

    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))

    return emb


def timestep_embedding(t, dim):
    """
    Sinusoidal embedding for diffusion timestep.
    t: (B,)
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1)
    )
    ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb