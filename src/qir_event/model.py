"""Small quality-aware, causal raster hazard model. No pretrained weights."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def integrated_hazard(eta, delta_t):
    if not torch.isfinite(delta_t).all() or (delta_t <= 0).any():
        raise ValueError("finite, positive exposure required")
    return torch.exp((eta + delta_t.log()).clamp(-30., 20.))


def event_probability(eta, delta_t):
    return -torch.expm1(-integrated_hazard(eta, delta_t))


def interval_nll(eta, delta_t, event, weights=None):
    if not torch.isfinite(event).all() or not ((event == 0) | (event == 1)).all():
        raise ValueError("events must be binary")
    exposure = integrated_hazard(eta, delta_t)
    loss = torch.where(event.bool(), -torch.log(-torch.expm1(-exposure)), exposure)
    if weights is None:
        return loss.mean()
    if (weights < 0).any() or not torch.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("finite nonnegative weights with positive total required")
    return (loss * weights).sum() / weights.sum()


class QIREventLite(nn.Module):
    def __init__(self, n_scales=3, hidden=24, mode="qir"):
        super().__init__()
        if mode not in {"baseline", "quality", "spatial", "qir"}:
            raise ValueError("unknown ablation")
        self.mode, self.n_scales = mode, n_scales
        self.baseline = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 1))
        self.quality_head = nn.Sequential(nn.Linear(n_scales+1, 12), nn.Tanh(), nn.Linear(12, 1))
        self.spatial = nn.Sequential(nn.Conv2d(2, 12, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(12, hidden, 3, padding=1), nn.ReLU())
        self.gate = nn.Linear(1, 1)
        self.temporal = nn.GRU(hidden+n_scales+1, hidden, batch_first=True)
        self.residual = nn.Linear(hidden, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, images, valid, times, lengths, baseline_x):
        # B,T,S,H,W; no batch normalization across timestamps or examples.
        b, t, s, h, w = images.shape
        if s != self.n_scales or valid.shape != images.shape or times.shape != (b, t):
            raise ValueError("inconsistent input shapes")
        if (lengths < 1).any() or (lengths > t).any():
            raise ValueError("invalid history lengths")
        eta0 = self.baseline(baseline_x).squeeze(-1)
        if self.mode == "baseline":
            return eta0
        active = torch.arange(t, device=images.device)[None] < lengths[:, None]
        mask = valid.bool() & active[:, :, None, None, None]
        clean = torch.where(mask, images, 0.)
        if not torch.isfinite(clean).all():
            raise ValueError("observed image values must be finite")
        quality = mask.float().mean((-1, -2))
        gaps = torch.cat([torch.zeros_like(times[:, :1]), times[:, 1:]-times[:, :-1]], dim=1)
        if not torch.isfinite(times[active]).all() or (gaps[:, 1:][active[:, 1:]] <= 0).any():
            raise ValueError("history times must be finite and strictly increasing")
        gaps = torch.where(active, gaps, 0.).clamp_min(0).log1p().unsqueeze(-1)
        last = lengths-1
        ii = torch.arange(b, device=images.device)
        if self.mode == "quality":
            return eta0 + self.quality_head(torch.cat([quality, gaps], -1)[ii, last]).squeeze(-1)
        inputs = torch.stack([clean, mask.float()], dim=3).reshape(b*t*s, 2, h, w)
        features = self.spatial(inputs)
        m = mask.reshape(b*t*s, 1, h, w).float()
        z = ((features*m).sum((-1, -2))/m.sum((-1, -2)).clamp_min(1)).reshape(b, t, s, -1)
        scores = self.gate(quality.unsqueeze(-1)).squeeze(-1)
        # Empty scales get zero weight, including fully unobserved frames.
        weights = torch.softmax(scores.masked_fill(quality == 0, -1e4), -1) * (quality > 0)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        z = (z*weights.unsqueeze(-1)).sum(2)
        if self.mode == "spatial":
            state = z[ii, last]
        else:
            encoded = torch.cat([z, quality, gaps], -1)
            packed = pack_padded_sequence(encoded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, state = self.temporal(packed)
            state = state[-1]
        return eta0 + self.residual(state).squeeze(-1)

