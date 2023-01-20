import functools

import torch
import torch.distributions

import utils

LOGGER = utils.log.getLogger(__name__)

__defined_kl = False

EPS = 1e-5


def clamp_probs(probs):
    probs = probs.clamp(EPS, 1. - EPS)  # Will no longer sum to 1
    return probs / probs.sum(-1, keepdim=True)  # to simplex


def grid(h, w, pad=0, device='cpu', dtype=torch.float32, norm=False):
    hr = torch.arange(h + 2 * pad, device=device) - pad
    wr = torch.arange(w + 2 * pad, device=device) - pad
    if norm:
        hr = hr / (h + 2 * pad - 1)
        wr = wr / (w + 2 * pad - 1)
    ig, jg = torch.meshgrid(hr, wr)
    g = torch.stack([jg, ig]).to(dtype)[None]
    return g


@functools.lru_cache(2)
def cached_grid(h, w, pad=0, device='cpu', dtype=torch.float32, norm=False):
    return grid(h, w, pad, device, dtype, norm)

