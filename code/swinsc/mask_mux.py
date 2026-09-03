"""Masking-based shared-embedding multiplexing (AI-native multiple access).

Follows the group's shared-embedding line of work (JSAC AI-native / SSE / EDMA / Nget et al.):

    E_i = S_i (.) M_i                 user-specific mask on the expanded embedding (L_e = beta * L_s)
    F   = sum_i w_i * E_i             superposition on one shared resource, unit power
    Z   = H (.) F + n                 channel
    Z_i = Z (.) M_i                   matched-filter de-multiplexing for user i

Mask families (cfg.mask_type):
    learned   - trainable per-user mask vectors, trained end-to-end (Nget et al. Eq. 15)
    hadamard  - fixed Walsh-Hadamard rows (+-1), mutually orthogonal (EDMA mask-family experiment: Walsh-Hadamard vs Haar)
    haar      - fixed random orthogonal L_e x L_e rotation per user, E_i = S_i M_i^T, Z_i = Z M_i (EDMA)
OMA (cfg.mask_type == "oma") is NOT a mask: the transmitter simply concatenates each user's own
L_e/U-dimensional block into the frame and the receiver slices its block (see transmitter.py /
receiver.py). Same frame symbols and total power as the shared schemes -> fair baseline.
The TX side only needs mask(i) for its own users; the RX side only its own mask.
"""
import math
import torch
import torch.nn as nn


def hadamard(n):
    assert n & (n - 1) == 0, "Walsh-Hadamard size must be a power of two"
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def haar(n, gen):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen))
    return q * torch.sign(torch.diagonal(r))


class UserMasks(nn.Module):
    """Holds the masks of all `users`; both TX and RX instantiate it with the same seed/config."""

    def __init__(self, users, dim, mask_type="learned", seed=2026):
        super().__init__()
        self.users, self.dim, self.mask_type = users, dim, mask_type
        if mask_type == "learned":
            self.W = nn.Parameter(torch.randn(users, dim) / math.sqrt(dim))
        elif mask_type == "learned_k":
            # Load-indexed masks (2026-09-03): one mask TABLE per active count K,
            # W[K-1, u, :]. A single learned mask per user partitions the band
            # (the trained N=4 masks concentrate on 2-3 of 8 dimensions), so a
            # user left alone kept only its own subset and recovered power but
            # not bandwidth. With a table the K=1 mask can span the whole
            # frame and the K=N masks can partition it. The receiver already
            # knows the active set, hence K, so no extra signaling is needed.
            self.W = nn.Parameter(torch.randn(users, users, dim) / math.sqrt(dim))
        elif mask_type == "hadamard":
            H = hadamard(dim)
            self.register_buffer("W", H[1:users + 1])            # skip the all-ones row
        elif mask_type == "haar":
            g = torch.Generator().manual_seed(seed)
            self.register_buffer("W", torch.stack([haar(dim, g) for _ in range(users)]))   # (U, dim, dim)
        else:
            raise ValueError(mask_type)

    def mask(self, u, K=None):
        """Mask of user u; K (active count) selects the table for learned_k and is ignored otherwise."""
        if self.mask_type == "haar":
            return self.W[u]
        if self.mask_type == "learned":
            w = self.W[u]
            return w / w.norm().clamp_min(1e-8) * math.sqrt(self.dim)   # unit average gain, as in SSE
        if self.mask_type == "learned_k":
            K = self.users if K is None else K
            w = self.W[K - 1, u]
            return w / w.norm().clamp_min(1e-8) * math.sqrt(self.dim)
        return self.W[u]

    def apply(self, s, u, K=None):
        """E_u = mask(S_u).  s: (B, N, dim)."""
        m = self.mask(u, K)
        return s @ m.T if self.mask_type == "haar" else s * m

    def demux(self, z, u, K=None):
        """Z_u = Demux(Z, M_u)."""
        m = self.mask(u, K)
        return z @ m if self.mask_type == "haar" else z * m


class Multiplexer(nn.Module):
    """F = sum_i w_i E_i, then unit average power (Nget et al. Eq. 3-4)."""

    def __init__(self, users, learn_weights=True):
        super().__init__()
        self.w = nn.Parameter(torch.ones(users)) if learn_weights else None
        self.users = users

    def forward(self, masked_list, active=None):
        """masked_list: list of (B,N,dim) tensors in user order; active: list of user ids present."""
        active = list(range(len(masked_list))) if active is None else active
        f = 0
        for e, u in zip(masked_list, active):
            f = f + (self.w[u] if self.w is not None else 1.0) * e
        return power_normalize(f)


def power_normalize(x):
    B = x.shape[0]
    p = x.reshape(B, -1).pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
    return x / p.view(B, *([1] * (x.dim() - 1)))
