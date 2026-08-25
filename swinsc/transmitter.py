"""Multi-user semantic TRANSMITTER (Swin encoder + mask multiplexing). No receiver imports.

    X_i --SwinEncoder--> S_i (B,N,L_s) --expand--> (B,N,L_e) --mask M_i--> E_i --mux--> F
"""
import torch
import torch.nn as nn

from .config import Config
from .swin import SwinEncoder
from .mask_mux import UserMasks, Multiplexer, power_normalize


class Transmitter(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = SwinEncoder(cfg.in_ch, cfg.patch, cfg.dims, cfg.depths, cfg.heads, cfg.window, cfg.l_s)
        self.oma = cfg.mask_type == "oma"
        if self.oma:
            # OMA baseline: no masks, no superposition. Each user owns a disjoint L_e/U block of the frame.
            assert cfg.l_e % cfg.users == 0, "L_e must be divisible by the number of users for OMA"
            self.blk = cfg.l_e // cfg.users
            self.expand = nn.Linear(cfg.l_s, self.blk)
        else:
            self.expand = nn.Linear(cfg.l_s, cfg.l_e)      # L_e = beta * L_s
            self.masks = UserMasks(cfg.users, cfg.l_e, cfg.mask_type, cfg.mask_seed)
            self.mux = Multiplexer(cfg.users)

    def encode_user(self, img, u):
        """Single-user semantic encoding (+ masking for shared schemes). Returns E_u, token grid hw."""
        s, hw = self.encoder(img)
        e = self.expand(s)
        return (e if self.oma else self.masks.apply(e, u)), hw

    def forward(self, imgs, active=None):
        """imgs: list of (B,3,H,W) tensors, one per active user (same H, W).
        Returns F (B,N,L_e) unit power, token grid (h,w)."""
        active = list(range(len(imgs))) if active is None else active
        es, hw = [], None
        for img, u in zip(imgs, active):
            e, hw = self.encode_user(img, u)
            es.append(e)
        if self.oma:                                          # resource partitioning: concat user blocks
            B, N, _ = es[0].shape
            f = es[0].new_zeros(B, N, self.cfg.l_e)
            for e, u in zip(es, active):
                f[:, :, u * self.blk:(u + 1) * self.blk] = e
            return power_normalize(f), hw
        return self.mux(es, active), hw
