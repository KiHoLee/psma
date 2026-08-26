"""Per-user semantic RECEIVER (demux by own mask + Swin decoder). No transmitter imports.

    Z --demux M_u--> Z_u (B,N,L_e) --reduce--> (B,N,L_s) --SwinDecoder--> X_u_hat
"""
import torch
import torch.nn as nn

from .config import Config
from .swin import SwinDecoder
from .mask_mux import UserMasks


class Receiver(nn.Module):
    """Holds decoders for all users so one checkpoint serves every receiver; a deployed
    receiver process only ever calls forward(z, u=its own id)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.oma = cfg.mask_type == "oma"
        in_dim = cfg.l_e // cfg.users if self.oma else cfg.l_e
        if self.oma:
            self.blk = in_dim                                  # OMA: slice own block, no mask
        else:
            self.masks = UserMasks(cfg.users, cfg.l_e, cfg.mask_type, cfg.mask_seed)
        self.reduce = nn.ModuleList([nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, cfg.l_s))
                                     for _ in range(cfg.users)])
        self.decoders = nn.ModuleList([
            SwinDecoder(cfg.in_ch, cfg.patch, tuple(reversed(cfg.dims)), tuple(reversed(cfg.depths)),
                        tuple(reversed(cfg.heads)), cfg.window, cfg.l_s) for _ in range(cfg.users)])

    def forward(self, z, u, hw):
        z_u = z[:, :, u * self.blk:(u + 1) * self.blk] if self.oma else self.masks.demux(z, u)
        return self.decoders[u](self.reduce[u](z_u), hw)

    def user_state_dict(self, u):
        """Only the tensors receiver `u` needs (mask + its own decoder)."""
        sd = self.state_dict()
        keep = {k: v for k, v in sd.items()
                if k.startswith("masks.") or k.startswith(f"reduce.{u}.") or k.startswith(f"decoders.{u}.")}
        return keep
