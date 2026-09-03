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
        self.deepma = cfg.mask_type == "deepma"
        self.lk = cfg.mask_type == "learned_k"
        in_dim = cfg.l_e // cfg.users if self.oma else cfg.l_e
        if self.oma:
            self.blk = in_dim                                  # OMA: slice own block, no mask
        elif self.deepma:
            pass                                               # DeepMA: own decoder reads the whole mixture
        else:
            self.masks = UserMasks(cfg.users, cfg.l_e, cfg.mask_type, cfg.mask_seed)
        # A plain linear map, deliberately WITHOUT a LayerNorm in front of it. A
        # LayerNorm over in_dim features discards the mean and the scale of the
        # received vector; for the OMA block that is fatal (in_dim = L/N = 2 keeps
        # only sign(a - b), in_dim = 1 outputs 0 identically), and the shared chain
        # needs no normalization either, since ZF already removes the channel gain.
        self.reduce = nn.ModuleList([nn.Linear(in_dim, cfg.l_s) for _ in range(cfg.users)])
        if self.lk:
            # load-indexed family: one reduction per user AND active count, the
            # receiver-side counterpart of the per-K projection in the transmitter
            self.reduce_k = nn.ModuleList([nn.ModuleList([nn.Linear(in_dim, cfg.l_s) for _ in range(cfg.users)])
                                           for _ in range(cfg.users)])   # [K-1][u]
        nl = cfg.users if getattr(cfg, "load_cond", False) else None
        self.decoders = nn.ModuleList([
            SwinDecoder(cfg.in_ch, cfg.patch, tuple(reversed(cfg.dims)), tuple(reversed(cfg.depths)),
                        tuple(reversed(cfg.heads)), cfg.window, cfg.l_s, n_loads=nl) for _ in range(cfg.users)])

    def forward(self, z, u, hw, K=None):
        """K is the active count the receiver learns from the grant; only the
        load-indexed mask family uses it, every other chain ignores it."""
        if self.oma:
            z_u = z[:, :, u * self.blk:(u + 1) * self.blk]
        elif self.deepma:
            z_u = z
        else:
            z_u = self.masks.demux(z, u, K)
        dec = self.decoders[u]
        kw = {"K": K} if dec.cond is not None else {}
        if self.lk:
            K = self.cfg.users if K is None else K
            return dec(self.reduce_k[K - 1][u](z_u), hw, **({"K": K} if dec.cond is not None else {}))
        return dec(self.reduce[u](z_u), hw, **kw)

    def user_state_dict(self, u):
        """Only the tensors receiver `u` needs (mask + its own decoder)."""
        sd = self.state_dict()
        keep = {k: v for k, v in sd.items()
                if k.startswith("masks.") or k.startswith(f"reduce.{u}.") or k.startswith(f"decoders.{u}.")
                or (k.startswith("reduce_k.") and k.split(".")[2] == str(u))}
        return keep
