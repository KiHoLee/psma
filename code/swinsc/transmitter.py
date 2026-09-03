"""Multi-user semantic TRANSMITTER (Swin encoder + mask multiplexing). No receiver imports.

    X_i --SwinEncoder--> S_i (B,N,L_s) --expand--> (B,N,L_e) --mask M_i--> E_i --mux--> F
"""
import torch
import torch.nn as nn

from .config import Config
from .swin import SwinEncoder
from .mask_mux import UserMasks, Multiplexer, ProgressiveSpreader, power_normalize


class Transmitter(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.oma = cfg.mask_type == "oma"
        self.deepma = cfg.mask_type == "deepma"
        self.prog = cfg.mask_type == "prog"
        if self.deepma:
            # DeepMA (Zhang et al., TCCN 2024) adapted to this frame: one INDEPENDENT
            # encoder-decoder pair per user, no masks and no codes. Every user's
            # L_e-dimensional semantic symbol vector is superposed at unit frame
            # power; separation is left to training on the mixture (their Alg. 1
            # has no explicit orthogonality term). Same Swin capacity per pair.
            self.encoders = nn.ModuleList([
                SwinEncoder(cfg.in_ch, cfg.patch, cfg.dims, cfg.depths, cfg.heads, cfg.window, cfg.l_s)
                for _ in range(cfg.users)])
            self.expands = nn.ModuleList([nn.Linear(cfg.l_s, cfg.l_e) for _ in range(cfg.users)])
            self.mux = Multiplexer(cfg.users, learn_weights=False)
        else:
            # the progressive-spread encoder stays load-agnostic by design (one
            # importance-ordered code for every load); conditioning applies to
            # its decoders only (prefix length), see receiver.py
            self.encoder = SwinEncoder(cfg.in_ch, cfg.patch, cfg.dims, cfg.depths, cfg.heads, cfg.window, cfg.l_s,
                                       n_loads=cfg.users if (getattr(cfg, "load_cond", False) and not self.prog) else None)
        if self.oma:
            # OMA baseline: no masks, no superposition. Each user owns a disjoint L_e/U block of the frame.
            assert cfg.l_e % cfg.users == 0, "L_e must be divisible by the number of users for OMA"
            self.blk = cfg.l_e // cfg.users
            self.expand = nn.Linear(cfg.l_s, self.blk)
        elif cfg.mask_type == "learned_k":
            # Load-indexed multiple-access layer: the projection onto the frame
            # is ALSO indexed by the active count. The Swin body is shared and
            # load-agnostic, so a K-independent projection would emit the same
            # 8 latent dimensions at every load and a mask could only select
            # among them; a per-K projection lets a lone user spread its latent
            # over the whole frame and a full-load user compress it into the
            # dimensions its mask keeps. N * (l_s * l_e + l_e) extra scalars.
            self.expands = nn.ModuleList([nn.Linear(cfg.l_s, cfg.l_e) for _ in range(cfg.users)])
            self.masks = UserMasks(cfg.users, cfg.l_e, cfg.mask_type, cfg.mask_seed)
            self.mux = Multiplexer(cfg.users)
        elif self.prog:
            # progressive-spread prototype: L ordered symbols per token, prefix
            # b_j(K) spread over disjoint orthonormal Walsh-Hadamard codes
            self.expand = nn.Linear(cfg.l_s, cfg.l_e)
            self.spreader = ProgressiveSpreader(cfg.l_e)
        elif not self.deepma:
            self.expand = nn.Linear(cfg.l_s, cfg.l_e)      # L_e = beta * L_s
            self.masks = UserMasks(cfg.users, cfg.l_e, cfg.mask_type, cfg.mask_seed)
            self.mux = Multiplexer(cfg.users)

    def encode_user(self, img, u, K=None):
        """Single-user semantic encoding (+ masking for shared schemes). Returns E_u, token grid hw.
        K is the active count; only the load-indexed family uses it."""
        if self.deepma:
            s, hw = self.encoders[u](img)
            return self.expands[u](s), hw
        s, hw = self.encoder(img, K) if self.encoder.cond is not None else self.encoder(img)
        if self.cfg.mask_type == "learned_k":
            K = self.cfg.users if K is None else K
            return self.masks.apply(self.expands[K - 1](s), u, K), hw
        e = self.expand(s)
        return (e if (self.oma or self.prog) else self.masks.apply(e, u, K)), hw

    def forward(self, imgs, active=None):
        """imgs: list of (B,3,H,W) tensors, one per active user (same H, W).
        Returns F (B,N,L_e) unit power, token grid (h,w)."""
        active = list(range(len(imgs))) if active is None else active
        es, hw = [], None
        for img, u in zip(imgs, active):
            e, hw = self.encode_user(img, u, K=len(active))
            es.append(e)
        if self.prog:                                         # prefix b_j(K) of user j on its own codes
            K = len(active)
            f = sum(self.spreader.spread(e, j, K) for j, e in enumerate(es))
            return power_normalize(f), hw
        if self.oma:                                          # resource partitioning: concat user blocks
            B, N, _ = es[0].shape
            f = es[0].new_zeros(B, N, self.cfg.l_e)
            for e, u in zip(es, active):
                f[:, :, u * self.blk:(u + 1) * self.blk] = e
            return power_normalize(f), hw
        return self.mux(es, active), hw
