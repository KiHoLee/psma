from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    img_size: int = 32          # training tile size (inference runs on any multiple of `down`)
    in_ch: int = 3
    patch: int = 2
    dims: tuple = (96, 192)     # Swin stage widths (encoder); decoder mirrors them
    depths: tuple = (2, 2)
    heads: tuple = (3, 6)
    window: int = 4
    l_s: int = 64               # semantic embedding dim per token (L_s)
    beta: int = 2               # expansion factor, L_e = beta * L_s (Nget et al.)
    users: int = 2              # number of multiplexed users
    mask_type: str = "learned"  # learned | hadamard | haar
    mask_seed: int = 2026
    channel: str = "awgn"
    rician_k: float = 4.0

    @property
    def l_e(self):
        return self.beta * self.l_s

    @property
    def down(self):
        return self.patch * 2 ** (len(self.dims) - 1)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        for k in ("dims", "depths", "heads"):
            d[k] = tuple(d[k])
        return cls(**d)
