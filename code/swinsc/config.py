from dataclasses import dataclass, asdict
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_main import MAIN, l_s as _l_s   # the ONE configuration (standard 7.9)


@dataclass
class Config:
    img_size: int = MAIN["CROP"]  # training tile size (inference runs on any multiple of `down`)
    in_ch: int = 3
    patch: int = 2
    dims: tuple = MAIN["DIMS"]  # Swin stage widths (encoder); decoder mirrors them
    depths: tuple = (2, 2)
    heads: tuple = (3, 6)
    window: int = 4
    l_s: int = _l_s()           # semantic embedding dim per token (L_s = L / beta)
    beta: int = MAIN["BETA"]    # expansion factor, L_e = beta * L_s (Nget et al.)
    users: int = MAIN["N"][0]   # number of multiplexed users
    mask_type: str = "learned"  # learned | learned_k | hadamard | haar | oma | deepma
    load_cond: bool = False     # FiLM load conditioning of the Swin body on the active count (learned_k family)
    mask_seed: int = MAIN["SIGNATURE_SEED"]
    channel: str = MAIN["CHANNEL"]
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
