"""Image <-> frame helpers for demo transmission.

Default (tile=0): the WHOLE image is one frame — it is only zero/reflect-padded up to a multiple of
`align` (= patch * 2^(stages-1) * window, 16 for the default config) so that the Swin windows fit,
and the encoder turns the entire picture into a single patch-token sequence.
tile>0 falls back to independent square tiles (legacy, only needed for models trained at 32x32).
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def load_image(path, max_side=None):
    im = Image.open(path).convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    return torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255   # (3,H,W)


def _pad(img, Hp, Wp):
    _, H, W = img.shape
    mode = "reflect" if (Hp - H < H and Wp - W < W) else "constant"
    return F.pad(img[None], (0, Wp - W, 0, Hp - H), mode=mode)[0]


def to_tiles(img, tile=0, align=16):
    """(3,H,W) -> (T,3,th,tw) and meta dict. tile=0: whole image as one frame."""
    _, H, W = img.shape
    if tile == 0:
        Hp, Wp = -(-H // align) * align, -(-W // align) * align
        return _pad(img, Hp, Wp)[None], {"H": H, "W": W, "Hp": Hp, "Wp": Wp, "th": Hp, "tw": Wp}
    Hp, Wp = -(-H // tile) * tile, -(-W // tile) * tile
    x = _pad(img, Hp, Wp)
    t = x.view(3, Hp // tile, tile, Wp // tile, tile).permute(1, 3, 0, 2, 4)
    return t.reshape(-1, 3, tile, tile), {"H": H, "W": W, "Hp": Hp, "Wp": Wp, "th": tile, "tw": tile}


def from_tiles(tiles, meta):
    H, W, Hp, Wp, th, tw = (meta[k] for k in ("H", "W", "Hp", "Wp", "th", "tw"))
    t = tiles.reshape(Hp // th, Wp // tw, 3, th, tw).permute(2, 0, 3, 1, 4)
    return t.reshape(3, Hp, Wp)[:, :H, :W]


def save_image(img, path):
    Image.fromarray((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(path)
