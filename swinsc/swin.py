"""Compact Swin Transformer blocks (Liu et al., ICCV 2021) for semantic encoder / decoder.

Resolution-agnostic: relative position bias is indexed by window size only, so a model trained
at one image size runs at any size divisible by patch * window * 2^(stages-1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, C)


def window_reverse(win, ws, H, W):
    B = win.shape[0] // ((H // ws) * (W // ws))
    x = win.view(B, H // ws, W // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, ws, heads):
        super().__init__()
        self.ws, self.heads, self.scale = ws, heads, (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (ws - 1)
        self.register_buffer("rel_idx", rel[..., 0] * (2 * ws - 1) + rel[..., 1], persistent=False)
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def forward(self, x, mask=None):
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_bias[self.rel_idx.reshape(-1)].reshape(N, N, -1).permute(2, 0, 1)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(Bn // nw, nw, self.heads, N, N) + mask[None, :, None]
            attn = attn.view(Bn, self.heads, N, N)
        attn = attn.softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(Bn, N, C))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, ws, shift, mlp_ratio=4.0):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def _shift_mask(self, H, W, device):
        if self.shift == 0:
            return None
        img = torch.zeros(1, H, W, 1, device=device)
        cnt = 0
        for hs in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
            for wsl in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
                img[:, hs, wsl] = cnt; cnt += 1
        mw = window_partition(img, self.ws).squeeze(-1)
        m = mw[:, None] - mw[:, :, None]
        return m.masked_fill(m != 0, -100.0)

    def forward(self, x, H, W):
        B, L, C = x.shape
        h = self.n1(x).view(B, H, W, C)
        if self.shift:
            h = torch.roll(h, (-self.shift, -self.shift), (1, 2))
        win = self.attn(window_partition(h, self.ws), self._shift_mask(H, W, x.device))
        h = window_reverse(win, self.ws, H, W)
        if self.shift:
            h = torch.roll(h, (self.shift, self.shift), (1, 2))
        x = x + h.reshape(B, L, C)
        return x + self.mlp(self.n2(x))


class SwinStage(nn.Module):
    def __init__(self, dim, depth, heads, ws):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, heads, ws, 0 if i % 2 == 0 else ws // 2) for i in range(depth)])

    def forward(self, x, H, W):
        for b in self.blocks:
            x = b(x, H, W)
        return x


class PatchMerging(nn.Module):
    """2x2 neighbourhood concat -> LayerNorm -> Linear(4C -> out)."""

    def __init__(self, dim, out):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.red = nn.Linear(4 * dim, out, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.red(self.norm(x.view(B, -1, 4 * C))), H // 2, W // 2


class PatchExpanding(nn.Module):
    """Linear(C -> 4*out) -> pixel shuffle x2 -> LayerNorm."""

    def __init__(self, dim, out):
        super().__init__()
        self.exp = nn.Linear(dim, 4 * out, bias=False)
        self.norm = nn.LayerNorm(out)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.exp(x).view(B, H, W, 2, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, 4 * L, -1)
        return self.norm(x), H * 2, W * 2


class SwinEncoder(nn.Module):
    """image -> patch embed -> [stage -> merge]* -> semantic tokens (B, N, L_s)."""

    def __init__(self, in_ch=3, patch=2, dims=(96, 192), depths=(2, 2), heads=(3, 6), ws=4, out_dim=None):
        super().__init__()
        self.patch, self.ws, self.n_stages = patch, ws, len(dims)
        self.embed = nn.Conv2d(in_ch, dims[0], patch, patch)
        self.norm0 = nn.LayerNorm(dims[0])
        self.stages, self.merges = nn.ModuleList(), nn.ModuleList()
        for i in range(len(dims)):
            self.stages.append(SwinStage(dims[i], depths[i], heads[i], ws))
            if i < len(dims) - 1:
                self.merges.append(PatchMerging(dims[i], dims[i + 1]))
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], out_dim) if out_dim else nn.Identity()
        self.down = patch * 2 ** (len(dims) - 1)

    def forward(self, img):
        x = self.embed(img)
        B, C, H, W = x.shape
        x = self.norm0(x.flatten(2).transpose(1, 2))
        for i, st in enumerate(self.stages):
            x = st(x, H, W)
            if i < len(self.merges):
                x, H, W = self.merges[i](x, H, W)
        return self.head(self.norm(x)), (H, W)


class SwinDecoder(nn.Module):
    """semantic tokens (B, N, L_s) -> [stage -> expand]* -> image (B, 3, H, W) in [0,1]."""

    def __init__(self, out_ch=3, patch=2, dims=(192, 96), depths=(2, 2), heads=(6, 3), ws=4, in_dim=None):
        super().__init__()
        self.patch = patch
        self.head = nn.Linear(in_dim, dims[0]) if in_dim else nn.Identity()
        self.norm0 = nn.LayerNorm(dims[0])
        self.stages, self.expands = nn.ModuleList(), nn.ModuleList()
        for i in range(len(dims)):
            self.stages.append(SwinStage(dims[i], depths[i], heads[i], ws))
            if i < len(dims) - 1:
                self.expands.append(PatchExpanding(dims[i], dims[i + 1]))
        self.norm = nn.LayerNorm(dims[-1])
        self.out = nn.Linear(dims[-1], out_ch * patch * patch)

    def forward(self, tok, hw):
        H, W = hw
        x = self.norm0(self.head(tok))
        for i, st in enumerate(self.stages):
            x = st(x, H, W)
            if i < len(self.expands):
                x, H, W = self.expands[i](x, H, W)
        x = self.out(self.norm(x))                                       # (B, H*W, 3*p*p)
        B = x.shape[0]
        x = x.view(B, H, W, -1, self.patch, self.patch).permute(0, 3, 1, 4, 2, 5)
        return torch.sigmoid(x.reshape(B, -1, H * self.patch, W * self.patch))
