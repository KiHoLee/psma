"""Physical channel (Eq. 3): R_x = H * T_x + N_p, N_p ~ CN(0, sigma_n^2).

Real-valued tensors (B,N,ch_dim) are interpreted as ch_dim/2 complex symbols per token.
T_x is power-normalised so each real component has unit average power, i.e. every
complex symbol has average power 2; the complex noise variance is set to 2*sigma^2 with
sigma^2 = 1/SNR so that SNR = E|x|^2 / E|n|^2 holds.
"""
import math
import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, kind="awgn", snr_db=10.0, rician_k=4.0, equalize=True):
        super().__init__()
        # rayleigh_fs: frequency-selective block fading — an independent CN(0,1) gain per complex
        # symbol dimension (like OFDM sub-carriers), constant over the tokens of a frame; MMSE equalised.
        assert kind in ("awgn", "rician", "rayleigh", "rayleigh_fs", "none")
        self.kind, self.snr_db, self.rician_k, self.equalize = kind, snr_db, rician_k, equalize

    @staticmethod
    def _complex(x):
        B, N, C = x.shape
        return torch.view_as_complex(x.reshape(B, N, C // 2, 2).contiguous())

    @staticmethod
    def _real(z, shape):
        return torch.view_as_real(z).reshape(shape)

    def fading(self, B, device):
        """Block fading coefficient H per image (unit average power)."""
        if self.kind == "rician":
            k = self.rician_k
            los = math.sqrt(k / (k + 1))
            nlos = math.sqrt(1 / (k + 1)) / math.sqrt(2)
            h = los + nlos * (torch.randn(B, device=device) + 1j * torch.randn(B, device=device))
        elif self.kind == "rayleigh":
            h = (torch.randn(B, device=device) + 1j * torch.randn(B, device=device)) / math.sqrt(2)
        else:
            h = torch.ones(B, device=device, dtype=torch.complex64)
        return h.to(torch.complex64)

    def forward(self, t_x, snr_db=None):
        if self.kind == "none":
            return t_x
        snr_db = self.snr_db if snr_db is None else snr_db
        shape = t_x.shape
        z = self._complex(t_x.float())
        sigma = math.sqrt(1.0 / (10 ** (snr_db / 10)))          # per real-dimension noise std
        noise = (torch.randn_like(z.real) + 1j * torch.randn_like(z.real)) * sigma
        if self.kind == "rayleigh_fs":
            B, N, C = z.shape
            h = (torch.randn(B, 1, C, device=z.device) + 1j * torch.randn(B, 1, C, device=z.device)) / math.sqrt(2)
            r = h * z + noise
            if self.equalize:                                   # per-dimension MMSE equalisation (perfect CSI)
                r = r * h.conj() / (h.abs() ** 2 + 2 * sigma ** 2 / 2)
            return self._real(r.to(torch.complex64), shape).to(t_x.dtype)
        h = self.fading(shape[0], t_x.device).view(-1, 1, 1)
        r = h * z + noise
        if self.equalize and self.kind != "awgn":
            r = r / h                                           # perfect-CSI zero-forcing equalisation
        return self._real(r, shape).to(t_x.dtype)
