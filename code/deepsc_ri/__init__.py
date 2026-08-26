"""DeepSC-RI: Robust image semantic communication with multi-scale ViT.

Peng et al., "A Robust Image Semantic Communication System With Multi-Scale
Vision Transformer," IEEE JSAC, vol. 43, no. 4, Apr. 2025.

Package layout (transmitter / receiver are fully separable):
    config.py       hyper-parameters (Table II of the paper)
    layers.py       shared building blocks (patch embedding, ViT block, ...)
    transmitter.py  fine-/coarse-grained extractors + fusion + channel encoder
    receiver.py     channel decoder + semantic decoder
    channel.py      AWGN / Rician physical channel
    impairment.py   PGD semantic impairment + ISII metric (Eq. 6)
    metrics.py      PSNR / LPIPS / accuracy
"""
from .config import Config
from .transmitter import Transmitter
from .receiver import Receiver
from .channel import Channel

__all__ = ["Config", "Transmitter", "Receiver", "Channel"]
