"""Performance metrics (Sec. IV-D): PSNR (Eq. 27), LPIPS (Eq. 29), accuracy (Eq. 30)."""
import torch
import torch.nn.functional as F


def psnr(x, y, max_val=1.0):
    """Per-image PSNR in dB. x, y in [0,1]."""
    mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(max_val ** 2 / mse)


class LPIPS:
    """Thin wrapper around the `lpips` package (pip install lpips). Falls back to None if absent."""

    def __init__(self, device="cpu", net="alex"):
        try:
            import lpips
            self.fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        except ImportError:
            self.fn = None

    @torch.no_grad()
    def __call__(self, x, y):
        if self.fn is None:
            return torch.full((x.shape[0],), float("nan"), device=x.device)
        return self.fn(x * 2 - 1, y * 2 - 1).flatten()


@torch.no_grad()
def accuracy(classifier, x, y):
    return (classifier(x).argmax(1) == y).float()
