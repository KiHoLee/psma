"""ToDMA-style baseline, stage 1: single-user VQ autoencoder (token-domain codebook).

Each Swin token is quantised to one of V codewords (dim L_e); the codeword vector itself is the
transmit signal. Multiple access is training-free (see todma_eval.py): users superpose their
codewords on the shared dimensions and the receiver detects the active codewords by OMP.

    python code/todma_train.py
"""
import argparse, os, sys, time, random
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, device as MAIN_DEVICE, wpath            # the ONE configuration (standard 7.9)
from swinsc import Config
from swinsc.swin import SwinEncoder, SwinDecoder
from swinsc.data import get_loader
from deepsc_ri.metrics import psnr

p = argparse.ArgumentParser()
# beta = 1 here, so the codeword dim IS the manuscript's L
p.add_argument("--l_s", type=int, default=MAIN["L"], help="codeword dim = transmitted reals per token")
p.add_argument("--v", type=int, default=MAIN["VOCAB"], help="codebook size")
p.add_argument("--img_size", type=int, default=MAIN["CROP"])
p.add_argument("--epochs", type=int, default=MAIN["EPOCHS"][0])
p.add_argument("--bs", type=int, default=MAIN["BATCH"])
p.add_argument("--lr", type=float, default=MAIN["LR"])
p.add_argument("--seed", type=int, default=MAIN["SEED"])
p.add_argument("--out", default=wpath("checkpoints", "todma_v256"))
a = p.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
dev = MAIN_DEVICE()
cfg = Config(img_size=a.img_size, l_s=a.l_s, beta=1, users=1, mask_type="learned")


class VQ(torch.nn.Module):
    def __init__(self, v, d):
        super().__init__()
        self.codebook = torch.nn.Parameter(torch.randn(v, d) / d ** 0.5)

    def forward(self, s):
        # s: (B,N,d) -> nearest codeword (straight-through)
        d2 = (s.pow(2).sum(-1, keepdim=True) - 2 * s @ self.codebook.t()
              + self.codebook.pow(2).sum(-1))
        idx = d2.argmin(-1)
        q = self.codebook[idx]
        loss = F.mse_loss(q, s.detach()) + 0.25 * F.mse_loss(s, q.detach())
        return s + (q - s).detach(), idx, loss


enc = SwinEncoder(cfg.in_ch, cfg.patch, cfg.dims, cfg.depths, cfg.heads, cfg.window, a.l_s).to(dev)
dec = SwinDecoder(cfg.in_ch, cfg.patch, tuple(reversed(cfg.dims)), tuple(reversed(cfg.depths)),
                  tuple(reversed(cfg.heads)), cfg.window, a.l_s).to(dev)
vq = VQ(a.v, a.l_s).to(dev)
params = list(enc.parameters()) + list(dec.parameters()) + list(vq.parameters())
opt = torch.optim.AdamW(params, a.lr, weight_decay=MAIN["WEIGHT_DECAY"])
tr = get_loader(MAIN["DATASET"], True, a.bs, a.img_size)
te = get_loader("imagenette", False, 64, a.img_size)
print(f"ToDMA VQ: V={a.v} d={a.l_s} enc {sum(q.numel() for q in enc.parameters())/1e6:.2f}M", flush=True)

for ep in range(a.epochs):
    enc.train(); dec.train(); t0 = time.time(); tot = 0
    for x, _ in tr:
        x = x.to(dev)
        s, hw = enc(x)
        q, idx, vq_loss = vq(s)
        out = dec(q, hw)
        loss = F.mse_loss(out, x) + vq_loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); tot += loss.item()
    enc.eval(); dec.eval(); ps = []
    with torch.no_grad():
        for i, (x, _) in enumerate(te):
            if i >= 4: break
            x = x.to(dev); s, hw = enc(x); q, idx, _ = vq(s)
            ps.append(psnr(dec(q, hw), x).cpu())
    used = len(torch.unique(idx))
    print(f"epoch {ep+1}/{a.epochs} loss {tot/len(tr):.4f} clean-PSNR {torch.cat(ps).mean():.2f} "
          f"codebook-used {used}/{a.v} ({time.time()-t0:.0f}s)", flush=True)

os.makedirs(a.out, exist_ok=True)
cfg.save(os.path.join(a.out, "config.json"))
torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "vq": vq.state_dict(),
            "v": a.v, "l_s": a.l_s}, os.path.join(a.out, "model.pt"))
print("done ->", a.out)
