"""Joint end-to-end training of the Swin multi-user semantic link with mask multiplexing.

Each step draws one independent CIFAR10 image batch per user, multiplexes them on the shared
embedding, passes the channel at a random SNR, and every user decodes its own image:
    L = (1/U) sum_i MSE(X_i, X_i_hat)          (Nget et al. Eq. 10, MSE in place of bin-CE)
Saves tx.pt / rx.pt separately.
"""
import argparse, os, sys, time, random
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, l_s as MAIN_L_S, device as MAIN_DEVICE, wpath   # the ONE configuration (7.9)
from swinsc import Config, Transmitter, Receiver, Channel
from swinsc.data import get_loader
from deepsc_ri.metrics import psnr

p = argparse.ArgumentParser()
p.add_argument("--dataset", default=MAIN["DATASET"], choices=["cifar", "imagenette"])
p.add_argument("--img_size", type=int, default=MAIN["CROP"])
p.add_argument("--stages", type=int, default=MAIN["STAGES"], choices=[2, 3, 4], help="Swin stages: token = (2*2^(stages-1))^2 pixels")
p.add_argument("--users", type=int, default=MAIN["N"][0])
p.add_argument("--mask", default="learned", choices=["learned", "learned_k", "hadamard", "haar", "oma", "deepma"])
p.add_argument("--beta", type=int, default=MAIN["BETA"])
p.add_argument("--l_s", type=int, default=MAIN_L_S(), help="L_s = L / beta; the manuscript's L is l_e")
p.add_argument("--channel", default=MAIN["CHANNEL"], choices=["awgn", "rician", "rayleigh", "rayleigh_fs"])
p.add_argument("--snr_min", type=float, default=MAIN["SNR_TRAIN"][0])
p.add_argument("--snr_max", type=float, default=MAIN["SNR_TRAIN"][1])
p.add_argument("--epochs", type=int, default=MAIN["EPOCHS"][0])
p.add_argument("--bs", type=int, default=MAIN["BATCH"])
p.add_argument("--lr", type=float, default=MAIN["LR"])
p.add_argument("--seed", type=int, default=MAIN["SEED"])
p.add_argument("--out", default=wpath("checkpoints", "swinsc"))
p.add_argument("--resume", action="store_true")
p.add_argument("--amp", action="store_true", help="bf16 autocast")
p.add_argument("--load_cond", action="store_true", help="FiLM-condition the Swin body on the active count (learned_k family)")
p.add_argument("--var_load", action="store_true", help="train with a random number of ACTIVE users per step (underload-robust)")
a = p.parse_args()

random.seed(a.seed); torch.manual_seed(a.seed)
dev = MAIN_DEVICE()
ARCH = {2: ((96, 192), (2, 2), (3, 6)), 3: ((96, 192, 384), (2, 2, 2), (3, 6, 12)), 4: ((96, 192, 384, 768), (2, 2, 2, 2), (3, 6, 12, 24))}
dims, depths, heads = ARCH[a.stages]
cfg = Config(img_size=a.img_size if a.dataset == "imagenette" else 32, users=a.users, mask_type=a.mask, beta=a.beta, l_s=a.l_s, channel=a.channel, load_cond=a.load_cond,
             dims=dims, depths=depths, heads=heads)
os.makedirs(a.out, exist_ok=True); cfg.save(os.path.join(a.out, "config.json"))
tx, rx = Transmitter(cfg).to(dev), Receiver(cfg).to(dev)
ch = Channel(a.channel, rician_k=cfg.rician_k)
# TX and RX each own a UserMasks copy; keep them tied during joint training
if hasattr(tx, "masks"):
    rx.masks = tx.masks          # OMA has no masks
params = [q for n, q in list(tx.named_parameters()) + list(rx.named_parameters())]
seen, uniq = set(), []
for q in params:
    if id(q) not in seen:
        seen.add(id(q)); uniq.append(q)
print(f"TX {sum(q.numel() for q in tx.parameters())/1e6:.2f}M  RX {sum(q.numel() for q in rx.parameters())/1e6:.2f}M  "
      f"users={cfg.users} mask={cfg.mask_type} L_s={cfg.l_s} L_e={cfg.l_e} stages={a.stages} (token={cfg.down}x{cfg.down}px, CBR={cfg.l_e/2/(cfg.down**2*3):.3f}) dataset={a.dataset} img={cfg.img_size}")
opt = torch.optim.AdamW(uniq, a.lr, weight_decay=MAIN["WEIGHT_DECAY"])
loaders = [get_loader(a.dataset, True, a.bs, cfg.img_size) for _ in range(cfg.users)]
test = get_loader(a.dataset, False, 64, cfg.img_size)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(loaders[0]))
start = 0
if a.resume and os.path.exists(os.path.join(a.out, "state.pt")):
    st = torch.load(os.path.join(a.out, "state.pt"), map_location=dev)
    tx.load_state_dict(st["tx"]); rx.load_state_dict(st["rx"]); opt.load_state_dict(st["opt"]); start = st["epoch"]


def run(imgs, snr, active=None):
    active = list(range(len(imgs))) if active is None else active
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
        f, hw = tx([imgs[u] for u in active], active=active)
        z = ch(f.float(), snr)
        outs = [rx(z, u, hw, K=len(active)).float() for u in active]
    return outs, active


for ep in range(start, a.epochs):
    tx.train(); rx.train(); t0 = time.time(); tot = 0
    for batches in zip(*loaders):
        imgs = [b[0].to(dev) for b in batches]
        act = sorted(random.sample(range(cfg.users), random.randint(1, cfg.users))) if a.var_load else None
        outs, act = run(imgs, random.uniform(a.snr_min, a.snr_max), act)
        loss = sum(F.mse_loss(o, imgs[u]) for o, u in zip(outs, act)) / len(act)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(uniq, MAIN["GRAD_CLIP"]); opt.step(); sched.step(); tot += loss.item()
    tx.eval(); rx.eval(); ps = {s: [] for s in (0, 10, 20)}
    with torch.no_grad():
        for i, (x, _) in enumerate(test):
            if i >= 4: break
            x = x.to(dev)
            imgs = [torch.roll(x, u * 37, 0) for u in range(cfg.users)]    # different images per user
            for s in ps:
                outs, _ = run(imgs, s)
                ps[s].append(torch.stack([psnr(o, xi) for o, xi in zip(outs, imgs)]).mean().cpu())
    msg = " ".join(f"PSNR@{s}dB {torch.stack(v).mean():.2f}" for s, v in ps.items())
    print(f"epoch {ep+1}/{a.epochs} loss {tot/len(loaders[0]):.5f} {msg} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(tx.state_dict(), os.path.join(a.out, "tx.pt"))
    torch.save(rx.state_dict(), os.path.join(a.out, "rx.pt"))
    for u in range(cfg.users):
        torch.save(rx.user_state_dict(u), os.path.join(a.out, f"rx_user{u}.pt"))
    torch.save(dict(tx=tx.state_dict(), rx=rx.state_dict(), opt=opt.state_dict(), epoch=ep + 1),
               os.path.join(a.out, "state.pt"))
print("done ->", a.out)
