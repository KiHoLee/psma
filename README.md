# Load-Adaptive Shared-Embedding Multiple Access — Reproducibility Package

Code, raw results, and figures for the manuscript *"Load-Adaptive
Mask-Based Multiple Access for Multi-User Semantic
Communications"* (submitted to IEEE Transactions on Signal Processing).

Mask-based superposition of Swin-transformer image embeddings with
variable-load training, compared against re-encoded orthogonal
multiple access (OMA) and token-domain multiple access (ToDMA) on
Rayleigh block-fading channels at a fixed channel bandwidth ratio.

## Requirements

- Python >= 3.12, PyTorch >= 2.10 (CUDA build), torchvision, pillow,
  pyarrow, matplotlib
- One GPU with >= 12 GB memory (results produced on an NVIDIA RTX
  A4500, PyTorch 2.11 + cu130)
- Dataset: Imagenette-160 (downloaded automatically by
  `scripts/prepare_imagenette.py` from the Hugging Face parquet mirror)

## Reproducing the results

All randomness is seeded (`seed = 0`). One script per experiment; the
plot script reads only the stored CSV.

```bash
python scripts/prepare_imagenette.py                 # dataset -> data/imagenette160

# --- training (one model per call; ~30-60 min each on an A4500) ---
python scripts/swin_train.py --dataset imagenette --img_size 128 --users 2 \
    --mask learned --l_s 8 --beta 1 --channel rayleigh --epochs 20 --amp --bs 24 \
    --out checkpoints/swinsc_ov_u2_learned
#   ... repeat with --users 4, --mask oma, and --var_load for the
#   variable-load model (see the table below for the exact flag sets)
python scripts/todma_train.py --l_s 8 --v 256 --epochs 15   # ToDMA VQ autoencoder

# --- evaluation grid (writes data/tsp_eval.csv) ---
python scripts/tsp_eval.py --out data/tsp_eval.csv

# --- companion evaluations ---
python scripts/sinr_model.py --out data/sinr_model.csv    # Prop.-2 model with trained masks (also dumps beta)
python scripts/ssim_eval.py  --out data/ssim_eval.csv     # SSIM at the 10 dB point (Table VI)
python scripts/visual_eval.py --out data/visual           # qualitative panels (Figs. 8-9)

# --- figures (read only data/, write fig/*.pdf) ---
python plot/plot_results.py
```

## Model table (flag sets)

| Checkpoint | Flags |
|---|---|
| `swinsc_ov_u2_learned` | `--users 2 --mask learned` |
| `swinsc_ov_u2_oma` | `--users 2 --mask oma` |
| `swinsc_ov_u4_learned` | `--users 4 --mask learned` |
| `swinsc_ov_u4_oma` | `--users 4 --mask oma` |
| `swinsc_ov_u4var_learned` | `--users 4 --mask learned --var_load` |
| `todma_v256` | `todma_train.py --l_s 8 --v 256` |
| dimension-rich pair | `--users 2 --mask learned/oma --l_s 16 --beta 2 --channel rayleigh` |

All runs share: Imagenette 128x128 crops, Rayleigh block fading with
perfect-CSI zero forcing, training SNR Uniform[0, 20] dB, Adam
3e-4 with cosine decay, bf16 autocast, batch 24 per user, seed 0.

## Figure -> script -> data

| Figure | Script | Data |
|---|---|---|
| fig_snr_u2.pdf (full load, N=K=2) | plot/plot_results.py | data/tsp_eval.csv |
| fig_snr_u4.pdf (overload, N=K=4) | plot/plot_results.py | data/tsp_eval.csv |
| fig_underload.pdf (K=1..4 on N=4) | plot/plot_results.py | data/tsp_eval.csv |
| fig_snr_rich.pdf (dimension-rich, L=32) | plot/plot_results.py | data/tsp_eval.csv |
| fig_sinr_model.pdf (SINR model) | plot/plot_results.py | data/sinr_model.csv |
| fig_masks.pdf (trained-mask heatmaps) | plot/plot_results.py | data/masks.csv |
| fig_visual_overload.pdf / fig_visual_underload.pdf | scripts/visual_eval.py | data/visual/visual_psnr.csv |
| Table VI (SSIM) | scripts/ssim_eval.py | data/ssim_eval.csv |

`data/tsp_eval.csv` columns: `scheme, designed, active, snr, psnr`
(mean per-user PSNR in dB over 200 validation images per user per
point).

## Conventions

- CBR = complex channel symbols / source values = L/96 for the
  2-stage encoder (token = 4x4 pixels = 48 values, L real dims/token).
- Frame power is normalized to 1 over the ACTIVE users; the noise
  variance per real dimension is 1/SNR; Rayleigh gain is constant per
  frame and zero-forced at the receiver.
- ToDMA follows Qiao et al. (IEEE INFOCOM WKSHPS 2025): fixed random
  Gaussian signatures per vocabulary entry, OMP support recovery,
  genie-aided association (an upper bound on any practical
  association).

## License / citation

To be finalized upon publication. Until then the repository is shared
for review purposes.
