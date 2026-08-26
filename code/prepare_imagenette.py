"""Download Imagenette (160px, 10 ImageNet classes, ~13k images) from the Hugging Face parquet
mirror and export to ImageFolder layout:  data/imagenette160/{train,val}/<class>/*.png
Also copies a few validation images into web/samples/, and into --win_dir when one is given."""
import argparse, io, os, sys, subprocess
import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["tench", "English_springer", "cassette_player", "chain_saw", "church",
           "French_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute"]
URL = "https://huggingface.co/datasets/frgfm/imagenette/resolve/refs%2Fconvert%2Fparquet/160px/{split}/0000.parquet"

p = argparse.ArgumentParser()
p.add_argument("--out", default=os.path.join(ROOT, "data", "imagenette160"))
p.add_argument("--samples", type=int, default=3, help="val images per class copied to web/samples")
p.add_argument("--win_dir", default="",
               help="optional extra directory to copy the sample images into")
a = p.parse_args()

pdir = os.path.join(ROOT, "data", "imagenette_parquet"); os.makedirs(pdir, exist_ok=True)
for split in ("train", "validation"):
    f = os.path.join(pdir, f"{split}.parquet")
    if not os.path.exists(f) or os.path.getsize(f) < 1e6:
        print("downloading", split, flush=True)
        subprocess.run(["curl", "-sL", "-o", f, URL.format(split=split)], check=True)

samples_dir = os.path.join(ROOT, "web", "samples"); os.makedirs(samples_dir, exist_ok=True)
if a.win_dir:
    os.makedirs(a.win_dir, exist_ok=True)
for split, out in (("train", "train"), ("validation", "val")):
    t = pq.read_table(os.path.join(pdir, f"{split}.parquet"))
    imgs, labels = t.column("image").to_pylist(), t.column("label").to_pylist()
    per_class = {}
    for i, (im, l) in enumerate(zip(imgs, labels)):
        cls = CLASSES[l]
        d = os.path.join(a.out, out, cls); os.makedirs(d, exist_ok=True)
        img = Image.open(io.BytesIO(im["bytes"])).convert("RGB")
        img.save(os.path.join(d, f"{i:05d}.png"))
        if out == "val" and per_class.get(l, 0) < a.samples:
            k = per_class.get(l, 0); per_class[l] = k + 1
            name = f"imagenette_{cls}_{k}.png"
            img.save(os.path.join(samples_dir, name))
            if a.win_dir:
                img.save(os.path.join(a.win_dir, name))
        if i % 2000 == 0:
            print(split, i, "/", len(labels), flush=True)
    print(split, "done:", len(labels), "images ->", os.path.join(a.out, out), flush=True)
print("ALL DONE")
