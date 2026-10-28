"""Dış test: kendi çektiğimiz fotoğraflar (README → "Kendi fotoğraflarımızla test").

    python -m src.eval_own_photos --ckpt checkpoints/best.pt --model resnet50_reg

own_photos/labels.csv sütunları: file, calories, angle (ust | yan), source, notes
Sonuç results.csv'ye split=own_photos (tümü) ve own_photos_ust / own_photos_yan olarak yazılır.
"""
import argparse

import pandas as pd
from PIL import Image

from .config import ROOT, get_device
from .metrics import regression_metrics
from .models import load_checkpoint, predict_image
from .results import log_result, save_preds

OWN_DIR = ROOT / "own_photos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True, help="results.csv'deki model adı, ör. resnet50_reg")
    ap.add_argument("--config", default="")
    ap.add_argument("--dir", default=str(OWN_DIR))
    args = ap.parse_args()

    df = pd.read_csv(f"{args.dir}/labels.csv")
    device = get_device()
    model, meta = load_checkpoint(args.ckpt, device)
    df["pred"] = [predict_image(model, meta, Image.open(f"{args.dir}/{f}"), device)["calories"] for f in df["file"]]

    parts = [("own_photos", df)] + [(f"own_photos_{a}", g) for a, g in df.groupby("angle")]
    for split, part in parts:
        if len(part) < 2:
            continue
        m = regression_metrics(part["calories"], part["pred"], ci=len(part) >= 10)
        log_result("calorie", args.model, args.config, split, m, notes=f"{len(part)} fotoğraf")
        print(f"{split:<16} n={len(part):<3} MAE={m['MAE']:.1f} kcal  PMAE=%{m['PMAE']:.1f}")
    save_preds(args.model, df["file"], df["calories"], df["pred"], split="own_photos")


if __name__ == "__main__":
    main()
