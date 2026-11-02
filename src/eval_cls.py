"""Kayıtlı bir sınıflandırma checkpoint'ini Food-101 test/val'de yeniden değerlendir: ROC-AUC dahil (hocanın istediği
metrik; ilk eğitimlerde sınıf olasılıkları kaydedilmediği için AUC hesaplanamamıştı) ve olasılıkları sakla.

    python -m src.eval_cls --ckpt checkpoints/vit_b16_384_cls.pt
    python -m src.eval_cls --ckpt checkpoints/convnext_tiny_cls.pt --name convnext_tiny_cls

Çıktılar: results.csv'ye yeni `cls` satırları (aynı model adı; make_figures sonuncuyu alır), results/probs/<ad>_<split>.npz
(ids, y_true, probs float16) — ROC eğrisi ve ileride başka metrikler için. Tahmin dosyaları (preds) değişmez.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .config import IMG_SIZE, RESULTS_DIR, get_device
from .data import FOOD101_CLASSES, Food101Subset
from .metrics import classification_metrics
from .models import load_checkpoint
from .results import log_result
from .trainer import loader, ms_per_image, predict
from .transforms import classification_transform

PROBS_DIR = RESULTS_DIR / "probs"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default=None, help="results.csv'deki model adı (varsayılan: checkpoint dosya adı)")
    ap.add_argument("--img-size", type=int, default=None, help="varsayılan: checkpoint'teki config'den (…_384px_…) ya da 224")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    device = get_device()
    name = args.name or Path(args.ckpt).stem
    model, meta = load_checkpoint(args.ckpt, device)
    assert meta["target"] == "cls" and list(meta["outputs"]) == list(FOOD101_CLASSES), "sınıflandırma checkpoint'i değil"
    config = meta["info"].get("config", "")
    size = args.img_size or next((int(t[:-2]) for t in config.split("_") if t.endswith("px")), IMG_SIZE)
    ms = ms_per_image(model, device, size)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    notes = f"yeniden değerlendirme (ROC-AUC, olasılıklar results/probs/); {n_params:.1f}M parametre; {ms:.2f} ms/görüntü ({device.type})"
    print(f"{name} · {size} px · cihaz {device}", flush=True)

    PROBS_DIR.mkdir(exist_ok=True)
    for split in args.splits:
        ds = Food101Subset(split, classification_transform(False, size, meta["mean"], meta["std"]))
        out, y = predict(model, loader(ds, args.batch_size, False, args.workers), device)
        probs = out.softmax(1).numpy()
        m = classification_metrics(y.numpy(), probs.argmax(1), probs, ci=True)
        print(f"  ✓ {split}: acc {m['accuracy']:.3f}  F1 {m['f1']:.3f} [{m['f1_ci_low']:.3f}–{m['f1_ci_high']:.3f}]"
              f"  ROC-AUC {m['roc_auc']:.4f}  top5 {m['top5']:.3f}", flush=True)
        if not args.dry_run:
            log_result("cls", name, config, split, m, notes=notes)
            ids = np.array([f"{p.parent.name}/{p.stem}" for p, _ in ds.samples])
            np.savez_compressed(PROBS_DIR / f"{name}_{split}.npz", ids=ids, y_true=y.numpy().astype(np.int16),
                                probs=probs.astype(np.float16), classes=np.array(FOOD101_CLASSES))
            print(f"  → results/probs/{name}_{split}.npz", flush=True)


if __name__ == "__main__":
    main()
