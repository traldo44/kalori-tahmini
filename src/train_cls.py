"""Adım 10 ve 13 — yemek türü sınıflandırması (Food-101, 25 sınıf).

    python -m src.train_cls --backbone resnet50
    python -m src.train_cls --backbone efficientnet_b0
    python -m src.train_cls --backbone resnet50 --epochs 1 --limit 64 --dry-run   # hızlı deneme (kayıt yok)

Kayıp: cross-entropy + label smoothing 0,1 (train etiketleri gürültülü, bkz. Food-101 EDA).
Model seçimi ve erken durdurma val macro-F1 ile; test en sonda bir kez ölçülür.
Tablo 1 için parametre sayısı ve çıkarım süresi notes sütununa yazılır.
"""
import argparse

import torch

from .config import CHECKPOINT_DIR, IMG_SIZE, get_device, set_seed
from .data import FOOD101_CLASSES, Food101Subset
from .metrics import classification_metrics
from .models import build_model, model_norm, save_checkpoint
from .results import log_result, save_preds
from .trainer import fit, history_path, loader, ms_per_image, predict
from .transforms import classification_transform


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--limit", type=int, default=None, help="her bölmeden ilk N görüntü (yalnızca deneme)")
    ap.add_argument("--dry-run", action="store_true", help="results.csv / preds / checkpoint yazma")
    args = ap.parse_args()
    if args.limit and not args.dry_run:
        ap.error("--limit yalnızca --dry-run ile kullanılabilir (yarım veriyle sonuç kaydedilmesin)")

    set_seed()
    device = get_device()
    name = args.name or f"{args.backbone}_cls"
    print(f"{name} · cihaz {device}", flush=True)

    model = build_model(args.backbone, n_outputs=len(FOOD101_CLASSES), pretrained=True).to(device)
    mean, std = model_norm(model)

    def make(split, train):
        ds = Food101Subset(split, classification_transform(train, args.img_size, mean, std))
        if args.limit:  # sınıflar sıralı → deneme için her sınıftan eşit örnek al
            ds.samples = ds.samples[::max(1, len(ds.samples) // args.limit)][:args.limit]
        return ds, loader(ds, args.batch_size, train, args.workers)

    (_, train_dl), (val_ds, val_dl), (test_ds, test_dl) = make("train", True), make("val", False), make("test", False)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def val_f1(_model):
        out, y = predict(model, val_dl, device)
        return classification_metrics(y.numpy(), out.argmax(1).numpy())["f1"]

    best, best_epoch, _ = fit(model, train_dl, val_dl, loss_fn, val_f1, epochs=args.epochs,
                              lr_backbone=args.lr_backbone, lr_head=args.lr_head, weight_decay=args.weight_decay,
                              patience=args.patience, higher_is_better=True, device=device,
                              history_path=None if args.dry_run else history_path(name))

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    ms = ms_per_image(model, device, args.img_size)
    config = (f"{args.backbone}.{model.pretrained_cfg.get('tag', '')}_lr{args.lr_backbone:g}/{args.lr_head:g}"
              f"_bs{args.batch_size}_{args.img_size}px_ep{best_epoch}/{args.epochs}_ls{args.label_smoothing:g}")
    notes = f"Adım 10/13 transfer CNN; {n_params:.1f}M parametre; {ms:.2f} ms/görüntü ({device.type})"
    results = {}
    for split, ds, dl in (("val", val_ds, val_dl), ("test", test_ds, test_dl)):
        out, y = predict(model, dl, device)
        m = classification_metrics(y.numpy(), out.argmax(1).numpy(), out.softmax(1).numpy(), ci=True)
        results[split] = m
        print(f"  ✓ {split}: acc {m['accuracy']:.3f}  F1 {m['f1']:.3f} [{m['f1_ci_low']:.3f}–{m['f1_ci_high']:.3f}]"
              f"  top5 {m['top5']:.3f}", flush=True)
        if not args.dry_run:
            log_result("cls", name, config, split, m, notes=notes)
            save_preds(name, [f"{p.parent.name}/{p.stem}" for p, _ in ds.samples], y.numpy(), out.argmax(1).numpy(), split=split)
    print(f"  {n_params:.1f}M parametre · {ms:.2f} ms/görüntü ({device.type})", flush=True)
    if not args.dry_run:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        save_checkpoint(CHECKPOINT_DIR / f"{name}.pt", model, args.backbone, outputs=FOOD101_CLASSES, target="cls",
                        mean=mean, std=std, val_F1=results["val"]["f1"], test_F1=results["test"]["f1"], config=config)
        print(f"  kaydedildi: checkpoints/{name}.pt", flush=True)


if __name__ == "__main__":
    main()
