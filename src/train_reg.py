"""Adım 11–13 — kalori regresyonu (Nutrition5k, yalnızca üstten RGB).

    python -m src.train_reg --backbone resnet50                                   # yalnızca kalori (Adım 11)
    python -m src.train_reg --backbone resnet50 --targets calories mass fat carb protein   # çok görevli (Adım 12)
    python -m src.train_reg --backbone resnet50 --no-aug                          # ablation: artırmasız baseline
    python -m src.train_reg --backbone resnet50 --epochs 1 --limit 64 --dry-run   # hızlı deneme (kayıt yok)

Kayıp: ölçeklenmiş hedeflerde L1 (EDA kararı: ham kcal + L1; --target-transform log1p --loss huber ablation için).
Model seçimi ve erken durdurma val kalori MAE'siyle; test en sonda bir kez ölçülür.
Çıktılar: results.csv (val + test, Tablo 2 ve 3), results/preds/<ad>.csv, results/history/<ad>.csv,
checkpoints/<ad>.pt.
"""
import argparse

import numpy as np
import torch

from .config import CALORIE_BIN_EDGES, CHECKPOINT_DIR, IMG_SIZE, set_seed, get_device
from .data import TARGETS, Nutrition5kDataset, load_n5k_split, make_val_split, read_dish_metadata
from .metrics import classification_metrics, regression_metrics, to_bins
from .models import build_model, decode_target, encode_target, model_norm, save_checkpoint
from .results import log_result, save_preds
from .trainer import fit, history_path, loader, predict
from .transforms import calorie_transform


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--targets", nargs="+", default=["calories"], choices=TARGETS)
    ap.add_argument("--target-transform", default="raw", choices=["raw", "log1p"])
    ap.add_argument("--loss", default="l1", choices=["l1", "huber"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--no-aug", action="store_true", help="veri artırmasız (ablation baseline'ı)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--name", default=None, help="results.csv'deki model adı (varsayılan: ayarlardan üretilir)")
    ap.add_argument("--limit", type=int, default=None, help="her bölmeden ilk N tabak (yalnızca deneme)")
    ap.add_argument("--dry-run", action="store_true", help="results.csv / preds / checkpoint yazma")
    args = ap.parse_args()
    if args.limit and not args.dry_run:
        ap.error("--limit yalnızca --dry-run ile kullanılabilir (yarım veriyle sonuç kaydedilmesin)")
    if "calories" not in args.targets:
        ap.error("--targets kaloriyi içermeli")
    targets = ["calories"] + [t for t in args.targets if t != "calories"]  # kalori her zaman 0. çıktı

    set_seed()
    device = get_device()
    name = args.name or "_".join([args.backbone, "reg"] + (["mt"] if len(targets) > 1 else [])
                                 + (["log1p"] if args.target_transform == "log1p" else [])
                                 + (["huber"] if args.loss == "huber" else []) + (["noaug"] if args.no_aug else []))
    print(f"{name} · cihaz {device}", flush=True)

    meta = read_dish_metadata()
    train_ids, val_ids = make_val_split(load_n5k_split("train", metadata=meta, verbose=False))
    test_ids = load_n5k_split("test", metadata=meta, verbose=False)
    if args.limit:
        train_ids, val_ids, test_ids = train_ids[:args.limit], val_ids[:args.limit], test_ids[:args.limit]

    model = build_model(args.backbone, n_outputs=len(targets), pretrained=True).to(device)
    mean, std = model_norm(model)
    y_train = torch.tensor([[meta[d][t] for t in targets] for d in train_ids])
    scale = encode_target(y_train, args.target_transform).mean(0)  # her hedef ~1 civarında olsun

    def make(ids, train):
        tf = calorie_transform(train=train and not args.no_aug, size=args.img_size, mean=mean, std=std)
        return loader(Nutrition5kDataset(ids, meta, tf, targets), args.batch_size, train, args.workers)

    train_dl, val_dl, test_dl = make(train_ids, True), make(val_ids, False), make(test_ids, False)
    base_loss = torch.nn.L1Loss() if args.loss == "l1" else torch.nn.HuberLoss(delta=0.5)
    loss_fn = lambda out, y: base_loss(out, encode_target(y, args.target_transform, scale.to(y.device)))  # noqa: E731

    def decoded(dl):
        out, y = predict(model, dl, device)
        return decode_target(out, args.target_transform, scale).clamp(min=0).numpy(), y.numpy()

    def val_mae(_model):
        pred, y = decoded(val_dl)
        return float(np.abs(pred[:, 0] - y[:, 0]).mean())

    best, best_epoch, _ = fit(model, train_dl, val_dl, loss_fn, val_mae, epochs=args.epochs,
                              lr_backbone=args.lr_backbone, lr_head=args.lr_head, weight_decay=args.weight_decay,
                              patience=args.patience, higher_is_better=False, device=device,
                              history_path=None if args.dry_run else history_path(name))

    config = (f"{args.backbone}.{model.pretrained_cfg.get('tag', '')}_lr{args.lr_backbone:g}/{args.lr_head:g}"
              f"_bs{args.batch_size}_{args.img_size}px_ep{best_epoch}/{args.epochs}_{args.loss}_{args.target_transform}"
              + ("_noaug" if args.no_aug else ""))
    results = {}
    for split, dl, ids in (("val", val_dl, val_ids), ("test", test_dl, test_ids)):
        pred, y = decoded(dl)
        m = regression_metrics(y[:, 0], pred[:, 0], ci=True)
        extra = ", ".join(f"{t} MAE {np.abs(pred[:, i] - y[:, i]).mean():.1f}" for i, t in enumerate(targets) if i)
        notes = "Adım 11–13 transfer CNN" + (f"; {extra}" if extra else "")
        bins = classification_metrics(to_bins(y[:, 0], CALORIE_BIN_EDGES), to_bins(pred[:, 0], CALORIE_BIN_EDGES), ci=True)
        results[split] = m
        print(f"  ✓ {split}: MAE {m['MAE']:.1f} [{m['MAE_ci_low']:.1f}–{m['MAE_ci_high']:.1f}]  PMAE %{m['PMAE']:.1f}  "
              f"R² {m['R2']:.2f}  | Tablo 3 F1 {bins['f1']:.3f}" + (f"  | {extra}" if extra else ""), flush=True)
        if not args.dry_run:
            log_result("calorie", name, config, split, m, notes=notes)
            log_result("calorie_bins", name, config, split, bins, notes=notes)
            save_preds(name, ids, y[:, 0], pred[:, 0], split=split)
    if not args.dry_run:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        save_checkpoint(CHECKPOINT_DIR / f"{name}.pt", model, args.backbone, outputs=targets,
                        target=args.target_transform, scale=scale, mean=mean, std=std,
                        val_MAE=results["val"]["MAE"], test_MAE=results["test"]["MAE"], config=config)
        print(f"  kaydedildi: checkpoints/{name}.pt", flush=True)


if __name__ == "__main__":
    main()
