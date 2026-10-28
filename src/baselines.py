"""Adım 8 — naif baseline'lar: hiçbir şey öğrenmeyen referans modeller (Tablo 1, 2, 3'ün ilk satırları).

    python -m src.baselines

Kalori: her tabağa train ortalaması / medyanı. Medyan, MAE'yi en aza indiren sabit tahmindir.
Sınıflandırma: her görüntüye train'deki en sık sınıf.
Sonuçlar results.csv'ye (val ve test), tahminler results/preds/'e yazılır.
"""
from collections import Counter

import numpy as np

from .config import CALORIE_BIN_EDGES, FOOD101_DIR
from .data import FOOD101_CLASSES, Food101Subset, load_n5k_split, make_val_split, read_dish_metadata
from .metrics import classification_metrics, regression_metrics, to_bins
from .results import log_result, save_preds

NOTE = "Adım 8 naif baseline"


def calorie_baselines():
    meta = read_dish_metadata()
    train, val = make_val_split(load_n5k_split("train", metadata=meta, verbose=False))
    test = load_n5k_split("test", metadata=meta, verbose=False)
    kcal = lambda ids: np.array([meta[d]["calories"] for d in ids])  # noqa: E731
    y_train = kcal(train)
    for name, const in [("naive_mean", y_train.mean()), ("naive_median", np.median(y_train))]:
        config = f"sabit={const:.1f}kcal"
        for split, ids in [("val", val), ("test", test)]:
            y = kcal(ids)
            pred = np.full(len(y), const)
            m = regression_metrics(y, pred, ci=True)
            log_result("calorie", name, config, split, m, notes=NOTE)
            save_preds(name, ids, y, pred, split=split)
            bins = classification_metrics(to_bins(y, CALORIE_BIN_EDGES), to_bins(pred, CALORIE_BIN_EDGES), ci=True)
            log_result("calorie_bins", name, config, split, bins, notes=NOTE)
            print(f"{name:<13} {split:<5} MAE {m['MAE']:6.1f} [{m['MAE_ci_low']:.1f}–{m['MAE_ci_high']:.1f}]  "
                  f"PMAE %{m['PMAE']:.1f}  | Tablo 3 acc {bins['accuracy']:.3f}")


def classification_baselines():
    train = Food101Subset("train", transform=None)
    most = Counter(label for _, label in train.samples).most_common(1)[0][0]
    config = f"sınıf={FOOD101_CLASSES[most]}"
    for split in ("val", "test"):
        ds = Food101Subset(split, transform=None)
        ids = [p.stem for p, _ in ds.samples]
        y = np.array([label for _, label in ds.samples])
        pred = np.full(len(y), most)
        m = classification_metrics(y, pred, ci=True)
        log_result("cls", "naive_most_frequent", config, split, m, notes=NOTE + " (sınıflar dengeli: 1/25)")
        save_preds("naive_most_frequent", ids, y, pred, split=split)
        print(f"naive_most_frequent {split:<5} acc {m['accuracy']:.3f}  macro-F1 {m['f1']:.4f}")


if __name__ == "__main__":
    calorie_baselines()
    if (FOOD101_DIR / "food-101" / "meta" / "train.json").exists():
        classification_baselines()
    else:
        print("Food-101 yok, sınıflandırma baseline'ı atlandı (python scripts/get_food101.py fetch)")
