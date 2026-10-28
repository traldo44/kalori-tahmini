"""ORTAK sonuç kaydı (§6.3): her deney results/results.csv'ye bir satır, tahminler results/preds/'e."""
import csv
import datetime
import getpass
import os

from .config import RESULTS_DIR

COLUMNS = ["date", "owner", "task", "model", "config", "split",
           "accuracy", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high", "top5",
           "MAE", "MAE_ci_low", "MAE_ci_high", "PMAE", "MAPE", "RMSE", "R2", "within_100kcal", "notes"]


def log_result(task, model, config, split, metrics, notes="", owner=None, path=None):
    """task: cls | calorie | calorie_bins | ablation.  metrics: metrics.py fonksiyonlarının çıktısı."""
    path = path or RESULTS_DIR / "results.csv"
    unknown = set(metrics) - set(COLUMNS)
    if unknown:
        raise ValueError(f"results.csv'de sütunu olmayan metrik: {sorted(unknown)}")
    row = {"date": datetime.date.today().isoformat(),
           "owner": owner or os.environ.get("KALORI_OWNER") or getpass.getuser(),
           "task": task, "model": model, "config": config, "split": split, "notes": notes}
    row.update({k: round(v, 4) for k, v in metrics.items()})
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _plain(v):
    """numpy sayısını düz Python sayısına çevirir (sınıf etiketleri int kalır, kcal float)."""
    return v.item() if hasattr(v, "item") else v


def save_preds(model, ids, y_true, y_pred, split="test", path=None):
    """results/preds/<model>.csv (test) veya <model>_<split>.csv: id, y_true, y_pred."""
    name = model if split == "test" else f"{model}_{split}"
    path = path or RESULTS_DIR / "preds" / f"{name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "y_true", "y_pred"])
        writer.writerows(zip(ids, map(_plain, y_true), map(_plain, y_pred)))
    return path
