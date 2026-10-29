"""ORTAK metrikler (§6.2). Tüm deneyler bu dosyayı kullanır.

PMAE ile MAPE aynı şey değil: literatürdeki "yüzde hata" PMAE'dir (MAE / ortalama gerçek değer).
Literatürle karşılaştırırken PMAE kullanın.

Güven aralıkları (bootstrap): test kümesi birkaç yüz tabak olduğu için iki model arasındaki
3–5 kcal'lik fark şans eseri olabilir. "A, B'den iyi" demeden önce `paired_bootstrap` ile bakın
(komut satırından: python -m src.compare results/preds/a.csv results/preds/b.csv).
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, top_k_accuracy_score

from .config import SEED


def mae(y_true, y_pred):
    return float(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float)).mean())


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def regression_metrics(y_true, y_pred, ci=False):
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = p - y
    out = {
        "MAE": mae(y, p),
        "PMAE": float(100 * np.abs(err).mean() / y.mean()),                 # literatürle karşılaştırma için
        "MAPE": float(100 * np.mean(np.abs(err) / np.clip(y, 1, None))),   # 0 kcal'e bölmeyi önler
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "within_100kcal": float(100 * np.mean(np.abs(err) <= 100)),
    }
    if ci:
        out["MAE_ci_low"], out["MAE_ci_high"] = bootstrap_ci(y, p, mae)
    return out


def classification_metrics(y_true, y_pred, y_score=None, ci=False):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p), "recall": float(r), "f1": float(f1)}
    if y_score is not None:  # (N, C) olasılıklar
        out["top5"] = float(top_k_accuracy_score(y_true, y_score, k=5, labels=np.arange(np.shape(y_score)[1])))
    if ci:
        out["f1_ci_low"], out["f1_ci_high"] = bootstrap_ci(y_true, y_pred, macro_f1)
    return out


def calorie_bins(train_calories):
    """Tablo 3 için aralık sınırları: TRAIN kalorilerinin üçte birlik dilimleri (bir kez hesaplanıp sabitlenir)."""
    return np.quantile(train_calories, [1 / 3, 2 / 3])


def to_bins(calories, edges):
    return np.digitize(calories, edges)  # 0 = düşük, 1 = orta, 2 = yüksek


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, alpha=0.05, seed=SEED):
    """Test örneklerini yerine koyarak yeniden örnekler; metriğin %95 güven aralığını (alt, üst) döner."""
    y, p = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    stats = [metric_fn(y[i], p[i]) for i in rng.integers(0, len(y), size=(n_boot, len(y)))]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap(y_true, pred_a, pred_b, metric_fn=mae, n_boot=1000, alpha=0.05, seed=SEED):
    """İki modeli AYNI tabaklar üzerinde karşılaştırır: fark = metrik(A) − metrik(B).

    MAE için fark < 0 ise A daha iyi (F1 gibi yüksek-iyi metriklerde tersi). Güven aralığı 0'ı içermiyorsa fark anlamlıdır
    ("significant": True); içeriyorsa "A daha iyi" demek için yeterli kanıt yok.
    """
    y, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    diffs = [metric_fn(y[i], a[i]) - metric_fn(y[i], b[i]) for i in rng.integers(0, len(y), size=(n_boot, len(y)))]
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": metric_fn(y, a) - metric_fn(y, b), "ci_low": float(lo), "ci_high": float(hi),
            "significant": bool(lo > 0 or hi < 0)}
