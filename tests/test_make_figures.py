"""Adım 15 figür betiğinin saf yardımcıları + sentetik verilerle uçtan uca duman testi (results/'a dokunmaz)."""
import csv

import numpy as np
import pandas as pd
import pytest

from src import make_figures as mf
from src.results import COLUMNS


def test_confused_pairs_counts_off_diagonal_only():
    classes = ["a", "b", "c_d"]
    y_true = [0, 0, 0, 1, 1, 2, 2, 2]
    y_pred = [0, 2, 2, 0, 1, 0, 2, 2]
    pairs = mf.confused_pairs(y_true, y_pred, classes, k=10)
    assert pairs.iloc[0].tolist() == ["a", "c d", 2]          # en sık çift önce, alt çizgi boşluğa döner
    assert set(zip(pairs["true"], pairs["pred"])) == {("a", "c d"), ("b", "a"), ("c d", "a")}
    assert len(mf.confused_pairs(y_true, y_pred, classes, k=1)) == 1


def test_confused_pairs_empty_when_perfect():
    assert mf.confused_pairs([0, 1], [0, 1], ["a", "b"]).empty


def test_per_bin_mae_uses_true_calorie_bins():
    y_true = [50, 100, 200, 300, 500, 900]   # düşük, düşük, orta, orta, yüksek, yüksek (sınır 116,1 / 313,7)
    y_pred = [60, 90, 220, 330, 400, 1000]
    t = mf.per_bin_mae(y_true, y_pred, edges=(116.1, 313.7))
    assert t["bin"].tolist() == ["düşük", "orta", "yüksek"]
    assert t["n"].tolist() == [2, 2, 2]
    assert t["MAE"].tolist() == pytest.approx([10, 25, 100])


def test_per_bin_mae_empty_bin_is_nan():
    t = mf.per_bin_mae([10, 20], [10, 25], edges=(116.1, 313.7))
    assert t["n"].tolist() == [2, 0, 0] and np.isnan(t["MAE"].iloc[1])


def test_fmt_tr_turkish_decimal():
    assert mf.fmt_tr(41.4435) == "41,4"
    assert mf.fmt_tr(0.9506, 3) == "0,951"
    assert mf.fmt_tr(65.93, 1, percent=True) == "%65,9"
    assert mf.fmt_tr(-0.0004, 2) == "0,00"      # -0,00 olmaz
    assert mf.fmt_tr(float("nan")) == "—"


def test_md_table_format():
    md = mf.md_table(["Model", "MAE"], [["A", "41,4"], ["B", "47,3"]])
    assert md.splitlines() == ["| Model | MAE |", "| --- | --- |", "| A | 41,4 |", "| B | 47,3 |"]


def test_worst_k_sorted_by_abs_error():
    preds = pd.DataFrame({"id": ["d1", "d2", "d3", "d4"], "y_true": [100, 200, 300, 400], "y_pred": [110, 150, 300, 500]})
    w = mf.worst_k(preds, k=2)
    assert w["id"].tolist() == ["d4", "d2"]
    assert w["abs_error"].tolist() == [100, 50]


def test_has_fish_exact_match_and_top_ingredients():
    ingr = {"d1": [("fish", 54.0, 48.6), ("pizza", 125.0, 300.0)], "d2": [("tuna salad", 228.0, 400.0)]}
    assert mf.has_fish(ingr, "d1") and not mf.has_fish(ingr, "d2") and not mf.has_fish(ingr, "yok")
    assert mf.top_ingredients(ingr, "d1", n=2) == "pizza (125 g), fish (54 g)"


# ---------------------------------------------------------------- duman testi

def _write_results(path, cls_models, reg_models):
    rng = np.random.default_rng(0)
    rows = []
    for m in cls_models:
        rows.append({"task": "cls", "model": m, "split": "test", "accuracy": 0.9, "precision": 0.9, "recall": 0.9,
                     "f1": 0.9, "f1_ci_low": 0.88, "f1_ci_high": 0.92, "top5": 0.99})
    for m in reg_models:
        mae = float(rng.uniform(40, 100))
        rows.append({"task": "calorie", "model": m, "split": "test", "MAE": mae, "MAE_ci_low": mae - 5, "MAE_ci_high": mae + 5,
                     "PMAE": 20.0, "MAPE": 30.0, "RMSE": mae * 1.5, "R2": 0.8, "within_100kcal": 80.0})
        rows.append({"task": "calorie_bins", "model": m, "split": "test", "accuracy": 0.8, "precision": 0.8, "recall": 0.8,
                     "f1": 0.8, "f1_ci_low": 0.75, "f1_ci_high": 0.85})
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    df[["date", "owner", "config", "notes"]] = ["2026-01-01", "test", "", ""]
    df.to_csv(path, index=False)


def test_main_smoke_on_synthetic_data(tmp_path):
    rng = np.random.default_rng(0)
    n_cls, n_reg = 25 * 8, 60
    classes = mf.FOOD101_CLASSES
    cls_models = mf.TABLO1_MODELS
    reg_models = sorted(set(mf.TABLO2_MODELS + mf.COMPARE_MODELS + mf.HISTORY_REG))
    (tmp_path / "preds").mkdir(), (tmp_path / "history").mkdir()
    (tmp_path / "n5k" / "metadata").mkdir(parents=True)
    dish_ids = [f"dish_{1560000000 + i}" for i in range(n_reg)]
    y_cal = rng.uniform(0, 900, n_reg)
    # sınıflandırma tahminleri (bazı hatalarla)
    y_true = np.repeat(np.arange(25), n_cls // 25)
    for m in cls_models:
        y_pred = np.where(rng.random(n_cls) < 0.85, y_true, rng.integers(0, 25, n_cls))
        pd.DataFrame({"id": [f"{classes[t]}/{i}" for i, t in enumerate(y_true)], "y_true": y_true, "y_pred": y_pred}
                     ).to_csv(tmp_path / "preds" / f"{m}.csv", index=False)
    for m in reg_models:
        pd.DataFrame({"id": dish_ids, "y_true": y_cal, "y_pred": y_cal + rng.normal(0, 60, n_reg)}
                     ).to_csv(tmp_path / "preds" / f"{m}.csv", index=False)
    for m in mf.HISTORY_REG:
        pd.DataFrame({"epoch": range(1, 6), "train_loss": 0.3, "val_score": rng.uniform(40, 90, 5), "lr_head": 1e-3, "sec": 10.0}
                     ).to_csv(tmp_path / "history" / f"{m}.csv", index=False)
    for m in mf.HISTORY_CLS:
        pd.DataFrame({"epoch": range(1, 6), "train_loss": 1.0, "val_score": rng.uniform(0.8, 0.95, 5), "lr_head": 1e-3, "sec": 10.0}
                     ).to_csv(tmp_path / "history" / f"{m}.csv", index=False)
    # Nutrition5k metadata biçimi: dish_id, kcal, kütle, yağ, karb, protein, sonra 7'li malzeme blokları
    with open(tmp_path / "n5k" / "metadata" / "dish_metadata_cafe1.csv", "w", newline="") as f:
        w = csv.writer(f)
        for i, (d, c) in enumerate(zip(dish_ids, y_cal)):
            ingr = ["ingr_1", "fish" if i % 5 == 0 else "rice", 100, c, 1, 1, 1]
            w.writerow([d, c, 150, 5, 20, 10] + ingr)
    _write_results(tmp_path / "results.csv", cls_models, reg_models)

    captions = mf.main(["--results-dir", str(tmp_path), "--n5k-dir", str(tmp_path / "n5k")])

    names = {"eval_karisiklik_matrisi", "eval_sinif_f1", "eval_tahmin_gercek", "eval_hata_histogrami", "eval_aralik_mae",
             "eval_model_karsilastirma", "eval_egitim_egrileri", "eval_en_kotu_tabaklar"}
    assert {n for n, _ in captions} == names
    for n in names:
        assert (tmp_path / "figures" / f"{n}.png").stat().st_size > 1000
    worst = pd.read_csv(tmp_path / "eval_en_kotu_20.csv")
    assert len(worst) == 20 and worst["abs_error"].is_monotonic_decreasing and worst["has_fish"].any()
    assert len(pd.read_csv(tmp_path / "eval_karisan_ciftler.csv")) == 10
    for name in ["tablo1.md", "tablo2.md", "tablo3_kalori_araligi.md"]:
        text = (tmp_path / name).read_text()
        assert text.startswith("| Tablo") and "," in text and "." not in text.split("\n")[2].split("|")[2]
