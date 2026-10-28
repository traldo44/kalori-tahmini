"""İki modelin test tahminlerini aynı tabaklar üzerinde karşılaştırır (eşleştirilmiş bootstrap).

    python -m src.compare results/preds/resnet50_reg.csv results/preds/vit_b16_reg.csv
"""
import argparse

import pandas as pd

from .metrics import mae, paired_bootstrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds_a")
    ap.add_argument("preds_b")
    args = ap.parse_args()
    a, b = pd.read_csv(args.preds_a), pd.read_csv(args.preds_b)
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        print(f"Uyarı: dosyalar farklı tabaklar içeriyor; yalnızca ortak {len(m)} tabak karşılaştırıldı.")
    r = paired_bootstrap(m["y_true_a"], m["y_pred_a"], m["y_pred_b"])
    print(f"A MAE = {mae(m['y_true_a'], m['y_pred_a']):.1f}   B MAE = {mae(m['y_true_a'], m['y_pred_b']):.1f}")
    print(f"Fark (A − B) = {r['diff']:+.1f} kcal, %95 güven aralığı [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}]")
    if r["significant"]:
        print("→ Fark anlamlı:", "A daha iyi." if r["diff"] < 0 else "B daha iyi.")
    else:
        print("→ Aralık 0'ı içeriyor: bu fark şans eseri olabilir, 'biri daha iyi' demeyin.")


if __name__ == "__main__":
    main()
