"""İki modelin test tahminlerini aynı örnekler üzerinde karşılaştırır (eşleştirilmiş bootstrap).

    python -m src.compare results/preds/resnet50_reg.csv results/preds/vit_b16_reg.csv   # kalori: MAE
    python -m src.compare results/preds/resnet50_cls.csv results/preds/vit_b16_cls.csv   # sınıf: macro-F1

Görev dosyadan anlaşılır: y_true ve y_pred tam sayıysa sınıflandırma (macro-F1), değilse kalori (MAE).
"""
import argparse

import pandas as pd
from pandas.api.types import is_integer_dtype

from .metrics import macro_f1, mae, paired_bootstrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds_a")
    ap.add_argument("preds_b")
    args = ap.parse_args()
    a, b = pd.read_csv(args.preds_a), pd.read_csv(args.preds_b)
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        print(f"Uyarı: dosyalar farklı örnekler içeriyor; yalnızca ortak {len(m)} örnek karşılaştırıldı.")
    cls = all(is_integer_dtype(df[c]) for df in (a, b) for c in ("y_true", "y_pred"))
    metric, name, unit = (macro_f1, "macro-F1", "") if cls else (mae, "MAE", " kcal")
    y = m["y_true_a"]
    r = paired_bootstrap(y, m["y_pred_a"], m["y_pred_b"], metric_fn=metric)
    fmt = ".3f" if cls else ".1f"
    print(f"A {name} = {metric(y, m['y_pred_a']):{fmt}}   B {name} = {metric(y, m['y_pred_b']):{fmt}}")
    print(f"Fark (A − B) = {r['diff']:+{fmt}}{unit}, %95 güven aralığı [{r['ci_low']:+{fmt}}, {r['ci_high']:+{fmt}}]")
    if r["significant"]:
        a_better = r["diff"] > 0 if cls else r["diff"] < 0   # F1 yüksek iyi, MAE düşük iyi
        print("→ Fark anlamlı:", "A daha iyi." if a_better else "B daha iyi.")
    else:
        print("→ Aralık 0'ı içeriyor: bu fark şans eseri olabilir, 'biri daha iyi' demeyin.")


if __name__ == "__main__":
    main()
