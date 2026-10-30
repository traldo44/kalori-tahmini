"""Kardeş-sızıntısı ölçümü (Adım 16): resmi test kümesindeki tabakların bir kısmı, train'deki bir tabağın
malzeme eklenmiş/çıkarılmış hali (EDA: ~%22). Bu tabaklarda hata daha düşükse, test MAE'si iyimserdir.

    python -m src.leakage                       # → results/eval_kardes_sizintisi.csv + figures/eval_kardes_sizintisi.png

Tanım (EDA ile aynı): test tabağının zaman olarak en yakın train komşusu `gap` saniye (600) içindeyse VE
malzeme kümeleri iç içeyse (biri diğerinin alt kümesi) "kardeşli". Yalnızca zaman komşuluğu yetmez: test
tabaklarının %82'sinin 10 dk içinde train komşusu var (kafeterya aralıksız çekim yapmış), iç içe olan %22.
Eğitim gerektirmez; tahmin dosyaları yeter.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plot_style
from .config import N5K_DIR, RESULTS_DIR, VAL_GROUP_GAP_SEC
from .data import dish_time, load_n5k_split, read_dish_ingredients
from .metrics import bootstrap_ci, mae

MODELS = ["naive_mean", "vlm_gemini_3_5_flash_lite", "cnn_xgb", "resnet50_reg_timminit", "resnet50_reg_mt",
          "efficientnet_b0_reg", "vit_b16_reg", "convnext_tiny_reg"]
LABELS = {"naive_mean": "Naif", "vlm_gemini_3_5_flash_lite": "VLM (Flash-Lite)", "cnn_xgb": "ResNet-50 + XGBoost",
          "resnet50_reg_timminit": "ResNet-50", "resnet50_reg_mt": "ResNet-50 çok görevli",
          "efficientnet_b0_reg": "EfficientNet-B0", "vit_b16_reg": "ViT-B/16", "convnext_tiny_reg": "ConvNeXt-T"}


def has_train_sibling(test_ids, train_ids, ingredients, gap=VAL_GROUP_GAP_SEC):
    """{test_id: True/False}: zaman olarak komşu (±1, `gap` sn içinde) bir train tabağı var ve malzemeleri iç içe mi?
    ingredients: {dish_id: [(malzeme, g, kcal), ...]} (read_dish_ingredients)."""
    names = {d: frozenset(n for n, _, _ in v) for d, v in ingredients.items()}
    by_time = {dish_time(d): d for d in train_ids}
    t_train = np.sort(list(by_time))
    out = {}
    for d in test_ids:
        t = dish_time(d)
        i = np.searchsorted(t_train, t)
        neigh = [by_time[t_train[j]] for j in (i - 1, i) if 0 <= j < len(t_train) and abs(t_train[j] - t) <= gap]
        out[d] = any(names[d] <= names[n] or names[n] <= names[d] for n in neigh if d in names and n in names)
    return out


def group_mae(preds, sibling):
    """Bir modelin tahmin dosyası için kardeşli / kardeşsiz MAE ve güven aralıkları."""
    df = preds.copy()
    df["sibling"] = df.id.map(sibling)
    rows = {}
    for key, name in ((True, "kardesli"), (False, "kardessiz")):
        g = df[df.sibling == key]
        lo, hi = bootstrap_ci(g.y_true.values, g.y_pred.values, mae)
        rows[name] = (mae(g.y_true.values, g.y_pred.values), lo, hi, len(g))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gap", type=int, default=VAL_GROUP_GAP_SEC)
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    train = load_n5k_split("train", N5K_DIR, verbose=False)
    test = load_n5k_split("test", N5K_DIR, verbose=False)
    sib = has_train_sibling(test, train, read_dish_ingredients(), args.gap)
    n_sib = sum(sib.values())
    print(f"test {len(test)} tabak; train'de {args.gap} sn içinde çekilmiş ve malzemeleri iç içe kardeşi olan: "
          f"{n_sib} (%{100 * n_sib / len(test):.1f})")

    out = []
    for m in args.models:
        path = RESULTS_DIR / "preds" / f"{m}.csv"
        if not path.exists():
            print(f"  atlandı (tahmin yok): {m}")
            continue
        r = group_mae(pd.read_csv(path), sib)
        out.append({"model": m, "label": LABELS.get(m, m),
                    "MAE_kardesli": r["kardesli"][0], "ci_low_kardesli": r["kardesli"][1], "ci_high_kardesli": r["kardesli"][2], "n_kardesli": r["kardesli"][3],
                    "MAE_kardessiz": r["kardessiz"][0], "ci_low_kardessiz": r["kardessiz"][1], "ci_high_kardessiz": r["kardessiz"][2], "n_kardessiz": r["kardessiz"][3]})
        out[-1]["oran"] = out[-1]["MAE_kardesli"] / out[-1]["MAE_kardessiz"]
        print(f"  {LABELS.get(m, m):24s} kardeşli {r['kardesli'][0]:6.1f} [{r['kardesli'][1]:.1f}–{r['kardesli'][2]:.1f}]  "
              f"kardeşsiz {r['kardessiz'][0]:6.1f} [{r['kardessiz'][1]:.1f}–{r['kardessiz'][2]:.1f}]  oran {out[-1]['oran']:.2f}")
    df = pd.DataFrame(out)
    csv_path = RESULTS_DIR / "eval_kardes_sizintisi.csv"
    df.round(2).to_csv(csv_path, index=False)

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(df))
    for k, (col, lab, color) in enumerate((("kardessiz", f"train'de kardeşi yok (n={df.n_kardessiz.iloc[0]})", plot_style.SERIES[0]),
                                           ("kardesli", f"train'de kardeşi var (n={df.n_kardesli.iloc[0]})", plot_style.SERIES[1]))):
        v = df[f"MAE_{col}"]; err = [v - df[f"ci_low_{col}"], df[f"ci_high_{col}"] - v]
        ax.barh(y + (k - 0.5) * 0.38, v, height=0.36, xerr=err, color=color, label=lab, error_kw={"elinewidth": 1, "capsize": 2})
    ax.set_yticks(y); ax.set_yticklabels(df.label); ax.invert_yaxis()
    ax.set_xlabel("test MAE (kcal), %95 bootstrap güven aralığı")
    ax.set_title("Kardeş-sızıntısı: train'de iç içe malzemeli komşusu olan test tabakları daha kolay mı?")
    ax.legend(loc="lower right")
    path = plot_style.savefig(fig, "eval_kardes_sizintisi")
    print(f"kaydedildi: {csv_path}, {path}")


if __name__ == "__main__":
    main()
