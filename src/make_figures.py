"""Adım 15 — Değerlendirme figürleri ve rapor tabloları (Tablo 1–3), tamamı results/ içindeki kayıtlardan.

    python -m src.make_figures                                 # varsayılanlar: ViT-B/16 (sınıf), ConvNeXt-T (kalori)
    python -m src.make_figures --cls-model convnext_tiny_cls --reg-model vit_b16_reg

GPU, veri seti veya checkpoint gerekmez: tahminler results/preds/, eğitim eğrileri results/history/, metrikler
results/results.csv'den okunur. Yalnızca "en kötü tabaklar" ızgarası data/nutrition5k/realsense_overhead altındaki
görüntüleri kullanır (yoksa gri kutu çizilir). Her figür için README'ye yapıştırılacak tek satırlık başlık ekrana yazılır.

Çıktılar (results/figures/eval_*.png, results/*.csv|md):
    eval_karisiklik_matrisi.png   25×25 karışıklık matrisi (satır normalize)
    eval_sinif_f1.png             sınıf başına F1
    eval_karisan_ciftler.csv      en çok karışan 10 sınıf çifti  → Adım 16
    eval_tahmin_gercek.png        tahmin–gerçek saçılımı (±100 kcal bandı, balıklı tabaklar ayrı)
    eval_hata_histogrami.png      işaretli hata (tahmin − gerçek) histogramı
    eval_aralik_mae.png           kalori aralığı başına MAE (düşük / orta / yüksek)
    eval_model_karsilastirma.png  test MAE + %95 güven aralığı, literatür referans çizgileri
    eval_egitim_egrileri.png      doğrulama skoru – epoch
    eval_en_kotu_tabaklar.png     en kötü 20 tabak ızgarası, eval_en_kotu_20.csv → Adım 16
    tablo1.md, tablo2.md, tablo3_kalori_araligi.md   rapora yapıştırılacak markdown tablolar

TODO (Adım 16): Grad-CAM / attention haritaları için kanca — en kötü 20 tabağın kimlikleri eval_en_kotu_20.csv'de;
checkpoint indirildikten sonra (Release `checkpoints`, ~111 MB) `src.models.build_model` ile yüklenip bu kimlikler
üzerinde çalıştırılacak. Bu betik checkpoint gerektirmez.
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from . import plot_style
from .config import CALORIE_BIN_EDGES, N5K_DIR, RESULTS_DIR
from .data import FOOD101_CLASSES, read_dish_ingredients
from .metrics import regression_metrics, to_bins
from .plot_style import AXIS, INK, SERIES, SURFACE, sequential_cmap

BIN_NAMES = ("düşük", "orta", "yüksek")
FISH_INGREDIENT = "fish"          # malzeme listesinde TAM eşleşme (bkz. EDA: balık kalorisi eksik kayıtlı)

# results.csv'deki model adı → raporda görünen ad
MODEL_NAMES = {
    "naive_mean": "Naif: train ortalaması",
    "naive_most_frequent": "Naif: en sık sınıf",
    "cnn_xgb": "Donmuş ResNet-50 → XGBoost",
    "cnn_logreg": "Donmuş ResNet-50 → lojistik regresyon",
    "resnet50_reg_timminit": "ResNet-50 (timm başlatması)",
    "resnet50_reg": "ResNet-50 (erken durdu, 14/30)",
    "resnet50_reg_mt": "ResNet-50 çok görevli",
    "efficientnet_b0_reg_timminit": "EfficientNet-B0 (hatalı başlatma)",
    "efficientnet_b0_reg": "EfficientNet-B0",
    "vit_b16_reg": "ViT-B/16",
    "convnext_tiny_reg": "ConvNeXt-T",
    "vlm_gemini_3_5_flash_lite": "VLM sıfır-atış (Gemini 3.5 Flash-Lite)",
    "resnet50_cls": "ResNet-50",
    "efficientnet_b0_cls": "EfficientNet-B0",
    "convnext_tiny_cls": "ConvNeXt-T",
    "vit_b16_cls": "ViT-B/16",
}
# Tablo 1 (sınıflandırma) ve Tablo 2 (kalori) satırları, README'deki sırayla
TABLO1_MODELS = ["naive_most_frequent", "cnn_logreg", "resnet50_cls", "efficientnet_b0_cls", "convnext_tiny_cls", "vit_b16_cls"]
TABLO2_MODELS = ["naive_mean", "cnn_xgb", "resnet50_reg_timminit", "resnet50_reg", "resnet50_reg_mt",
                 "efficientnet_b0_reg_timminit", "efficientnet_b0_reg", "vit_b16_reg", "convnext_tiny_reg",
                 "vlm_gemini_3_5_flash_lite"]
# Model karşılaştırma figürü ve aralık başına MAE: kötü/yinelenen run'lar hariç
COMPARE_MODELS = ["naive_mean", "vlm_gemini_3_5_flash_lite", "cnn_xgb", "resnet50_reg_timminit", "resnet50_reg_mt",
                  "efficientnet_b0_reg", "vit_b16_reg", "convnext_tiny_reg"]
HISTORY_REG = ["convnext_tiny_reg", "vit_b16_reg", "efficientnet_b0_reg", "resnet50_reg_mt", "resnet50_reg_timminit", "resnet50_reg"]
HISTORY_CLS = ["vit_b16_cls", "convnext_tiny_cls", "efficientnet_b0_cls", "resnet50_cls"]
# Literatür referansları (README §2 tablosu): kcal MAE
LITERATURE = [("Sıfır-atış VLM (Gemini 3.0 Flash)", 80.7), ("Nutrition5k RGB (JFT ön eğitimli)", 70.6)]
LITERATURE_BAND = (37.0, 40.0, "Segmentasyon / tahmini derinlik")


# ---------------------------------------------------------------- saf yardımcılar (testler bunları kullanır)

def class_label(name):
    """food101 sınıf adını okunur hale getirir: spaghetti_bolognese → spaghetti bolognese."""
    return name.replace("_", " ")


def model_name(model):
    return MODEL_NAMES.get(model, model)


def fig_name(model):
    """Figür metni: sistem yazı tipinde "→" karakteri yok, "+" kullanılır (tablolarda README gibi "→" kalır)."""
    return model_name(model).replace(" → ", " + ")


def confused_pairs(y_true, y_pred, classes, k=10):
    """En çok karışan k sınıf çifti (yönlü: gerçek → tahmin), köşegen hariç. Sütunlar: true, pred, count."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    off = y_true != y_pred
    pairs = pd.DataFrame({"true": y_true[off], "pred": y_pred[off]}).value_counts().reset_index(name="count")
    pairs = pairs.sort_values(["count", "true", "pred"], ascending=[False, True, True]).head(k).reset_index(drop=True)
    pairs["true"] = [class_label(classes[i]) for i in pairs["true"]]
    pairs["pred"] = [class_label(classes[i]) for i in pairs["pred"]]
    return pairs


def per_bin_mae(y_true, y_pred, edges=CALORIE_BIN_EDGES):
    """Gerçek kaloriye göre aralık (düşük / orta / yüksek) başına MAE ve tabak sayısı."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    b = to_bins(y, edges)
    rows = []
    for i, name in enumerate(BIN_NAMES):
        m = b == i
        rows.append({"bin": name, "n": int(m.sum()), "MAE": float(np.abs(p[m] - y[m]).mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def fmt_tr(x, decimals=1, percent=False):
    """Türkçe ondalık virgül: 41.44 → '41,4'; percent=True → '%41,4'. NaN → '—'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    s = f"{x:.{decimals}f}"
    s = s.replace("-0." + "0" * decimals, "0." + "0" * decimals).replace(".", ",")  # -0,00 → 0,00
    return f"%{s}" if percent else s


def md_table(headers, rows):
    """Markdown tablo metni (README stili: '| a | b |' + '| --- | --- |')."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def worst_k(preds, k=20):
    """Mutlak hatası en büyük k tabak (büyükten küçüğe). Sütunlar: id, y_true, y_pred, abs_error."""
    df = preds[["id", "y_true", "y_pred"]].copy()
    df["abs_error"] = (df["y_pred"] - df["y_true"]).abs()
    return df.sort_values(["abs_error", "id"], ascending=[False, True]).head(k).reset_index(drop=True)


def has_fish(ingredients, dish_id, name=FISH_INGREDIENT):
    return any(n == name for n, _, _ in ingredients.get(dish_id, []))


def top_ingredients(ingredients, dish_id, n=3):
    """Grama göre en büyük n malzeme: 'rice (120 g), chicken (80 g)'."""
    items = sorted(ingredients.get(dish_id, []), key=lambda t: -t[1])[:n]
    return ", ".join(f"{name} ({grams:.0f} g)" for name, grams, _ in items)


def results_row(results, task, model, split="test"):
    """results.csv'den tek satır (aynı model/split birden çok kez kaydedildiyse sonuncusu)."""
    r = results[(results.task == task) & (results.model == model) & (results.split == split)]
    if r.empty:
        raise KeyError(f"results.csv'de satır yok: task={task} model={model} split={split}")
    return r.iloc[-1]


def tablo1_md(results, models=TABLO1_MODELS):
    rows = []
    for m in models:
        r = results_row(results, "cls", m)
        ci = f" [{fmt_tr(r.f1_ci_low, 3)}–{fmt_tr(r.f1_ci_high, 3)}]" if pd.notna(r.get("f1_ci_low")) else ""
        rows.append([model_name(m), fmt_tr(100 * r.accuracy, 1, percent=True), fmt_tr(r.precision, 3), fmt_tr(r.recall, 3),
                     fmt_tr(r.f1, 3) + ci, fmt_tr(100 * r.top5, 1, percent=True) if pd.notna(r.get("top5")) else "—"])
    return md_table(["Tablo 1 · Food-101 (25 sınıf, test)", "Doğruluk", "Precision", "Recall", "Macro-F1 [%95 GA]", "Top-5"], rows)


def tablo2_md(results, models=TABLO2_MODELS):
    rows = []
    for m in models:
        r = results_row(results, "calorie", m)
        rows.append([model_name(m), f"{fmt_tr(r.MAE)} [{fmt_tr(r.MAE_ci_low)}–{fmt_tr(r.MAE_ci_high)}]",
                     fmt_tr(r.PMAE, 1, percent=True), fmt_tr(r.RMSE), fmt_tr(r.R2, 2), fmt_tr(r.within_100kcal, 1, percent=True)])
    return md_table(["Tablo 2 · Nutrition5k (yalnızca RGB, test)", "MAE (kcal) [%95 GA]", "PMAE", "RMSE", "R²", "±100 kcal içinde"], rows)


def tablo3_md(results, models=TABLO2_MODELS):
    rows = []
    for m in models:
        r = results_row(results, "calorie_bins", m)
        rows.append([model_name(m), fmt_tr(100 * r.accuracy, 1, percent=True), fmt_tr(r.precision, 3), fmt_tr(r.recall, 3),
                     f"{fmt_tr(r.f1, 3)} [{fmt_tr(r.f1_ci_low, 3)}–{fmt_tr(r.f1_ci_high, 3)}]"])
    return md_table(["Tablo 3 · Kalori aralığı (düşük < 116,1 ≤ orta < 313,7 ≤ yüksek kcal, test)",
                     "Doğruluk", "Precision (macro)", "Recall (macro)", "Macro-F1 [%95 GA]"], rows)


# ---------------------------------------------------------------- veri okuma

def load_preds(preds_dir, model, split="test"):
    name = model if split == "test" else f"{model}_{split}"
    return pd.read_csv(Path(preds_dir) / f"{name}.csv")


def load_history(history_dir, model):
    return pd.read_csv(Path(history_dir) / f"{model}.csv")


# ---------------------------------------------------------------- figürler

def fig_confusion(y_true, y_pred, classes, title, annotate_min=0.02):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
    labels = [class_label(c) for c in classes]
    fig, ax = plt.subplots(figsize=(9.5, 8.6))
    ax.grid(False)
    im = ax.imshow(cm, cmap=sequential_cmap(), vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), labels, fontsize=8)
    ax.set(xlabel="tahmin edilen sınıf", ylabel="gerçek sınıf", title=title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm[i, j]
            if i == j:
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.6 else INK["primary"])
            elif v >= annotate_min:
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.5, color=INK["primary"])
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.ax.set_yticklabels(["%0", "%25", "%50", "%75", "%100"])
    cb.set_label("satır payı (gerçek sınıf içindeki oran)", color=INK["secondary"])
    cb.outline.set_edgecolor(AXIS)
    fig.tight_layout()
    return fig


def fig_class_f1(y_true, y_pred, classes, name_main, second=None, name_second=None):
    f1 = f1_score(y_true, y_pred, average=None, labels=range(len(classes)), zero_division=0)
    order = np.argsort(f1)
    labels = [class_label(classes[i]) for i in order]
    fig, ax = plt.subplots(figsize=(7, 7))
    y = np.arange(len(classes))
    ax.barh(y, f1[order], color=SERIES[0], height=0.62, label=name_main, zorder=3)
    if second is not None:
        f1b = f1_score(second[0], second[1], average=None, labels=range(len(classes)), zero_division=0)
        ax.scatter(f1b[order], y, color=SERIES[1], s=26, zorder=4, label=name_second)
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=8.5)
    ax.set_yticks(y, labels, fontsize=8.5)
    lo = min(0.6, float(f1.min()) - 0.05)
    for yi, v in zip(y, f1[order]):
        ax.text(lo + 0.005, yi, f"{v:.2f}", ha="left", va="center", fontsize=7, color="white")
    ax.set(xlim=(lo, 1.0), xlabel="F1 (test)", title=f"Sınıf başına F1 — {name_main}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_pred_vs_true(preds, fish_mask, name, band=100):
    """Doğrusal eksen: ±100 kcal bandı sabit genişlikte kalır, 0–50 kcal'lik tabaklar log-log'da aşırı yer kaplardı."""
    y, p = preds["y_true"].to_numpy(float), preds["y_pred"].to_numpy(float)
    m = regression_metrics(y, p)
    hi = float(max(y.max(), p.max())) * 1.04
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.fill_between([0, hi], [-band, hi - band], [band, hi + band], color=SERIES[0], alpha=0.08, lw=0, zorder=1,
                    label=f"±{band} kcal")
    ax.plot([0, hi], [0, hi], color=INK["secondary"], lw=1, ls="--", zorder=2, label="y = x")
    ax.scatter(y[~fish_mask], p[~fish_mask], s=14, color=SERIES[0], alpha=0.65, lw=0, zorder=3,
               label=f"diğer tabaklar ({(~fish_mask).sum()})")
    ax.scatter(y[fish_mask], p[fish_mask], s=22, color=SERIES[1], alpha=0.9, lw=0, zorder=4,
               label=f"balıklı tabaklar ({fish_mask.sum()})")
    ax.set(xlim=(0, hi), ylim=(0, hi), xlabel="gerçek kalori (kcal)", ylabel="tahmin (kcal)",
           title=f"Tahmin – gerçek (test) — {name}")
    ax.set_aspect("equal")
    ax.text(0.03, 0.97, f"MAE {m['MAE']:.1f} kcal\nPMAE %{m['PMAE']:.1f}\nR² {m['R2']:.2f}\n±{band} kcal içinde %{m['within_100kcal']:.0f}",
            transform=ax.transAxes, va="top", fontsize=9, color=INK["secondary"])
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    return fig


def fig_error_hist(preds, name, second=None, name_second=None, clip=400):
    err = (preds["y_pred"] - preds["y_true"]).to_numpy(float)
    bins = np.linspace(-clip, clip, 41)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    if second is not None:
        err2 = (second["y_pred"] - second["y_true"]).to_numpy(float)
        ax.hist(np.clip(err2, -clip, clip), bins=bins, color=SERIES[1], alpha=0.35, edgecolor=SURFACE, linewidth=0.6,
                label=name_second, zorder=2)
    counts, _, _ = ax.hist(np.clip(err, -clip, clip), bins=bins, color=SERIES[0], alpha=0.85, edgecolor=SURFACE, linewidth=0.6,
                           label=name, zorder=3)
    top = counts.max() * 1.25
    for v, lab, ha, h in [(np.median(err), "medyan", "right", 0.92), (err.mean(), "ortalama", "left", 0.82)]:
        ax.axvline(v, color=INK["secondary"], lw=1, ls="--", zorder=4)
        ax.text(v + (6 if ha == "left" else -6), top * h, f"{lab} {v:+.0f}".replace("-", "−"), ha=ha, fontsize=8,
                color=INK["secondary"])
    ax.set(ylim=(0, top), xlabel=f"tahmin − gerçek (kcal, ±{clip}'e kırpıldı)", ylabel="tabak sayısı",
           title="İşaretli hata dağılımı (test)")
    n_out = int((np.abs(err) > clip).sum())
    ax.text(0.98, 0.95, f"|hata| > {clip} kcal: {n_out} tabak", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=INK["secondary"])
    if second is not None:
        ax.legend(loc="upper right", bbox_to_anchor=(1, 0.88))
    fig.tight_layout()
    return fig


def fig_bin_mae(bin_table):
    """bin_table: sütunlar model (görünen ad), bin, MAE. Modeller x ekseninde, aralıklar renk."""
    models = list(dict.fromkeys(bin_table["model"]))
    x = np.arange(len(models))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(models)), 4.2))
    for i, b in enumerate(BIN_NAMES):
        vals = [float(bin_table[(bin_table.model == m) & (bin_table["bin"] == b)]["MAE"].iloc[0]) for m in models]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=SERIES[i], label=b, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=7,
                    color=INK["secondary"])
    ax.set_xticks(x, [m.replace(": ", ":\n").replace(" + ", "\n+ ").replace(" (", "\n(") for m in models], fontsize=8)
    ax.set(ylabel="MAE (kcal)", title="Kalori aralığı başına MAE (test)")
    ax.grid(axis="x", visible=False)
    ax.legend(title="gerçek kalori aralığı", loc="upper right", fontsize=8.5, title_fontsize=8.5)
    fig.tight_layout()
    return fig


def fig_model_comparison(table):
    """table: sütunlar model (görünen ad), MAE, lo, hi — en iyiden kötüye değil, verilen sırada (yukarıdan aşağı)."""
    n = len(table)
    y = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * n + 2.2))
    lo, hi, lab = LITERATURE_BAND
    ax.axvspan(lo, hi, color=SERIES[2], alpha=0.18, lw=0, zorder=1, label=f"{lab} ({lo:.0f}–{hi:.0f})")
    for (name, v), ls in zip(LITERATURE, [":", "--"]):
        ax.axvline(v, color=INK["muted"], lw=1, ls=ls, zorder=2, label=f"{name}: {v}")
    err = np.vstack([table["MAE"] - table["lo"], table["hi"] - table["MAE"]])
    ax.errorbar(table["MAE"], y, xerr=err, fmt="o", color=SERIES[0], ecolor=SERIES[0], elinewidth=1.6, capsize=3,
                ms=6, zorder=4)
    for yi, (v, l, h) in zip(y, zip(table["MAE"], table["lo"], table["hi"])):
        ax.text(h + 2, yi, f"{v:.1f} [{l:.1f}–{h:.1f}]", va="center", fontsize=8, color=INK["secondary"])
    ax.set_yticks(y, table["model"], fontsize=9)
    ax.set(xlim=(0, float(table["hi"].max()) * 1.3), ylim=(-0.7, n - 0.3),
           xlabel="test MAE (kcal), %95 bootstrap güven aralığı", title="Kalori modelleri – literatür referansları")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8, title="literatür (yalnızca RGB değil)", title_fontsize=8)
    fig.tight_layout()
    return fig


def fig_training_curves(hist_reg, hist_cls):
    """hist_*: {görünen ad: DataFrame(epoch, val_score)}. Kalori: val MAE (düşük iyi); sınıf: val macro-F1 (yüksek iyi)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, hists, better, ylabel, title in [(a, hist_reg, np.argmin, "doğrulama MAE (kcal)", "Kalori modelleri"),
                                             (b, hist_cls, np.argmax, "doğrulama macro-F1", "Sınıflandırma modelleri")]:
        for i, (name, h) in enumerate(hists.items()):
            ax.plot(h["epoch"], h["val_score"], color=SERIES[i], lw=1.6, label=name, zorder=3)
            k = better(h["val_score"].to_numpy())
            ax.scatter([h["epoch"].iloc[k]], [h["val_score"].iloc[k]], color=SERIES[i], s=36, zorder=4,
                       edgecolor=SURFACE, linewidth=1)
        ax.set(xlabel="epoch", ylabel=ylabel, title=title)
        ax.legend(fontsize=8)
    a.text(0.02, 0.03, "nokta = en iyi epoch (checkpoint)", transform=a.transAxes, ha="left", va="bottom", fontsize=8,
           color=INK["secondary"])
    fig.tight_layout()
    return fig


def fig_worst_dishes(worst, img_dir, ncols=5, w=2.6):
    """worst: id, y_true, y_pred, abs_error, has_fish. Görüntü yoksa gri kutu."""
    ncols = max(1, min(ncols, len(worst)))
    nrows = max(1, math.ceil(len(worst) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, w * 0.86 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, worst.itertuples()):
        path = Path(img_dir) / r.id / "rgb.png"
        if path.exists():
            ax.imshow(Image.open(path))
        else:
            ax.imshow(np.full((10, 13, 3), 0.85))
        tag = " · balıklı" if r.has_fish else ""
        ax.set_title(f"gerçek {r.y_true:.0f} · tahmin {r.y_pred:.0f} kcal{tag}", fontsize=8, fontweight="normal",
                     color=SERIES[1] if r.has_fish else INK["secondary"])
        ax.text(0.02, 0.03, r.id.replace("dish_", ""), transform=ax.transAxes, fontsize=6, color=SURFACE)
    fig.suptitle(f"En kötü {len(worst)} test tabağı (mutlak hata, büyükten küçüğe)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- ana akış

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Adım 15 değerlendirme figürleri ve Tablo 1–3 (results/ kayıtlarından).")
    ap.add_argument("--cls-model", default="vit_b16_cls", help="karışıklık matrisi ve sınıf F1 için model")
    ap.add_argument("--cls-second", default="convnext_tiny_cls", help="sınıf F1 üstüne bindirilecek ikinci model ('' = yok)")
    ap.add_argument("--reg-model", default="convnext_tiny_reg", help="saçılım, histogram ve en kötü tabaklar için model")
    ap.add_argument("--reg-second", default="vit_b16_reg", help="histograma bindirilecek ikinci model ('' = yok)")
    ap.add_argument("--worst-k", type=int, default=20)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="results.csv, preds/, history/ burada; md/csv çıktıları buraya")
    ap.add_argument("--figures-dir", type=Path, default=None, help="varsayılan: <results-dir>/figures")
    ap.add_argument("--n5k-dir", type=Path, default=N5K_DIR, help="metadata/ ve realsense_overhead/ (yalnızca en kötü tabaklar)")
    ap.add_argument("--compare-models", nargs="*", default=COMPARE_MODELS)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plot_style.apply()
    results_dir, preds_dir, history_dir = args.results_dir, args.results_dir / "preds", args.results_dir / "history"
    figures_dir = args.figures_dir or results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    captions = []

    def save(fig, name, dpi=None):
        if args.figures_dir is None and results_dir == RESULTS_DIR:
            path = plot_style.savefig(fig, name, dpi=dpi)
        else:
            path = figures_dir / f"{name}.png"
            fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    def caption(name, text):
        captions.append((name, text))
        print(f"[{name}.png] {text}")

    results = pd.read_csv(results_dir / "results.csv")
    classes = FOOD101_CLASSES
    cls_name, reg_name = fig_name(args.cls_model), fig_name(args.reg_model)

    # 1–3. sınıflandırma: karışıklık matrisi, sınıf F1, karışan çiftler
    cls = load_preds(preds_dir, args.cls_model)
    assert cls.y_true.between(0, len(classes) - 1).all(), "sınıf kimlikleri FOOD101_CLASSES dizinine uymuyor"
    save(fig_confusion(cls.y_true, cls.y_pred, classes, f"Karışıklık matrisi (test, satır normalize) — {cls_name}"),
         "eval_karisiklik_matrisi")
    caption("eval_karisiklik_matrisi", f"{cls_name} karışıklık matrisi (Food-101, 25 sınıf, test; satırlar gerçek sınıf, "
            "hücreler satır içindeki yüzde; yalnızca ≥ %2'lik karışmalar, yani ≥ 5 görüntü, yazılı).")
    second = None
    if args.cls_second:
        s = load_preds(preds_dir, args.cls_second)
        second = (s.y_true, s.y_pred)
    save(fig_class_f1(cls.y_true, cls.y_pred, classes, cls_name, second, fig_name(args.cls_second) if second else None),
         "eval_sinif_f1")
    caption("eval_sinif_f1", f"Sınıf başına F1 (test), {cls_name} çubuk" +
            (f", {model_name(args.cls_second)} nokta" if second else "") + "; en zor sınıflar en altta.")
    pairs = confused_pairs(cls.y_true, cls.y_pred, classes, k=10)
    pairs.to_csv(results_dir / "eval_karisan_ciftler.csv", index=False)
    print(f"\nEn çok karışan 10 çift ({cls_name}, test) → results/eval_karisan_ciftler.csv")
    print(pairs.to_string(index=False), "\n")

    # 4–5. kalori: saçılım ve hata histogramı
    reg = load_preds(preds_dir, args.reg_model)
    ingr = read_dish_ingredients(args.n5k_dir / "metadata")
    fish = reg["id"].map(lambda d: has_fish(ingr, d)).to_numpy(bool)
    save(fig_pred_vs_true(reg, fish, reg_name), "eval_tahmin_gercek")
    caption("eval_tahmin_gercek", f"{reg_name} test tahminleri gerçek kaloriye karşı (doğrusal eksen; gölgeli bant ±100 kcal; "
            f"turuncu = balıklı {int(fish.sum())} tabak).")
    reg2 = load_preds(preds_dir, args.reg_second) if args.reg_second else None
    save(fig_error_hist(reg, reg_name, reg2, fig_name(args.reg_second) if reg2 is not None else None), "eval_hata_histogrami")
    caption("eval_hata_histogrami", f"İşaretli hata (tahmin − gerçek) dağılımı, {reg_name}" +
            (f" ve {model_name(args.reg_second)} (açık renk)" if reg2 is not None else "") +
            "; kesikli çizgiler ortalama ve medyan.")

    # 6. aralık başına MAE
    rows = []
    for m in args.compare_models:
        t = per_bin_mae(load_preds(preds_dir, m)["y_true"], load_preds(preds_dir, m)["y_pred"])
        t.insert(0, "model", fig_name(m))
        t.insert(0, "model_id", m)
        rows.append(t)
    bin_table = pd.concat(rows, ignore_index=True)
    bin_table.to_csv(results_dir / "eval_aralik_mae.csv", index=False)
    save(fig_bin_mae(bin_table), "eval_aralik_mae")
    caption("eval_aralik_mae", "Gerçek kalori aralığına (düşük < 116,1 ≤ orta < 313,7 ≤ yüksek kcal) göre test MAE; "
            "hata kaloriyle birlikte büyür, yüksek aralıkta modeller arasındaki fark en belirgin.")

    # 7. model karşılaştırma (CI results.csv'den)
    comp = pd.DataFrame([{"model": fig_name(m), "MAE": results_row(results, "calorie", m).MAE,
                          "lo": results_row(results, "calorie", m).MAE_ci_low, "hi": results_row(results, "calorie", m).MAE_ci_high}
                         for m in args.compare_models])
    save(fig_model_comparison(comp), "eval_model_karsilastirma")
    caption("eval_model_karsilastirma", "Kalori modellerinin test MAE'si ve %95 bootstrap güven aralığı; noktalı çizgiler "
            "literatür referansları (VLM 80,7; Nutrition5k RGB 70,6), yeşil bant segmentasyon/derinlik çalışmaları (37–40 kcal).")

    # 8. eğitim eğrileri
    hist_reg = {fig_name(m): load_history(history_dir, m) for m in HISTORY_REG if (history_dir / f"{m}.csv").exists()}
    hist_cls = {fig_name(m): load_history(history_dir, m) for m in HISTORY_CLS if (history_dir / f"{m}.csv").exists()}
    save(fig_training_curves(hist_reg, hist_cls), "eval_egitim_egrileri")
    caption("eval_egitim_egrileri", "Doğrulama skoru – epoch: kalori modelleri (val MAE, sol) ve sınıflandırma modelleri "
            "(val macro-F1, sağ); nokta = checkpoint olarak saklanan en iyi epoch.")

    # 9. en kötü tabaklar
    worst = worst_k(reg, args.worst_k)
    worst["has_fish"] = worst["id"].map(lambda d: has_fish(ingr, d))
    worst["top_ingredients"] = worst["id"].map(lambda d: top_ingredients(ingr, d))
    worst.round(1).to_csv(results_dir / "eval_en_kotu_20.csv", index=False)
    save(fig_worst_dishes(worst, args.n5k_dir / "realsense_overhead"), "eval_en_kotu_tabaklar", dpi=110)
    caption("eval_en_kotu_tabaklar", f"{reg_name} için mutlak hatası en büyük {len(worst)} test tabağı "
            f"(başlıkta gerçek → tahmin; turuncu başlık = balıklı tabak, {int(worst.has_fish.sum())} tane).")
    print(f"\nEn kötü {len(worst)} tabak → results/eval_en_kotu_20.csv  (balıklı: {int(worst.has_fish.sum())})")
    print(worst[["id", "y_true", "y_pred", "abs_error", "has_fish", "top_ingredients"]].round(0).to_string(index=False), "\n")

    # 10. tablolar
    for name, text in [("tablo1.md", tablo1_md(results)), ("tablo2.md", tablo2_md(results)),
                       ("tablo3_kalori_araligi.md", tablo3_md(results))]:
        (results_dir / name).write_text(text, encoding="utf-8")
        print(f"→ results/{name}\n{text}")
    return captions


if __name__ == "__main__":
    main()
