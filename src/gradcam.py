"""Grad-CAM: kalori modelinin tabağın neresine baktığı (Adım 15 figürü).

Kullanım:
    python -m src.gradcam --ckpt checkpoints/convnext_tiny_reg.pt            # → results/figures/eval_gradcam.png
    python -m src.gradcam --ckpt checkpoints/convnext_tiny_reg.pt --n 4 --device cpu

Regresyonda "sınıf skoru" yok; tahmin edilen kalorinin son evre özellik haritasına göre gradyanı alınır.
Isı haritası kırmızıysa o bölge tahmini yukarı çekiyor. Hangi tabaklar: tahmin dosyasındaki en iyi n
ve en kötü n tabak (mutlak hata), böylece doğru ve yanlış tahminlerin nereye baktığı yan yana görülür.
Kütüphane kullanılmadı (pytorch-grad-cam kurulu değil); 30 satırlık kanca yeterli.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import plot_style
from .config import N5K_DIR, RESULTS_DIR
from .data import read_dish_ingredients
from .models import decode_target, load_checkpoint
from .transforms import calorie_transform


def _target_layer(model):
    """Son evre özellik haritası: ConvNeXt stages[-1] (7×7×768), ResNet layer4, EfficientNet conv_head."""
    if hasattr(model, "stages"):
        return model.stages[-1]
    if hasattr(model, "layer4"):
        return model.layer4
    if hasattr(model, "conv_head"):
        return model.conv_head
    raise ValueError("Grad-CAM için uygun katman bulunamadı (ViT için uygun değil)")


def gradcam(model, x, layer):
    """x: (1,3,H,W) → (h,w) ısı haritası [0,1] ve ölçeklenmiş tahmin."""
    store = {}

    def keep(_m, _i, out):  # kancalar None döndürmeli; değer döndürmek çıktıyı değiştirir
        store["f"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = layer.register_forward_hook(keep)
    try:
        model.zero_grad()
        out = model(x)[0, 0]  # ilk çıktı = kalori
        out.backward()
    finally:
        handle.remove()
    f, g = store["f"][0], store["g"][0]                      # (C,h,w)
    cam = torch.relu((g.mean(dim=(1, 2), keepdim=True) * f).sum(0))
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), out.item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preds", default=str(RESULTS_DIR / "preds" / "convnext_tiny_reg.csv"))
    ap.add_argument("--n", type=int, default=4, help="en iyi n + en kötü n tabak")
    ap.add_argument("--device", default="cpu", help="cpu güvenli; mps'te backward kancaları bazen boş döner")
    ap.add_argument("--out", default="eval_gradcam")
    args = ap.parse_args()

    model, meta = load_checkpoint(args.ckpt, args.device)
    layer = _target_layer(model)
    tf = calorie_transform(train=False, mean=meta["mean"], std=meta["std"])
    ingr = read_dish_ingredients()

    df = pd.read_csv(args.preds)
    df["ae"] = (df.y_true - df.y_pred).abs()
    df = df.sort_values("ae")
    picks = pd.concat([df.head(args.n).assign(grup="en iyi"), df.tail(args.n).assign(grup="en kötü")])

    plot_style.apply()
    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.2))
    for j, (_, r) in enumerate(picks.iterrows()):
        img = Image.open(N5K_DIR / "realsense_overhead" / r.id / "rgb.png").convert("RGB")
        x = tf(img).unsqueeze(0).to(args.device)
        cam, z = gradcam(model, x, layer)
        pred = float(decode_target(torch.tensor([z]), meta["target"], meta["scale"])[0])
        shown = np.asarray(tf.transforms[0](img))  # Compose'un ilk adımı Resize: modelin gördüğü görüntü
        cam_up = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(shown.shape[1::-1], Image.BILINEAR)) / 255.0
        fish = any(nm == "fish" for nm, _, _ in ingr.get(r.id, []))
        axes[0, j].imshow(shown)
        axes[0, j].set_title(f"{r.grup}{' · balık' if fish else ''}", fontsize=9)
        axes[1, j].imshow(shown)
        axes[1, j].imshow(cam_up, cmap="jet", alpha=0.45)
        axes[1, j].set_xlabel(f"gerçek {r.y_true:.0f} · tahmin {pred:.0f} kcal", fontsize=9)
        for a in axes[:, j]:
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            for sp in a.spines.values():
                sp.set_visible(False)
    fig.suptitle("Grad-CAM, ConvNeXt-T kalori modeli: kırmızı bölgeler tahmini yukarı çekiyor (test tabakları)", x=0.01, ha="left")
    fig.tight_layout()
    path = plot_style.savefig(fig, args.out, dpi=130)
    print(f"kaydedildi: {path}")
    print("Seçilen tabaklar:")
    print(picks[["id", "grup", "y_true", "y_pred", "ae"]].to_string(index=False))


if __name__ == "__main__":
    main()
