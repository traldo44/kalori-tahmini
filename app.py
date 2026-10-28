"""Demo (Adım 19): fotoğraf yükle → tahmini kcal.

    python app.py --ckpt checkpoints/best.pt
"""
import argparse

import gradio as gr

from src.config import get_device
from src.models import load_checkpoint, predict_image
from src.transforms import crop_to_aspect

GUIDE = """
**Nasıl çekmeli?** Model tek bir kafeteryada, sabit yükseklikteki üst kamerayla çekilmiş tabaklarla eğitildi.
En iyi sonuç için:
- Tabağı **tam üstten** (kuş bakışı) çekin, yandan değil
- Tabak fotoğrafın genişliğinin yaklaşık **yarısı ile üçte ikisi** arasını kaplasın
- Tek tabak, düz zemin, iyi ışık

**Sınırlılık:** Bu koşullara uymayan fotoğraflarda (yandan, çok yakından, birden fazla tabak) tahmin çok sapabilir.
Sağda modelin gerçekte gördüğü kırpılmış görüntü gösterilir.
"""

UNITS = {"calories": "kcal", "mass": "g", "fat": "g yağ", "carb": "g karbonhidrat", "protein": "g protein"}


def make_demo(model, meta, device):
    typical = meta["info"].get("test_MAE")

    def run(img):
        if img is None:
            return None, ""
        pred = predict_image(model, meta, img, device)
        text = f"## ≈ {pred['calories']:.0f} kcal"
        if typical:
            text += (f"\n\nBu model test kümesinde ortalama **±{typical:.0f} kcal** hata yaptı "
                     "(üstten çekilmiş kafeterya tabakları). Telefon fotoğraflarında hata daha büyük olabilir.")
        extras = [f"{v:.0f} {UNITS.get(k, k)}" for k, v in pred.items() if k != "calories"]
        if extras:
            text += "\n\n" + " · ".join(extras)
        return crop_to_aspect(img), text

    with gr.Blocks(title="Kalori Tahmini") as demo:
        gr.Markdown("# Yemek fotoğrafından kalori tahmini")
        gr.Markdown(GUIDE)
        with gr.Row():
            inp = gr.Image(type="pil", label="Fotoğraf (tam üstten)")
            with gr.Column():
                seen = gr.Image(label="Modelin gördüğü", interactive=False)
                out = gr.Markdown()
        inp.change(run, inp, [seen, out])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--share", action="store_true", help="geçici herkese açık link (sunum için)")
    args = ap.parse_args()
    device = get_device()
    model, meta = load_checkpoint(args.ckpt, device)
    make_demo(model, meta, device).launch(share=args.share)
