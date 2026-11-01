"""Demo (Adım 19): telefondan yemek fotoğrafı çek veya yükle → tahmini kcal, kütle ve makrolar.

    python app.py                      # Mac'te: http://127.0.0.1:7860 (varsayılan checkpoint: convnext_tiny_reg_mt.pt)
    python app.py --lan                # aynı Wi-Fi'daki telefondan: http://<mac-ip>:7860
    python app.py --share              # geçici herkese açık gradio.live linki (sunum yedeği)
    python scripts/deploy_space.py     # kalıcı adres: Hugging Face Space (README §7 Faz 4)

Hugging Face Space'te bu dosya olduğu gibi çalışır (argüman yok → varsayılanlar).
"""
import argparse
import csv
import os
import socket
from pathlib import Path

import gradio as gr

from src.config import get_device
from src.models import load_checkpoint, predict_image
from src.transforms import crop_to_aspect

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = os.environ.get("KALORI_CKPT", str(ROOT / "checkpoints" / "convnext_tiny_reg_mt.pt"))
EXAMPLES_DIR = ROOT / "demo_ornekleri"

UNITS = {"mass": ("Kütle", "g"), "fat": ("Yağ", "g"), "carb": ("Karbonhidrat", "g"), "protein": ("Protein", "g")}

GUIDE = """
Model, tek bir kafeteryada sabit yükseklikteki üst kamerayla çekilmiş 2 752 tabakla (Nutrition5k) eğitildi.
En iyi sonuç için:

- Tabağı **tam üstten** (kuş bakışı) çekin, yandan değil
- Tabak fotoğrafın genişliğinin yaklaşık **yarısı ile üçte ikisi** arasını kaplasın
- Tek tabak, düz zemin, iyi ışık

Bu koşullara uymayan fotoğraflarda (yandan, çok yakından, birden fazla tabak) tahmin çok sapabilir.
"Modelin gördüğü" bölümünde fotoğrafın eğitimdeki 4:3 orana kırpılmış hâli gösterilir.
"""

ABOUT = """
**Model:** ConvNeXt-T omurgası (ImageNet-12k ön eğitimli), çok görevli başlık (kalori + kütle + yağ + karbonhidrat + protein),
224 px, yalnızca RGB. **Nutrition5k test (507 tabak):** MAE **40,3 kcal** [%95 GA 36,2–45,3], PMAE %15,8, R² 0,90,
tahminlerin %88'i gerçeğin ±100 kcal içinde. Karşılaştırma: sıfır-atış VLM (Gemini 3.5 Flash-Lite) 82,1 kcal,
klasik ML (donmuş ResNet-50 → XGBoost) 77,4 kcal, Nutrition5k makalesinin RGB baseline'ı 70,6 kcal.

Kod ve rapor: `github.com/traldo44/kalori-tahmini`.
"""

CSS = """
.kcal-box {text-align: center; padding: 8px 0 4px;}
.kcal-box h1 {font-size: 3.2rem; margin: 0; line-height: 1.1;}
.kcal-box p {margin: 4px 0 0; opacity: .75;}
footer {display: none !important;}
"""


def load_examples():
    """demo_ornekleri/gercek_degerler.csv → [[yol], ...] ve açıklama tablosu."""
    csv_path = EXAMPLES_DIR / "gercek_degerler.csv"
    if not csv_path.exists():
        return [], ""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    examples = [[str(EXAMPLES_DIR / r["file"])] for r in rows if (EXAMPLES_DIR / r["file"]).exists()]
    table = "| Örnek | Gerçek kcal |\n| --- | --- |\n" + "\n".join(
        f"| {i} | {r['gercek_kcal']} |" for i, r in enumerate(rows, 1))
    return examples, table


def make_demo(model, meta, device):
    typical = meta["info"].get("test_MAE")
    examples, example_table = load_examples()

    def run(img):
        if img is None:
            return "", "", None
        pred = predict_image(model, meta, img, device)
        kcal = pred["calories"]
        err = f"<p>Test kümesinde tipik hata ±{typical:.0f} kcal</p>" if typical else ""
        kcal_html = f'<div class="kcal-box"><h1>≈ {kcal:.0f} kcal</h1>{err}</div>'
        extras = [(UNITS[k][0], v, UNITS[k][1]) for k, v in pred.items() if k in UNITS]
        detail = ""
        if extras:
            detail = "| | |\n| --- | ---: |\n" + "\n".join(f"| {n} | {v:.0f} {u} |" for n, v, u in extras)
        return kcal_html, detail, crop_to_aspect(img)

    camera = gr.WebcamOptions(mirror=False, constraints={"video": {"facingMode": "environment"}})
    with gr.Blocks(title="Kalori Tahmini", css=CSS) as demo:
        gr.Markdown("# 🍽️ Yemek fotoğrafından kalori tahmini")
        gr.Markdown("Tabağı **tam üstten** çekin veya galeriden yükleyin; sonuç birkaç saniyede gelir.")
        inp = gr.Image(type="pil", sources=["upload", "webcam"], webcam_options=camera,
                       label="Fotoğraf çek veya yükle", height=320)
        kcal_out = gr.HTML()
        detail_out = gr.Markdown()
        if examples:
            gr.Examples(examples=examples, inputs=inp, label="Örnek tabaklar (Nutrition5k test kümesinden)")
            with gr.Accordion("Örneklerin gerçek kalorileri", open=False):
                gr.Markdown(example_table)
        with gr.Accordion("Modelin gördüğü (4:3 kırpma)", open=False):
            seen = gr.Image(label="Kırpılmış görüntü", interactive=False)
        with gr.Accordion("Nasıl çekmeli? Sınırlılıklar", open=False):
            gr.Markdown(GUIDE)
        with gr.Accordion("Model ve sonuçlar hakkında", open=False):
            gr.Markdown(ABOUT)
        inp.change(run, inp, [kcal_out, detail_out, seen])
    return demo


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="GitHub Release'den: GH_TOKEN=$(gh auth token) python scripts/share_checkpoint.py download convnext_tiny_reg_mt.pt")
    ap.add_argument("--share", action="store_true", help="geçici herkese açık link (sunum yedeği)")
    ap.add_argument("--lan", action="store_true", help="aynı ağdaki telefondan erişim için 0.0.0.0'a bağlan")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = ap.parse_args()
    device = get_device()
    model, meta = load_checkpoint(args.ckpt, device)
    if args.lan:
        print(f"\nTelefondan (aynı Wi-Fi): http://{local_ip()}:{args.port}\n")
    make_demo(model, meta, device).launch(share=args.share, server_name="0.0.0.0" if (args.lan or os.environ.get("SPACE_ID")) else None,
                                          server_port=args.port)
