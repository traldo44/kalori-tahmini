"""Demo (Adım 19): telefondan yemek fotoğrafı çek veya yükle → tahmini kcal, kütle ve makrolar.

    python app.py                      # Mac'te: http://127.0.0.1:7860 (varsayılan checkpoint: convnext_tiny_reg_mt.pt)
    python app.py --lan                # aynı Wi-Fi'daki telefondan: http://<mac-ip>:7860
    python app.py --share              # geçici herkese açık gradio.live linki (sunum yedeği)
    python scripts/deploy_space.py     # Hugging Face Space (README §7 Faz 4; Gradio Space için PRO gerekiyor)

Hugging Face Space'te bu dosya olduğu gibi çalışır (argüman yok → varsayılanlar).
"""
import argparse
import csv
import os
import socket
from pathlib import Path

import gradio as gr

from src.config import CALORIE_BIN_EDGES, get_device
from src.models import load_checkpoint, predict_image
from src.transforms import crop_to_aspect

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = os.environ.get("KALORI_CKPT", str(ROOT / "checkpoints" / "convnext_tiny_reg_mt.pt"))
EXAMPLES_DIR = ROOT / "demo_ornekleri"
DAILY_KCAL = 2000  # etiketlerdeki referans günlük enerji
MACROS = [("fat", "Yağ", 9), ("carb", "Karbonhidrat", 4), ("protein", "Protein", 4)]  # (anahtar, ad, kcal/g)

# Sakin, "ürün gibi" görünüm: tek vurgu rengi, az yuvarlak köşe, sistem yazı tipi; açık/koyu temaya uyar.
THEME = gr.themes.Base(
    primary_hue="emerald", neutral_hue="stone", radius_size="sm", spacing_size="md", text_size="md",
    font=("-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", "sans-serif"),
).set(
    button_primary_background_fill="*primary_600", button_primary_background_fill_hover="*primary_700",
    block_shadow="none", block_border_width="1px",
)

CSS = """
.gradio-container {max-width: 560px !important; margin: 0 auto !important; padding: 0 14px 40px !important;}
#baslik h1 {font-size: 1.45rem; font-weight: 600; letter-spacing: -0.01em; margin: 18px 0 2px;}
#baslik p {margin: 0 0 10px; color: var(--body-text-color-subdued); font-size: .95rem;}
#sonuc {border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 16px 18px 14px; margin-top: 4px;
        background: var(--block-background-fill);}
#sonuc .kcal {font-size: 2.9rem; font-weight: 600; line-height: 1; letter-spacing: -0.02em;}
#sonuc .kcal small {font-size: 1.1rem; font-weight: 500; margin-left: 4px; color: var(--body-text-color-subdued);}
#sonuc .aralik {margin-top: 6px; color: var(--body-text-color-subdued); font-size: .92rem;}
#sonuc .etiket {display: inline-block; margin-top: 10px; padding: 2px 8px; border-radius: 4px; font-size: .8rem;
                border: 1px solid var(--border-color-primary); color: var(--body-text-color);}
#sonuc table {width: 100%; border-collapse: collapse; margin-top: 14px; font-size: .95rem;}
#sonuc td {padding: 6px 0; border-top: 1px solid var(--border-color-primary); vertical-align: middle;}
#sonuc td.ad {width: 34%;}
#sonuc td.deger {width: 20%; text-align: right; font-variant-numeric: tabular-nums; padding-right: 10px;}
#sonuc td.cubuk div {height: 6px; border-radius: 3px; background: var(--border-color-primary); overflow: hidden;}
#sonuc td.cubuk div span {display: block; height: 100%; background: var(--primary-500);}
#sonuc .not {margin-top: 12px; font-size: .85rem; color: var(--body-text-color-subdued); line-height: 1.4;}
#bos {border: 1px dashed var(--border-color-primary); border-radius: 8px; padding: 18px; color: var(--body-text-color-subdued);
      font-size: .95rem; line-height: 1.5;}
#bos ol {margin: 6px 0 0 18px; padding: 0;}
#bos li {margin: 2px 0;}
#altbilgi {margin-top: 22px; font-size: .82rem; color: var(--body-text-color-subdued); line-height: 1.5;}
#altbilgi a {color: inherit;}
/* telefon: hiçbir blok içeriğinden dolayı ekrandan geniş olmasın; örnek küçük resimleri alt satıra geçsin */
.gradio-container > *, .gradio-container .main, .gradio-container .wrap, .gradio-container .block,
.gradio-container .form, .gradio-container .row, .gradio-container .column {min-width: 0 !important; max-width: 100% !important;}
.gradio-container .gallery {flex-wrap: wrap !important;}
.gradio-container .gallery-item, .gradio-container .gallery > button {flex: 0 0 auto;}
body {overflow-x: hidden;}
footer {display: none !important;}
"""

EMPTY = """<div id="bos"><strong>Nasıl çekilir?</strong>
<ol>
<li>Tabağı tam üstten, kuş bakışı çekin; yandan değil.</li>
<li>Tabak karenin yarısı ile üçte ikisi arasını doldursun.</li>
<li>Tek tabak, düz zemin, gölgesiz ışık.</li>
</ol>
Model bu koşullardaki tabaklarla eğitildi; yandan veya çok yakından çekimde sapma büyür.</div>"""

GUIDE = """
Model, Nutrition5k veri setindeki 2 752 tabakla eğitildi: tek bir kafeterya, sabit yükseklikte üst kamera,
tabak başına tartılmış malzeme listesinden hesaplanan kalori. Telefon fotoğrafı bu koşullardan uzaklaştıkça
(yandan açı, farklı mesafe, birden fazla tabak, paketli ürün) hata büyür. "Modelin gördüğü" bölümü, fotoğrafın
eğitimdeki 4:3 oranına kırpılmış hâlidir; tabağın kenarı kesiliyorsa biraz uzaktan çekin.

Sonuç bir tahmindir: aynı tabak için test kümesinde ortalama hata ±40 kcal; yüksek kalorili tabaklarda daha büyük.
"""

ABOUT = """
| | |
| --- | --- |
| Model | ConvNeXt-T omurgası (ImageNet-12k ön eğitimli), çok görevli başlık: kalori, kütle, yağ, karbonhidrat, protein |
| Girdi | Yalnızca RGB fotoğraf, 224 px |
| Eğitim | Nutrition5k, 2 752 tabak (üst kamera) |
| Test, 507 tabak | MAE 40,3 kcal [%95 GA 36,2–45,3], PMAE %15,8, R² 0,90; tahminlerin %88'i ±100 kcal içinde |
| Karşılaştırma | Sıfır-atış VLM (Gemini 3.5 Flash-Lite) 82,1 kcal · klasik ML (donmuş ResNet-50 → XGBoost) 77,4 · Nutrition5k makalesi RGB 70,6 |

Kod, deneyler ve rapor: [github.com/traldo44/kalori-tahmini](https://github.com/traldo44/kalori-tahmini)
"""


def load_examples():
    """demo_ornekleri/gercek_degerler.csv → [[yol], ...] ve gerçek kalori tablosu."""
    csv_path = EXAMPLES_DIR / "gercek_degerler.csv"
    if not csv_path.exists():
        return [], ""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    examples = [[str(EXAMPLES_DIR / r["file"])] for r in rows if (EXAMPLES_DIR / r["file"]).exists()]
    table = "| Örnek | Gerçek | Model |\n| --- | ---: | ---: |\n" + "\n".join(
        f"| {i} | {r['gercek_kcal']} kcal | {r['model_kcal']} kcal |" for i, r in enumerate(rows, 1))
    return examples, table


def calorie_band(kcal):
    lo, hi = CALORIE_BIN_EDGES
    return "düşük kalorili" if kcal < lo else "orta kalorili" if kcal < hi else "yüksek kalorili"


def result_html(pred, typical):
    kcal = pred["calories"]
    daily = 100 * kcal / DAILY_KCAL
    parts = [f'<div id="sonuc"><div class="kcal">{kcal:.0f}<small>kcal</small></div>']
    if typical:
        parts.append(f'<div class="aralik">yaklaşık {max(kcal - typical, 0):.0f}–{kcal + typical:.0f} kcal arası · '
                     f"günlük {DAILY_KCAL} kcal'e göre %{daily:.0f}</div>")
    parts.append(f'<span class="etiket">{calorie_band(kcal)}</span>')
    macros = [(ad, pred[k], f) for k, ad, f in MACROS if k in pred]
    if macros or "mass" in pred:
        parts.append("<table>")
        if "mass" in pred:
            parts.append(f'<tr><td class="ad">Porsiyon</td><td class="deger">{pred["mass"]:.0f} g</td><td class="cubuk"></td></tr>')
        if macros:
            energy = [g * f for _, g, f in macros]
            total = sum(energy) or 1
            for (ad, g, _), e in zip(macros, energy):
                pct = 100 * e / total
                parts.append(f'<tr><td class="ad">{ad}</td><td class="deger">{g:.0f} g</td>'
                             f'<td class="cubuk"><div><span style="width:{pct:.0f}%"></span></div></td></tr>')
        parts.append("</table>")
        if macros:
            parts.append(f'<div class="not">Çubuklar enerjinin makrolara dağılımı (yağ 9, karbonhidrat ve protein 4 kcal/g); '
                         f'makrolardan hesaplanan enerji ≈ {sum(energy):.0f} kcal.</div>')
    parts.append("</div>")
    return "".join(parts)


def make_demo(model, meta, device):
    typical = meta["info"].get("test_MAE")
    examples, example_table = load_examples()

    def run(img):
        if img is None:
            return EMPTY, None
        pred = predict_image(model, meta, img, device)
        return result_html(pred, typical), crop_to_aspect(img)

    camera = gr.WebcamOptions(mirror=False, constraints={"video": {"facingMode": "environment"}})
    with gr.Blocks(title="Kalori Tahmini", theme=THEME, css=CSS) as demo:
        gr.HTML('<div id="baslik"><h1>Kalori tahmini</h1><p>Tabağın üstten fotoğrafından kalori, porsiyon ve makro tahmini.</p></div>')
        inp = gr.Image(type="pil", sources=["upload", "webcam"], webcam_options=camera, label="Fotoğraf", height=300)
        out = gr.HTML(EMPTY)
        if examples:
            gr.Examples(examples=examples, inputs=inp, label="Denemek için örnek tabaklar", examples_per_page=4)
        with gr.Accordion("Modelin gördüğü", open=False):
            seen = gr.Image(label="4:3 kırpılmış görüntü", interactive=False, show_label=False)
        if example_table:
            with gr.Accordion("Örneklerin gerçek değerleri", open=False):
                gr.Markdown(example_table)
        with gr.Accordion("Sınırlılıklar", open=False):
            gr.Markdown(GUIDE)
        with gr.Accordion("Model ve sonuçlar", open=False):
            gr.Markdown(ABOUT)
        gr.HTML('<div id="altbilgi">Bitirme projesi, 2026. Tahminler bilgilendirme amaçlıdır; tıbbi veya diyet kararı için kullanılmaz. '
                '<a href="https://github.com/traldo44/kalori-tahmini">Kaynak kod</a></div>')
        inp.change(run, inp, [out, seen])
    return demo


def local_ip():
    """Telefonun ulaşabileceği Wi-Fi adresi. macOS'ta en0/en1 (sanal ağ arayüzleri 172.x yanıltıyor), yoksa varsayılan rota."""
    import subprocess
    for iface in ("en0", "en1"):
        try:
            ip = subprocess.run(["ipconfig", "getifaddr", iface], capture_output=True, text=True, timeout=3).stdout.strip()
            if ip:
                return ip
        except (OSError, subprocess.TimeoutExpired):
            pass
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
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
        print(f"\nTelefondan (aynı Wi-Fi): http://{local_ip()}:{args.port}\n", flush=True)
    make_demo(model, meta, device).launch(share=args.share, server_name="0.0.0.0" if (args.lan or os.environ.get("SPACE_ID")) else None,
                                          server_port=args.port)
