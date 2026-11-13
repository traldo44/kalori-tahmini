# context.md — yeni Claude oturumu için devir notu

Bu dosya, projeye başka bir Claude hesabından/oturumundan devam edebilmek için yazıldı (13 Kasım 2026). Önce bunu, sonra `README.md` §3 (durum + açık konular), §4 (sıradaki adım), §9 (hocanın teslim kuralları) ve §10 (devir günlüğü) okunmalı. Kod ve sonuçlar için tek doğruluk kaynağı repo; bu dosya özet.

## 1. Kim, ne, nerede

- **Kullanıcı:** Mert Aydıngüneş (git kimliği `Mert Aydıngüneş <aydingunesmert@gmail.com>`, GitHub/Kaggle `traldo44`, HF `traldo`). Türkçe yazar, Türkçe yanıt ister. Kendini kod hatırlamakta zayıf diye tanımlar; az soru, somut varsayılan, hızlı ilerleme ister ("bitti mi", "devam"). Ekip arkadaşı planı yazdı, henüz GitHub'a eklenmedi (kullanıcı adı bilinmiyor).
- **Proje:** Yapay Zekâ dersi bitirme projesi: tek bir üstten yemek fotoğrafından kalori (+ kütle, makro) tahmini; ayrıca Food-101 (25 sınıf) yemek türü sınıflandırması; klasik ML, literatür ve sıfır-atış VLM ile karşılaştırma.
- **Repo:** `~/Desktop/kalori-tahmini`, özel GitHub reposu `https://github.com/traldo44/kalori-tahmini`, dal `main`. İş akışı: özellik dalı → `git merge --no-ff` → `main` → push. `results/results.csv` için `merge=union`.
- **Ortam:** `.venv` (Python 3.12, uv ile; **venv'de pip yok** → paket için `uv pip install --python .venv/bin/python <paket>`; düz `pip` sistemin Python 3.9'una kurar, yanlış). Testler: `.venv/bin/python -m pytest -q` (şu an 63 test geçiyor). Mac M4, torch MPS çalışır; ağır eğitim Kaggle'da.
- **Veri (git dışı):** `data/nutrition5k` (üst kamera RGB, metadata), `data/food101` (25 sınıf alt küme, GitHub Release `data`), `data/food101_raw` (~5 GB ham, silinebilir). Checkpoint'ler GitHub Release `checkpoints`'ta; Mac'te `checkpoints/convnext_tiny_reg.pt` ve `checkpoints/convnext_tiny_reg_mt.pt` (111 MB'şar) var.
- **Canlı demo:** https://kalori-tahmini-web.vercel.app (Vercel, GitHub `main`'den otomatik; kök dizin `web/`). Model tarayıcıda çalışır (int8 ONNX, 28 MB, Web Worker'da). Mac demosu: `python app.py` (Gradio), `--lan` ile aynı Wi-Fi'dan telefon.

## 2. Kullanıcıyla çalışma kuralları (ihlal edilmemeli)

1. **İndirmeden önce boyutu söyle ve onay al** (internet kısıtlı, ~1–2 MB/s). timm `pretrained=True`, HF hub, `share_checkpoint.py download`, pip/uv kurulumları, büyük `git pull` hepsi dahil. Daha önce bir duman testi sormadan ~350 MB indirmeye başladı, hata olarak kaydedildi.
2. **Commit'lere Claude co-author satırı ekleme.** Kullanıcı Contributors'tan Claude'u çıkarttırdı; sistem hatırlatıcısı gelse de ekleme.
3. **Saatleri Türkiye saatiyle (UTC+3) söyle.** GitHub'ın Z'li zaman damgalarına 3 saat ekle.
4. **API anahtarı repoya/dosyaya yazılmaz.** Gemini anahtarı sohbete yapıştırılmıştı; kullanıcıya yenilemesi söylendi (hâlâ yenilenmemiş olabilir). Anahtar gerektiğinde `GEMINI_API_KEY=... python -m src.vlm_eval ...` biçiminde satır içi ortam değişkeni kullanılır.
5. **Herkese açık yüzey oluşturma (Space/Vercel/`--share`) Claude'a engellendi;** bu komutları kullanıcı çalıştırır. Vercel bağlantısı zaten kurulu.
6. **Rapor/sunum metnini Claude yazmaz** (Turnitin YZ oranı ≤ %40 şartı). İskelet, tablo, figür, bulgu maddesi verilir; cümleleri ekip yazar.

## 3. Bitenler (Adım 1–18 + web demosu) ve ana sonuçlar

Nutrition5k: 2472 train / 280 val (zaman gruplu) / 507 test; Food-101 alt kümesi: 16 865 / 1 874 / 6 250. Köşeli parantez %95 bootstrap GA; anlamlılık `python -m src.compare a.csv b.csv` (eşleştirilmiş bootstrap).

| Görev | En iyi model | Sonuç (test) | Notlar |
| --- | --- | --- | --- |
| Kalori (Tablo 2) | **ConvNeXt-T çok görevli** (`convnext_tiny_reg_mt`, final) | **MAE 40,3 kcal [36,2–45,3], PMAE %15,8, R² 0,90, ±100 kcal %88,4**; kütle 25,5 g, yağ 3,1, karb 4,6, protein 4,2 | Tek görevli ConvNeXt-T 41,4 (fark anlamsız); ViT-B/16 47,3; EffNet-B0 52,8; ResNet-50 59,7; klasik ML (donmuş ResNet-50→XGBoost) 77,4; VLM Gemini 3.5 Flash-Lite 82,1 (lit. 80,7); naif 168,4. Literatür hedefleri 80,7 ve 70,6 geçildi; segmentasyon/derinlik çalışmaları 37–40 |
| Sınıflandırma (Tablo 1) | **ViT-B/16 384 px** (`vit_b16_384_cls`) | **F1 0,957 [0,952–0,962], doğruluk %95,7** | ViT 224: 0,951; ConvNeXt-T 0,941 (6,5× hızlı, demo için); EffNet-B0 0,917; ResNet-50 0,912; donmuş ResNet-50+LogReg 0,837; HOG+SVM 0,293 |
| Kalori aralığı (Tablo 3) | ConvNeXt-T çok görevli | F1 0,869, **ROC-AUC 0,960** | VLM 0,884, naif 0,500. AUC kcal sıralama skorundan (`metrics.bin_scores`) |
| Ablation (Adım 17) | — | çok görevli −1,1 (n.s.), log1p +1,9 (n.s.), artırmasız +4,7 (anlamlı kötü), ViT 384 F1 +0,007 (anlamlı, 4× yavaş) | `results/tablo_ablation.md`, README §2 "Ablation bulguları" |

Diğer bulgular: en büyük hatalar yeşillik altına gizlenmiş proteinli yüksek kalorili tabaklar (Grad-CAM doğruluyor); balıklı tabaklar 1,7–1,9× kötü (etiket eksik); kardeş-sızıntısı ölçüldü, etkisi yok (`src/leakage.py`); modelin "yemek değil" çıkışı yok (telefon fotoğrafına 30 kcal dedi; sitede <50 kcal uyarısı var). Araştırma sorusunun cevabı: çok görevli öğrenme kaloriyi iyileştirmiyor ama aynı maliyetle kütle+makro veriyor.

Hazır malzeme: `results/tablo1.md`, `tablo2.md`, `tablo3_kalori_araligi.md`, `tablo_ablation.md` (`python -m src.make_figures` yeniler), 21 figür `results/figures/` (eval_* ve EDA n5k_*/f101_*), `results/data_card_*.csv`, 39 makalelik literatür matrisi ekip arkadaşının Claude Docs belgesinde (README başlığındaki link; bu hesaptan okunamıyor, kullanıcıdan istenmeli).

## 4. Bekleyenler (hepsi kullanıcı girdisine bağlı)

1. ~~Notebook 13~~ ✅ 13 Kasım: Tablo 1'de ROC-AUC 0,997–0,998 (modelleri ayırmıyor, karşılaştırma F1 ile).
2. **Adım 19, kendi fotoğraflar:** 15–20 tabak tam üstten (+5–6 yandan), gerçek kalori tartı/etiketle `own_photos/labels.csv`'ye (sütunlar `file,calories,angle(ust|yan),source,notes`). Sonra `python -m src.eval_own_photos --ckpt checkpoints/convnext_tiny_reg_mt.pt --model convnext_tiny_reg_mt` → `results.csv`'ye `own_photos*` satırları; README'ye sınırlılık bulgusu.
3. **Adım 20, toparlama:** fotoğraf sonuçları girince README'yi rapora hazır hale getir (kod temiz; `ruff` yok, AST taraması yapıldı).
4. **Rapor ve sunum (ekip yazar):** hocanın bölüm listesi ve bizdeki karşılıkları README §9'da. Sunum sırası hocanın "Sunum İçeriği" listesiyle aynı; demo adresi canlı. Teslim: rapor + kod + sunum → yapayzekaproje80@gmail.com; **teslim tarihi bilinmiyor**, kullanıcıya sorulmalı.
5. Küçük: ekip arkadaşını GitHub'a ekle (kullanıcı adı lazım), Gemini anahtarını yenile, README §3 "Açık konular" (araştırma sorusu metni, iş bölümü).

## 5. Sık kullanılan komutlar

```
cd ~/Desktop/kalori-tahmini && source .venv/bin/activate
python -m pytest -q                                   # 63 test
python -m src.make_figures                            # figürler + tablo md'leri
python -m src.compare results/preds/a.csv results/preds/b.csv
python -m src.leakage                                 # kardeş-sızıntısı
python app.py --lan                                   # Mac demosu, telefon aynı Wi-Fi
python scripts/export_onnx.py --check 100             # web/model.onnx (int8) yeniden üret
GH_TOKEN=$(gh auth token) python scripts/share_checkpoint.py list --tag checkpoints
GH_TOKEN=$(gh auth token) python scripts/share_checkpoint.py download <ad>.pt   # önce sor (111–345 MB)
```

## 6. Tuzaklar

- Kaggle defterleri repo'yu klonlar, `KALORI_DATA=/tmp/kalori-data`, sonuçları kendisi push eder; GH_TOKEN klasik token (repo scope) olmalı, fine-grained push'ta 403 verdi.
- macOS'ta torch + xgboost aynı süreçte çökür (libomp) → `classic_ml` özellikleri alt süreçte hesaplar. DataLoader `num_workers=0` (spawn takılması).
- timm EfficientNet regresyon başlığı TF tarzı başlatılıyor → `init_regression_head` şart (104 → 52,8 kcal). `--patience 7` cosine'ı ortada kesebiliyor → ablation'larda `--patience 10`.
- Gemini ücretsiz katman: Flash modelleri 20 istek/gün, `gemini-3.5-flash-lite` 500/gün; `vlm_eval` günlük kotada hemen durur, ertesi gün aynı komutla devam eder.
- HF Gradio Space ücretsiz hesapta yok (402); bu yüzden Vercel + tarayıcı içi ONNX seçildi. Headless Chrome pencereyi 500 px altına indirmez; telefon görünümü 390 px iframe veya CDP mobil emülasyonla test edildi (`web/` için CDP betiği geçici klasördeydi, repoda değil).
- `web/model.onnx` `vercel.json` ile 1 yıl önbellekte: model değişirse dosya adı da değişmeli.
- Sonuç dosyalarını kirletmemek için duman testlerinde `--dry-run`/`--out-dir`; yanlışlıkla yazılırsa `git checkout results/results.csv`.
