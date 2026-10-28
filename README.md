# Yemek Fotoğrafından Kalori Tahmini — Proje Durumu ve Geliştirici Rehberi

> **Son güncelleme:** 28 Ekim 2026 (repo kuruldu, Adım 1–4 tamam; ayrıntı §3 ve §10)
> **Kime:** Geliştirmeyi devralan ve kaldığı yerden sürdürecek kişi
> **Ayrıntılı yol haritası ve 39 makalelik literatür matrisi:** [Proje yol haritası belgesi](https://claude.ai/code/artifact/ff3e6427-a400-4a8f-a87e-6713e9a1e544). Belge size paylaşılmadıysa proje sahibinden isteyin.

Bu dosyayı okuyan biri, başka kaynağa bakmadan projenin ne olduğunu, nerede olduğumuzu ve sıradaki adımın ne olduğunu anlayabilmeli. Geliştirme tek bir sıralı akış olarak ilerler (§4). Çalışmayı bıraktığınızda **§3 Durum** tablosunu güncelleyin ve **§10 Devir günlüğü**ne kaldığınız yeri yazın; sonraki kişi oradan devam eder.

---

## 1. Proje bir bakışta

**Amaç:** Tek bir yemek fotoğrafından tabaktaki toplam kaloriyi (kcal) tahmin eden bir derin öğrenme modeli geliştirmek. Ayrıca bu modeli klasik yöntemlerle, literatürle ve hazır görü-dil modelleriyle (VLM) karşılaştırmak.

**Araştırma sorusu:** Üstten çekilmiş tek bir RGB fotoğraftan derin öğrenme tabanlı bir model toplam kaloriyi ne kadar doğru tahmin edebilir? İki görevli yaklaşım (yemek türü + kalori), uçtan uca regresyona ve sıfır-atış VLM'lere göre daha başarılı mı?

**Mimari:** Ortak bir omurga (ResNet-50 / EfficientNet / ConvNeXt / ViT) iki başlığı besler:

```
Fotoğraf → Ön işleme (224 px, normalizasyon, augmentation)
        → Ortak omurga (ImageNet ön eğitimli, transfer öğrenme)
            ├── Yemek türü başlığı (softmax)      → Tablo 1: accuracy, precision, recall, F1
            └── Kalori başlığı (kütle × yoğunluk)  → Tablo 2: MAE, PMAE, RMSE, R²
                                                    → Tablo 3: kalori aralığı F1 / recall
```

**Neden üç tablo?** Kalori sürekli bir sayıdır, bu yüzden accuracy, F1 ve recall ona doğrudan uygulanamaz. Hocanın istediği sınıflandırma metriklerini iki yerden üretiyoruz: yemek türü sınıflandırmasından ve kalori tahminlerini düşük / orta / yüksek aralıklarına çevirerek.

**Takvim:** Geliştirme toplam 9 gün. Rapor ve sunum geliştirme bittikten sonra hazırlanacak; bu dosyanın kapsamı dışında.

---

## 2. Bugüne kadar alınan kararlar

| Konu | Karar | Gerekçe |
| --- | --- | --- |
| Ana veri seti (kalori) | **Nutrition5k**, yalnızca üstten RGB görüntüler | Gerçek, tartıyla ölçülmüş kalori etiketi var, resmi train/test bölmesi var ve literatürde çok sayıda sonuç yayınlanmış |
| Sınıflandırma veri seti | **Food-101**, başlangıçta 20–30 sınıflık alt küme | Standart benchmark. Tam set yaklaşık 5 GB olduğu için zaman kısıtında alt küme yeterli |
| Yan veri setleri (opsiyonel) | ECUSTFD (hacim), SNAPMe veya DiningBench (dış test) | Ancak zaman kalırsa kullanılacak |
| Girdi türü | Yalnızca RGB, gerçek derinlik kullanılmayacak | Gerçek hayattaki telefon senaryosuna uygun. Derinlik gerekirse tek görüntüden tahmin edilecek |
| Değerlendirme protokolü | Nutrition5k resmi test bölmesi. MAE + PMAE birlikte raporlanacak, girdi türü açıkça yazılacak | Literatürle adil karşılaştırma için şart |
| VLM'in rolü | Eğitim yok, sabit prompt'la **sıfır-atış referans satırı** | Literatürde hazır VLM'ler eğitilmiş modellerin 2–3 katı hata yapıyor |
| Literatür | **Tamamlandı**: 39 çalışma, sayılar kaynaklarından doğrulandı | Ayrıntılar yol haritası belgesinin "Literatür matrisi" sekmesinde |
| Platform | **Kaggle** (eğitim) + **Mac** (kod, EDA, klasik ML, VLM betiği, demo) | Kaggle: haftalık ~30 saat GPU, 12 saatlik kesintisiz oturum. Mac (M4, 16 GB) eğitim dışı işler için yeterli |
| Ortak çalışma | **GitHub** (kod, `results.csv`, tahmin dosyaları) + GitHub Release (checkpoint) | §6.4 |
| Doğrulama kümesi | Train'den **zaman gruplarıyla** ayrılır, rastgele değil | Tabaklar malzeme malzeme eklenerek art arda çekilmiş; rastgele ayırma sızıntı yapar (§6.1) |
| Kalori için veri artırma | Yalnızca çevirme, 90° döndürme, renk. **Ölçek değiştiren artırma ve kırpma yok** | Model porsiyonu tabağın kapladığı alandan öğrenir (§6.1, `src/transforms.py`) |
| Sonuçların güvenilirliği | MAE ve F1 için **%95 bootstrap güven aralığı**; modeller **eşleştirilmiş bootstrap** ile karşılaştırılır | Test kümesi 507 tabak; birkaç kcal'lik fark şans eseri olabilir (§6.2) |

### Literatürden çıkan ve tasarımı yönlendiren bulgular

1. **Yemek tanıma büyük ölçüde çözülmüş; asıl hata porsiyondan geliyor.** Food-101'de top-1 doğruluk %90'ın üzerinde. Nutrition5k'da ise yalnızca RGB ile kalori hatası %26,1. Derinlik eklenince %18,8'e, gerçek kütle verilince %9,5'e iniyor.
2. **Tek RGB görüntüyle de iyi sonuç alınabiliyor.** Tahmin edilmiş derinlik (DPF-Nutrition, %14,7) ve segmentasyon (UNet + ResNet-101, %15,7) bunu sağlıyor. İyileştirme fazının ana adayları bunlar.
3. **Hazır VLM'ler zayıf.** Nutrition5k'da en iyi sıfır-atış VLM 80,7 kcal MAE yapıyor; ince ayarlı özel modeller 27–40 kcal aralığında.
4. **Ön eğitim verisi belirleyici.** Açık veriyle (ImageNet) ön eğitilmiş ViT 88 kcal, ResNet-50 117 kcal hata yapmış. Orijinal makalenin 70,6 kcal sonucu, kamuya açık olmayan JFT-300M ön eğitimiyle alınmış.

### Hedeflerimiz (Nutrition5k test, yalnızca RGB)

| Referans | Kalori MAE (kcal) | PMAE (%) | Bizim için anlamı |
| --- | --- | --- | --- |
| Ortalama tahmin baseline'ı | 150,8 | 60,2 | Mutlaka geçilmeli |
| ResNet-50, ImageNet ön eğitimli (Andrade 2025) | 117,4 | 46,6 | İlk derin model bu civarda çıkabilir |
| En iyi sıfır-atış VLM (Gemini 3.0 Flash) | 80,7 | — | **1. hedef** |
| Orijinal Nutrition5k RGB baseline'ı (JFT ön eğitimli) | 70,6 | 26,1 | **2. hedef** |
| Segmentasyon veya tahmini derinlikle güncel çalışmalar | 37–40 | 14,7–15,7 | İyileştirme fazında yaklaşılacak üst sınır |

---

## 3. Durum — şu an neredeyiz?

| Faz | İş | Durum |
| --- | --- | --- |
| 0 | Literatür taraması (39 makale) | ✅ Tamamlandı |
| 1 | Veri seti seçimi ve gerekçesi | ✅ Tamamlandı |
| — | Platform seçimi | ✅ Kaggle (eğitim) + Mac. Kaggle hesabı/kurulumu ⏳ |
| 2 | Kurulum, veri indirme | ✅ Mac'te: repo, `metrics.py`, veri okuyucular, testler; Nutrition5k indirildi |
| 2 | Temizlik, EDA | ⬜ Gün 1–2 (1 hatalı tabak şimdiden bulundu, `src/data.py` → `BAD_DISHES`) |
| 3 | Baseline modeller | ⬜ Gün 2–3 |
| 3 | CNN / Transformer eğitimi | ⬜ Gün 3–5 |
| 4 | Değerlendirme + VLM referansı | ⬜ Gün 5–6 |
| 5 | İyileştirme ve ablation | ⬜ Gün 6–8 |
| — | Tablolar, figürler, kodun temizlenmesi | ⬜ Gün 8–9 |
| 6–7 | Rapor ve sunum | ⬜ Geliştirmeden sonra |

**Kısaca:** Repo kuruldu; ortak kod (metrikler, veri okuyucular, dönüşümler, demo iskeleti) yazıldı ve testleri geçiyor. Nutrition5k Mac'e indirildi. Sıradaki adımlar: **Kaggle kurulumu (§5.1)**, **Adım 5** (Food-101'i indir) ve **Adım 6** (Nutrition5k EDA).

### Açık konular (karar bekliyor)

| Konu | Durum |
| --- | --- |
| **Araştırma sorusu deneylerle uyuşmuyor** | §1'deki soru "yemek türü + kalori birlikte" diyor, ancak tür etiketi Food-101'de, kalori Nutrition5k'da; hiçbir fotoğrafta ikisi birden yok, bu yüzden soru hiçbir deneyde test edilmiyor. Öneri: soruyu Adım 12'ye uydurmak → "Kaloriyle birlikte kütle ve makroları da tahmin etmek, yalnızca kalori tahmininden daha mı iyi?" Food-101 sınıflandırması Tablo 1 için ayrı deney olarak kalır. |
| Teslim tarihi | Belgede yok. Gün 1'in tarihini ve teslim gününü §4'e yazın |
| İş bölümü | Adım 5'ten sonra iki bağımsız hat var: **kalori** (Nutrition5k EDA → kalori modelleri → iyileştirme) ve **sınıflandırma** (Food-101 → klasik ML → CNN/ViT → VLM betiği). İki kişi paralel çalışırsa süre ve GPU kotası ikiye katlanır |
| VLM seçimi ve bütçe | Hangi model, API anahtarı kimde? Gemini'nin ücretsiz katmanı 300 tabak için muhtemelen yeter |
| Kalori hedefi gerçekçi mi? | "80,7 kcal'in altı" iddialı: ImageNet ön eğitimli ViT literatürde 88 kcal'de kalmış. Tutmazsa raporda dürüstçe yazılır |

---

## 4. Geliştirme akışı (sıralı)

Adımlar sırayla yapılır; her adımın bir çıktısı ve bir "bitti" kriteri var. Bir adım bitmeden sonrakine geçmeyin, çünkü her adım bir öncekinin çıktısını kullanır. Ayrıntılar parantez içinde verilen bölümlerde.

| Adım | Gün | İş | Çıktı / bitti kriteri |
| --- | --- | --- | --- |
| 1 | 1 | Platformu seç ve ortamı kur (§5.1) | Paketler kurulu, GPU görünüyor |
| 2 | 1 | Repo iskeletini oluştur, bu dosyayı `README.md` olarak ekle (§5.2) | Repo açık, `.gitignore` hazır |
| 3 | 1 | `src/metrics.py` dosyasını yaz (§6.2) | Fonksiyonlar örnek veriyle çalışıyor |
| 4 | 1 | Nutrition5k'yı indir; metadata ve bölme okuyucularını yaz (§5.3) | Kullanılabilir train/test tabak sayıları not edildi |
| 5 | 1 | Food-101 alt kümesini hazırla (§5.4) | Sabit sınıf listesi `src/data.py` içinde |
| 6 | 2 | Nutrition5k EDA (§7, Faz 2) | Kalori histogramı, kalori–kütle grafiği, aykırı değer listesi, `log1p` kararı |
| 7 | 2 | Food-101 EDA ve veri kartı (§7, Faz 2) | Sınıf dağılımı, örnek ızgara, veri kartı tablosu |
| 8 | 3 | Naif baseline'lar: en sık sınıf ve train ortalaması | `results.csv`'de ilk satırlar |
| 9 | 3 | Klasik ML: HOG + renk → SVM, RF (sınıflandırma); CNN gömme → Ridge, XGBoost (kalori) | Tablo 1 ve 2'de klasik ML satırları. **Kontrol noktası 1** |
| 10 | 4 | Transfer CNN sınıflandırma: ResNet-50, EfficientNet-B0 | Tablo 1'de CNN satırları |
| 11 | 4 | Transfer CNN kalori regresyonu: ResNet-50 (yalnızca kalori) | Tablo 2'de ilk derin model satırı |
| 12 | 5 | Çok görevli kalori başlığı (kalori + kütle + makrolar) | Tablo 2'de çok görevli satır |
| 13 | 5 | Güncel omurgalar: ConvNeXt-T, ViT-B/16 (sınıflandırma ve kalori) | Tablo 1 ve 2'de güncel model satırları |
| 14 | 6 | VLM sıfır-atış betiği ve Nutrition5k test çalıştırması (§7, Faz 3) | Tablo 2'de VLM satırı, maliyet ve hatalı yanıt oranı |
| 15 | 6 | Değerlendirme: Tablo 3 ve tüm figürler (§7, Faz 4) | Karışıklık matrisi, tahmin–gerçek grafiği, hata histogramı, Grad-CAM. **Kontrol noktası 2** |
| 16 | 6 | Hata analizi: en kötü 20 tabak, karışan sınıf çiftleri | İyileştirme için seçilen 3–4 teknik |
| 17 | 7–8 | İyileştirmeleri tek tek ekle (§7, Faz 5) | Ablation tablosu dolu |
| 18 | 8 | En iyi modeli seç, test kümesinde son kez ölç | Final satırları ve checkpoint |
| 19 | 9 | Demo: fotoğraf yükle → kcal (`app.py` iskeleti hazır) + kendi fotoğraflarımızla test (§7 Faz 4) | Çalışan demo, kısa bir video kaydı, `own_photos` satırları |
| 20 | 9 | Toparlama: figürlerin son hali, kod temizliği, README | `main` dalında baştan sona çalışan kod. **Kontrol noktası 3** |

**Durum (28 Ekim):** Adım 2, 3, 4 ✅ · Adım 1: Mac ✅, Kaggle ⏳ · Adım 5: sınıf listesi ✅ (`src/data.py`), indirme ⏳

**Sıradaki adım:** Kaggle kurulumu (§5.1), ardından Adım 5 ve 6.

---

## 5. Gün 1 — kurulum ayrıntıları (Adım 1–5)

### 5.1 Platform

**Seçim: Kaggle (eğitim) + Mac (geri kalan her şey).** İkimiz de aynı düzeni kullanıyoruz.

**Mac'te kurulum** (Python 3.12; [uv](https://docs.astral.sh/uv/) ile):

```bash
git clone https://github.com/traldo44/kalori-tahmini.git && cd kalori-tahmini
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download_nutrition5k.py      # ~1,4 GB, birkaç dakika
.venv/bin/python -m pytest -q                         # hepsi geçmeli
```

**Kaggle'da kurulum** (bir kez):

1. kaggle.com → Settings → **Phone verification** (GPU ve internet için şart)
2. GitHub'da yalnızca bu repoya erişimi olan bir token açın: Settings → Developer settings → **Fine-grained tokens** → Repository access: `kalori-tahmini` → Permissions: **Contents: Read and write**
3. Kaggle notebook'ta: Add-ons → **Secrets** → `GH_TOKEN` adıyla token'ı ekleyin
4. Notebook ayarları: Accelerator **GPU T4 x2** · Internet **On** · Environment **"Pin to original environment"** (paket sürümleri oturumlar arasında değişmesin)

Her notebook'un ilk hücresi:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["GH_TOKEN"] = token = UserSecretsClient().get_secret("GH_TOKEN")
!git clone -q https://{token}@github.com/traldo44/kalori-tahmini.git
%cd kalori-tahmini
!git config user.name "Ad Soyad" && git config user.email "eposta@ornek.com"
!python scripts/download_nutrition5k.py --out /kaggle/working/data/nutrition5k
os.environ["KALORI_DATA"] = "/kaggle/working/data"
os.environ["KALORI_OWNER"] = "ad_soyad"          # results.csv'deki owner sütunu
```

Kaggle'da torch/torchvision/timm zaten kurulu; `pip install -r requirements.txt` **yapmayın** (torch'u yeniden kurar). Eksik paket çıkarsa yalnızca onu kurun. Nutrition5k'yı her oturumda indirmek birkaç dakika sürer. İsterseniz ilk indirmeden sonra notebook çıktısını "New Dataset" olarak kaydedip `KALORI_DATA`'yı `/kaggle/input/...`'a çevirebilirsiniz.

### 5.2 Repo yapısı

```
kalori-tahmini/
├── README.md              ← bu dosya
├── data/                  ← git'e EKLENMEZ (.gitignore)
│   ├── nutrition5k/
│   └── food101/
├── notebooks/
│   ├── 01_eda_nutrition5k.ipynb
│   ├── 02_eda_food101.ipynb
│   └── ...
├── src/
│   ├── config.py          ← ORTAK sabitler (tohum, boyut, yollar, cihaz)       ✅
│   ├── data.py            ← veri yükleyiciler, val ayrımı, BAD_DISHES, Food-101 sınıf listesi ✅
│   ├── transforms.py      ← kalori / sınıflandırma dönüşümleri (§6.1)          ✅
│   ├── metrics.py         ← ORTAK: tüm metrikler + güven aralıkları             ✅
│   ├── results.py         ← results.csv'ye satır ekleme, tahmin kaydetme        ✅
│   ├── compare.py         ← iki modeli eşleştirilmiş bootstrap ile karşılaştır  ✅
│   ├── models.py          ← omurga + başlıklar, checkpoint biçimi, tek foto tahmini ✅ (temel)
│   ├── eval_own_photos.py ← kendi fotoğraflarımızla dış test (§7, Faz 4)       ✅
│   ├── train_cls.py       ← sınıflandırma eğitimi                               ⬜
│   ├── train_reg.py       ← kalori eğitimi                                      ⬜
│   └── vlm_eval.py        ← VLM sıfır-atış referansı                            ⬜
├── scripts/
│   ├── download_nutrition5k.py  ← gsutil'siz indirme (Mac ve Kaggle)           ✅
│   └── share_checkpoint.py      ← checkpoint'leri GitHub Release ile paylaş (§6.4) ✅
├── tests/                 ← python -m pytest -q
├── app.py                 ← Gradio demosu (Adım 19)                            ✅ (iskelet)
├── own_photos/            ← kendi fotoğraflarımız + labels.csv (§7, Faz 4)
├── results/
│   ├── results.csv        ← ORTAK: her deneyin sonucu buraya
│   ├── preds/             ← her modelin test tahminleri (git'te)
│   └── figures/
└── checkpoints/           ← git'e EKLENMEZ
```

### 5.3 Nutrition5k indirme (yalnızca gerekli kısım)

Tam arşiv **181,4 GB**; videolar ve derinlik dahil. Bize yalnızca üstten RGB görüntüler, metadata ve bölme dosyaları lazım. Veri CC BY 4.0 lisanslı. Bucket herkese açık, `gsutil` gerekmiyor:

```bash
python scripts/download_nutrition5k.py     # metadata + bölmeler + 3490 rgb.png (~1,4 GB); yarıda kalırsa tekrar çalıştırın
```

Yapı: `data/nutrition5k/realsense_overhead/dish_XXXXXXXXXX/rgb.png` (640×480), `metadata/dish_metadata_cafe{1,2}.csv`, `dish_ids/splits/`.

**Kullanılabilir tabak sayıları** (28 Ekim'de doğrulandı, `load_n5k_split` bunu otomatik yapar):

| Bölme | Listedeki kimlik | Üstten görüntüsü olan | Hatalı (atıldı) | Kullanılan |
| --- | --- | --- | --- | --- |
| train (`rgb_train_ids.txt`) | 4059 | 2755 | 1 | **2754** → 2476 train + 278 val |
| test (`rgb_test_ids.txt`) | 709 | 507 | 0 | **507** |

Üstten görüntüsü olan tabaklar tam olarak `depth_{train,test}_ids.txt` listeleri. Literatürdeki "üstten RGB" sonuçları da bu 507 tabaklık test kümesiyle karşılaştırılmalı. Bölmeler yalnızca Cafe 1'i kapsıyor (Cafe 2 kullanılmıyor, bkz. `dish_ids/README`).

**Metadata:** Her satır bir tabak: `dish_id, total_calories, total_mass, total_fat, total_carb, total_protein`, ardından malzeme başına 7'li gruplar. Satır uzunlukları farklı olduğu için `pandas.read_csv` hata verir; `src/data.py` → `read_dish_metadata` bunu `csv` modülüyle okur.

**Görüntülerden öğrendiklerimiz:** Tabaklar her zaman ortada değil, bazıları kenara taşıyor (o yüzden kırpma yok, §6.1). Bazı görüntüler hatalı: ör. `dish_1559172356`'da yemek değil telefon var ama etiketi 313 kcal. EDA'da bulunanlar `BAD_DISHES`'a eklenir.

### 5.4 Food-101 (alt küme)

```python
from torchvision.datasets import Food101
train = Food101(root="data/food101", split="train", download=True)  # ~5 GB, bir kez indirilir
test  = Food101(root="data/food101", split="test",  download=True)
```

Alt küme **25 sınıf** olarak sabitlendi: `src/data.py` → `FOOD101_CLASSES`. Liste sonradan değişirse önceki sınıflandırma sonuçları karşılaştırılamaz hale gelir; ilk sonuçtan önce itirazınız varsa şimdi değiştirin. Listede bilerek karışabilir çiftler var (bolognese/carbonara, caesar/greek salad, pancakes/waffles); karışıklık matrisi ve hata analizi için iyi malzeme olur. Sınıf başına 750 eğitim ve 250 test görüntüsü → 18.750 eğitim / 6.250 test. Okuyucu: `Food101Subset("train" | "test", transform)`.

### 5.5 Gün 1 bitiş kriterleri

- [x] Platform seçildi ve §3'e yazıldı (Kaggle hesabı/kurulumu ⏳)
- [x] Repo açıldı, bu dosya `README.md` olarak eklendi
- [x] `src/metrics.py` yazıldı (§6.2) ve tüm deneyler onu kullanıyor
- [x] Nutrition5k RGB görüntüleri, metadata ve bölmeler indirildi; kullanılabilir train/test tabak sayıları not edildi
- [ ] Food-101 alt kümesi hazır (sınıf listesi ✅, indirme ⏳)

---

## 6. Ortak kurallar (sonuçların karşılaştırılabilir olması için)

### 6.1 Sabitler

| Ayar | Değer |
| --- | --- |
| Rastgelelik tohumu | `42` (torch, numpy, random) |
| Görüntü boyutu | 224 × 224 (iyileştirme fazında 384 denenebilir). Nutrition5k: 640×480'in **tamamı** 224×224'e küçültülür, kırpılmaz |
| Normalizasyon | ImageNet ortalama/std |
| Doğrulama kümesi | Train'in ~%10'u, **zaman gruplarıyla** ayrılır: `make_val_split()`. Aralarında 10 dakikadan az olan tabaklar aynı gruptadır ve aynı kümeye düşer. Sonuç: 2476 train / 278 val. Gerekçe: tabaklar malzeme malzeme eklenerek art arda çekilmiş; ardışık tabak çiftlerinin %57'sinde malzeme listeleri iç içe. Rastgele ayırma neredeyse aynı tabağı hem train'e hem val'e koyar |
| Veri artırma, **kalori** | Yalnızca yatay/dikey çevirme, 90° döndürme, renk/parlaklık: `calorie_transform()`. **RandomResizedCrop, zoom, ölçekli kırpma YASAK**: model porsiyonu tabağın kapladığı alandan öğrenir; ölçeği değiştirmek etiketle çelişir. `tests/test_transforms.py` bunu kontrol eder |
| Veri artırma, **sınıflandırma** | RandomResizedCrop + çevirme + RandAugment serbest: `classification_transform()` |
| Test kümesi | Nutrition5k resmi test bölmesi. Hiperparametre seçimi **asla** test üzerinde yapılmaz |
| Başlangıç eğitim ayarı | AdamW, lr 1e-4 (omurga) / 1e-3 (başlık), cosine scheduler, 20–30 epoch, batch 32, erken durdurma, AMP |
| Kayıp | Sınıflandırma: cross-entropy + label smoothing 0,1. Regresyon: L1 veya Huber |

### 6.2 Metrikler — `src/metrics.py`

> **PMAE ile MAPE aynı şey değil.** Nutrition5k ve literatürdeki "yüzde hata" PMAE'dir: MAE'nin gerçek değerlerin ortalamasına oranı. MAPE ise her örneğin yüzde hatasının ortalamasıdır. Literatürle karşılaştırırken **PMAE** kullanın.

Dosya yazıldı ve test edildi. İçindekiler:

| Fonksiyon | Ne yapar |
| --- | --- |
| `regression_metrics(y, p, ci=False)` | MAE, PMAE, MAPE, RMSE, R², ±100 kcal içinde kalma oranı. `ci=True` → `MAE_ci_low/high` da ekler |
| `classification_metrics(y, p, y_score=None, ci=False)` | accuracy, macro precision/recall/F1, top-5. `ci=True` → `f1_ci_low/high` |
| `calorie_bins(train_kcal)` / `to_bins(kcal, edges)` | Tablo 3 için düşük/orta/yüksek aralıkları (sınırlar TRAIN'den) |
| `bootstrap_ci(y, p, metrik)` | Herhangi bir metriğin %95 güven aralığı |
| `paired_bootstrap(y, pred_a, pred_b)` | İki modelin farkı anlamlı mı? Aralık 0'ı içeriyorsa "A daha iyi" **denmez** |

**Güven aralığı kuralı:** Test sonuçları her zaman `ci=True` ile hesaplanır ve tablolarda "92,3 [86,0–98,9] kcal" gibi yazılır. İki modeli karşılaştırırken:

```bash
python -m src.compare results/preds/resnet50_reg.csv results/preds/vit_b16_reg.csv
# Fark (A − B) = -6.2 kcal, %95 güven aralığı [-11.0, -1.5] → Fark anlamlı: A daha iyi.
```

### 6.3 Sonuç kaydı — `results/results.csv`

Her deney bitince **bir satır** ekleyin. Tablolar raporda buradan üretilecek. Elle yazmayın, `src/results.py`'yi kullanın:

```python
from src.metrics import regression_metrics
from src.results import log_result, save_preds

m = regression_metrics(y_true, y_pred, ci=True)
log_result("calorie", "resnet50_reg", "lr1e-4_bs32_224", "test", m, notes="augmentation yok")
save_preds("resnet50_reg", dish_ids, y_true, y_pred)        # → results/preds/resnet50_reg.csv
```

Sütunlar: `date, owner, task, model, config, split, accuracy, precision, recall, f1, f1_ci_low, f1_ci_high, top5, MAE, MAE_ci_low, MAE_ci_high, PMAE, MAPE, RMSE, R2, within_100kcal, notes`

- `task`: `cls` (Food-101), `calorie` (Nutrition5k regresyon), `calorie_bins` (Tablo 3), `ablation`
- `split`: `val`, `test`, `own_photos` (kendi fotoğraflarımız, §7 Faz 4)
- Tahmin dosyaları (`id, y_true, y_pred`) **git'e girer**. Grafikler, hata analizi ve model karşılaştırmaları bunlardan üretilir, checkpoint gerekmez.

### 6.4 Git

- Repo: [traldo44/kalori-tahmini](https://github.com/traldo44/kalori-tahmini) (private); iki kişi de collaborator. Kaggle notebook'ları repo'yu token ile klonlar (§5.1).
- `main` dalı her zaman çalışır durumda olmalı. Her adım kendi dalında yapılır (ör. `adim-09-klasik-ml`) ve bittiğinde `main`'e birleştirilir.
- Çalışmayı bırakmadan önce yarım işi de commit'leyip push edin; commit mesajında hangi adımda kaldığınızı yazın. İşe başlarken önce `git pull`.
- Birleştirmeden önce `python -m pytest -q` geçmeli.
- `results/results.csv` için `.gitattributes`'ta `merge=union` var: iki kişi aynı anda satır eklerse git çakışma çıkarmaz, iki satırı da tutar.
- `data/`, `checkpoints/` ve `*.pt` `.gitignore`'da.

**Checkpoint paylaşımı.** Model dosyaları büyük (ResNet-50 ~95 MB, ViT-B ~330 MB), git'e girmez. Analiz için zaten gerekmez, tahmin dosyaları yeter. Yalnızca demoda ve Grad-CAM'de kullanılacak modeller repo'nun `checkpoints` Release'ine yüklenir:

```bash
export GH_TOKEN=...                    # Kaggle'da ilk hücre bunu zaten yapıyor
python scripts/share_checkpoint.py upload checkpoints/resnet50_reg.pt
python scripts/share_checkpoint.py list
python scripts/share_checkpoint.py download resnet50_reg.pt     # → checkpoints/
```

Checkpoint biçimi `src/models.py` → `save_checkpoint()`: omurga adı, çıktılar, hedef dönüşümü (`log1p`) ve `test_MAE`. `test_MAE` varsa demo bunu kullanıcıya "tipik hata" olarak gösterir.

---

## 7. Fazların ayrıntısı (ne üretilecek?)

### Faz 2 — EDA (Gün 1–2)

Her adımın çıktısı bir grafik veya tablo olmalı; bunlar doğrudan rapora girecek.

1. Görüntü sayısı, çözünürlük, bozuk ve kopya dosya kontrolü. **Yemek olmayan görüntüler** de var (ör. telefon). Hızlıca göz gezdirmek için tüm görüntülerin küçük resim ızgarasını çıkarın; bulduklarınızı `BAD_DISHES`'a ekleyin
2. Food-101: sınıf başına görüntü sayısı
3. Nutrition5k: kalori histogramı, ortalama, medyan, çarpıklık, aykırı değerler. **Sağa çarpıksa hedefe `log1p` uygulayın.**
4. Kalori–kütle saçılım grafiği, makrolar arası korelasyon ısı haritası, Atwater kontrolü (kalori ≈ 4·protein + 4·karbonhidrat + 9·yağ). Bu kontrolden çok sapan tabaklar etiket hatası adayıdır.
5. Her kalori aralığından örnek görüntü ızgarası
6. **Veri kartı:** bölme başına tabak sayısı, kalori ortalaması ± std

### Faz 3 — Modeller (Gün 2–5)

| Katman | Sınıflandırma | Kalori regresyonu |
| --- | --- | --- |
| 0 — Naif | En sık sınıf | Train ortalaması |
| 1 — Klasik ML | HOG + renk histogramı → SVM, Random Forest, k-NN | Donmuş CNN gömmeleri → Ridge, Random Forest, XGBoost |
| 2 — Transfer CNN | ResNet-50, EfficientNet-B0 | ResNet-50 / EfficientNet + çok görevli başlık (kalori, kütle, makrolar) |
| 3 — Güncel | ConvNeXt-T, ViT-B/16 | ConvNeXt / ViT regresyon |
| Referans | — | VLM sıfır-atış |

**Kalori başlığı:** kaloriyi doğrudan değil, `kütle × kalori yoğunluğu` olarak tahmin etmeyi deneyin. Nutrition5k'da kütle etiketi var; çok görevli öğrenme literatürde hatayı düşürüyor.

**VLM referansı (`vlm_eval.py`):**
- Bir veya iki güncel VLM, Nutrition5k test kümesinde (maliyet sorunsa rastgele 300 tabak, tohum 42)
- Sabit prompt: önce görünen her yiyeceğin gram tahmini, sonra toplam kcal; JSON çıktı; sıcaklık 0
- Ayrıştırılamayan yanıt oranını ve toplam maliyeti kaydedin
- API anahtarlarını **asla** repoya koymayın; ortam değişkeni kullanın

### Faz 4 — Değerlendirme (Gün 5–6)

- **Tablo 1** (Food-101 alt kümesi): accuracy, precision, recall, F1 (macro), top-5, parametre sayısı, çıkarım süresi
- **Tablo 2** (Nutrition5k test): MAE, PMAE, RMSE, R², ±100 kcal içinde kalma oranı
- **Tablo 3** (kalori aralığı): aynı regresyon tahminleri düşük / orta / yüksek aralıklarına çevrilerek accuracy, precision, recall ve F1
- **Figürler:** karışıklık matrisi, tahmin–gerçek saçılım grafiği, hata histogramı, sınıf veya aralık başına MAE, Grad-CAM örnekleri
- **Güven aralıkları:** Tablo 1'de F1, Tablo 2'de MAE köşeli parantez içinde (§6.2)

#### Kendi fotoğraflarımızla test (dış test, demo öncesi)

Nutrition5k tek bir kafeteryada, sabit üst kamerayla çekildi. Telefon fotoğraflarında model kötüleşecek; bunu sunumda sürpriz olarak değil, **ölçülmüş bir sınırlılık** olarak göstereceğiz.

1. **15–20 tabak** hazırlayın. Gerçek kaloriyi şöyle bulun: paketli ürünlerde etiketten; diğerlerinde her yiyeceği mutfak tartısında tartın ve 100 g başına kaloriyi bir besin tablosundan ([TürKomp](https://turkomp.tarimorman.gov.tr) veya USDA FoodData Central) alın. Kaynağı `source` sütununa yazın.
2. Her tabağı **tam üstten** çekin: tabak fotoğraf genişliğinin yaklaşık yarısı ile üçte ikisi arasını kaplasın, tek tabak, düz zemin. Bunlar `angle=ust`.
3. Aynı tabaklardan 5–6 tanesini **yandan** da çekin (`angle=yan`). Bunlar kurala uymayınca hatanın ne kadar büyüdüğünü göstermek için.
4. Fotoğrafları küçültüp `own_photos/`'a koyun: `sips -Z 1280 *.jpg` (Mac). Ardından `labels.csv`'yi doldurun:
   ```
   file,calories,angle,source,notes
   tabak01_ust.jpg,520,ust,tartı+TürKomp,pilav 180 g + tavuk 120 g
   tabak01_yan.jpg,520,yan,tartı+TürKomp,
   ```
5. En iyi model hazır olunca: `python -m src.eval_own_photos --ckpt checkpoints/best.pt --model <ad>`. Sonuç `results.csv`'ye `own_photos`, `own_photos_ust` ve `own_photos_yan` satırları olarak yazılır.

Raporda: Nutrition5k test MAE'si ile kendi fotoğraflarımızdaki MAE yan yana; üstten ile yandan arasındaki fark "sınırlılıklar" bölümüne. Demo (`app.py`) kullanıcıya nasıl çekmesi gerektiğini söyler, modelin gördüğü kırpılmış görüntüyü ve modelin test kümesindeki tipik hatasını gösterir.

### Faz 5 — İyileştirme (Gün 6–8)

Önce hata analizi yapılır, teknik ona göre seçilir. Her teknik **tek tek** eklenir ve etkisi ablation tablosuna yazılır.

| Hata analizinde görülen | Denenecek teknik |
| --- | --- |
| Train–val farkı büyük | RandAugment, dropout, weight decay, erken durdurma |
| Sınıflar karışıyor | Daha büyük çözünürlük, MixUp/CutMix, label smoothing, daha güçlü omurga |
| Uç kalorilerde büyük hata | `log1p` hedef, Huber kaybı, aykırı etiket temizliği |
| Porsiyon hatası baskın | Kütle başlığı, tahmini derinliği (Depth Anything V2) 4. kanal olarak ekleme, segmentasyon maskesi |
| Makrolar ile kalori tutarsız | Atwater tutarlılık kaybı |
| Son birkaç puan | TTA, farklı omurgaların ensemble'ı |

**Ablation tablosu** (`task=ablation`):

| Deney | MAE | PMAE | F1 (sınıf) |
| --- | --- | --- | --- |
| Baseline (ResNet-50, augmentation yok) | | | |
| + veri artırma (kalori: yalnızca `calorie_transform`, §6.1) | | | |
| + çok görevli başlık (kütle) | | | |
| + `log1p` hedef + Huber | | | |
| + tahmini derinlik kanalı | | | |
| + ensemble / TTA | | | |

**Zaman sıkışırsa:** yalnızca veri artırma ve kütle başlıklı çok görevli öğrenmeyi deneyin. En büyük kazancı bu ikisi getirir.

---

## 8. Kontrol noktaları ve riskler

| Ne zaman | Kontrol | Kaçarsa |
| --- | --- | --- |
| Gün 1 sonu (Adım 5) | Veri indirildi, `metrics.py` hazır | Food-101 alt kümesini 10 sınıfa indirin |
| **Gün 3 sonu (Adım 9, kontrol noktası 1)** | Baseline sonuçları Tablo 1 ve 2'de | İyileştirme fazı 2 tekniğe iner (yukarıda) |
| Gün 6 sonu (Adım 15, kontrol noktası 2) | En iyi model 80,7 kcal'in (VLM) altında | Çok görevli başlık ve `log1p` hedefe öncelik verin |
| Gün 9 (Adım 20, kontrol noktası 3) | Tüm tablolar dolu, kod `main`'de çalışıyor | — |

| Risk | Önlem |
| --- | --- |
| Nutrition5k çok büyük | Sadece §5.3'teki komutlarla indirin; derinlik ve video yok |
| GPU süresi yetmez | EfficientNet-B0, 224 px, AMP, her epoch sonunda checkpoint |
| Veri sızıntısı (aynı tabak train ve test'te) | Bölmeyi tabak kimliğiyle yapın, resmi bölmeyi değiştirmeyin |
| Literatürle adil olmayan karşılaştırma | Aynı resmi test bölmesi; PMAE; girdi türünü (RGB) her tabloda yazın |
| Devir sırasında ayarlar değişir, sonuçlar karşılaştırılamaz | Aynı tohum, aynı `metrics.py`, aynı `results.csv` formatı; her değişiklik devir günlüğüne yazılır |
| Doğrulama kümesinde sızıntı (art arda çekilmiş benzer tabaklar) | Zaman gruplu ayrım: `make_val_split()` (§6.1) |
| Ölçek değiştiren artırma porsiyon bilgisini bozar | Kalori için yalnızca `calorie_transform()`; testle korunuyor |
| Küçük farkları "iyileşme" sanmak | Güven aralığı + `src.compare` (§6.2) |
| Demo telefon fotoğraflarında kötü çalışır | Çekim talimatı demoda; kendi fotoğraflarımızla ölçülüp raporlanır (§7 Faz 4) |
| Kaggle oturumu kapanınca dosyalar silinir | Kod ve tahminler git'e push, önemli checkpoint'ler Release'e (§6.4) |

---

## 9. Sonraki aşama (geliştirme bittikten sonra)

Bu dosyanın kapsamı dışında; bilgi için:

- **Rapor:** Özet, Giriş (araştırma sorusu), Metodlar, Sonuçlar, Karşılaştırma, Referanslar. Tablolar `results.csv`'den, figürler `results/figures/` klasöründen gelecek.
- **Sunum:** yaklaşık 15 slayt; literatüre 2 slayt ayrılacak. Gün 8–9'da kaydedilecek demo videosu sunumda kullanılacak.
- **Karşılaştırma tablosu:** §2'deki hedef tablosunun sonuna "Bu çalışma" satırı eklenecek.

---

## 10. Devir günlüğü

Çalışmayı bırakan kişi en üste bir satır ekler; devralan kişi önce bu tabloyu ve §3'ü okur.

| Tarih | Kim | Son tamamlanan adım | Yarım kalan iş / sıradaki adım | Notlar (değişen ayarlar, sorunlar) |
| --- | --- | --- | --- | --- |
| 2026-10-28 | traldo | Adım 2, 3, 4 (+ Adım 1 Mac tarafı, Adım 5 sınıf listesi) | Kaggle kurulumu (§5.1) → Adım 5: Food-101 indir → Adım 6: Nutrition5k EDA | **Yeni:** zaman gruplu val ayrımı, kalori için ölçeksiz artırma, bootstrap güven aralıkları, `src.compare`, kendi fotoğraf testi, demo iskeleti, checkpoint paylaşımı. Food-101 25 sınıf. `BAD_DISHES`'ta 1 tabak. Kullanılabilir veri: 2476 train / 278 val / 507 test. "Açık konular" tablosuna bakın (§3) |
| 2026-10-24 | — | Planlama ve literatür bitti | Adım 1: platform seçimi | Kod tarafında henüz bir şey yok |

---

## Kaynaklar

- Nutrition5k veri seti ve indirme: [google-research-datasets/Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) · makale: [Thames vd., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.pdf)
- Food-101: [ETH Zürich veri sayfası](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101)
- Açık ön eğitimle ResNet/ViT sonuçları: [Andrade vd., 2025](https://arxiv.org/pdf/2508.03996v2)
- VLM sıfır-atış benchmark'ı: [Sterling vd., bioRxiv 2026](https://www.biorxiv.org/content/10.64898/2026.07.26.740845.full.pdf)
- Tahmini derinlik: [DPF-Nutrition, Han vd., 2023](https://arxiv.org/abs/2310.11702) · [Depth Anything V2](https://depth-anything-v2.github.io/)
- Segmentasyon + regresyon: [Zhao vd., Frontiers in Nutrition 2024](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1469878/full)
- 39 çalışmanın tamamı: yol haritası belgesindeki "Literatür matrisi" sekmesi
