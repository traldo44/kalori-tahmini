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

### EDA'dan çıkan bulgular (Adım 6, `notebooks/01_eda_nutrition5k.ipynb`)

1. **3 hatalı tabak** eğitimden çıkarıldı (`BAD_DISHES`). Test kümesine dokunulmadı.
2. **Hedef dönüşümü:** Baseline **ham kcal + L1 kaybı** (`target="raw"`). Ham dağılımın çarpıklığının çoğu tek bir hatalı tabaktan geliyordu (2,68 → 1,18). `log1p` dağılımı ters yöne bozuyor (−0,80; 10 kcal altı tabaklar uzun bir sol kuyruk yapıyor) ve MAE yerine göreli hatayı optimize ediyor. `log1p` + Huber yalnızca ablation'da denenecek.
3. **Balık etiketleri tutarsız:** Malzeme tablosunda "fish" 0,90 kcal/g, ama makrolarına göre ~1,9 olmalı. Balıklı 183 tabağın 149'unda kalori, makrolardan hesaplanandan (Atwater) 50 kcal'den fazla düşük. Çıkarılmıyor; hata analizinde balıklı/balıksız MAE **ayrı** raporlanacak.
4. **Resmi test kümesinde de sızıntı var:** Test tabaklarının %22'sinin train'de malzeme eklenmiş/çıkarılmış bir "kardeşi" var. Literatür aynı ayrımı kullandığı için karşılaştırma adil, ama **bizim val MAE'miz büyük ihtimalle test MAE'sinden kötü çıkacak**. Bu bir hata değil; raporda sınırlılık olarak yazılacak.
5. Kalori en çok **yağ (0,88)** ve **protein (0,85)** ile ilişkili, kütleyle 0,74. Bu, çok görevli başlığın (Adım 12) gerekçesi.
6. **Tablo 3 sınırları:** düşük < 116,1 ≤ orta < 313,7 ≤ yüksek kcal. `CALORIE_BIN_EDGES` olarak sabitlendi; bir test bunların değişmediğini kontrol ediyor.

### Food-101 EDA bulguları (Adım 7, `notebooks/02_eda_food101.ipynb`)

1. **Sızıntı:** 9 test görüntüsünün train/val'de birebir ya da neredeyse aynı kopyası vardı. Train/val tarafı `FOOD101_EXCLUDE` ile çıkarıldı; test'e dokunulmadı.
2. **Çelişkili etiket:** Aynı fotoğraf hem `omelette` hem `pancakes` etiketliydi; ikisi de çıkarıldı.
3. **Etiket gürültüsü:** Food-101'in train'i bilerek temizlenmemiş. 10 komşusunun hiçbiri aynı etiketi taşımayan görüntü oranı train'de %8,0, temiz test'te %5,5. Hangisinin gerçekten yanlış olduğu kesin olmadığı için çıkarılmıyor; label smoothing (0,1) koruma sağlar.
4. **Naif baseline (Adım 8):** En sık sınıf → doğruluk %4 (1/25), macro-F1 0,003. Kalori: ortalama tahmin → test MAE **168,4** [158,0–180,1] kcal, PMAE %65,9.

### Klasik ML bulguları (Adım 9, `python -m src.classic_ml`) — Kontrol noktası 1

Hiperparametreler yalnızca val'de seçildi; test bir kez ölçüldü. Köşeli parantez = %95 bootstrap güven aralığı.

| Tablo 1 · Food-101 (25 sınıf, test) | Doğruluk | Macro-F1 | Top-5 |
| --- | --- | --- | --- |
| Naif: en sık sınıf | %4,0 | 0,003 | — |
| HOG + renk → k-NN (k=5) | %9,9 | 0,080 [0,073–0,087] | %32,3 |
| HOG + renk → Random Forest | %27,0 | 0,250 [0,239–0,260] | %61,1 |
| HOG + renk → SVM (RBF, C=10) | %29,8 | 0,293 [0,283–0,304] | %65,3 |
| **Donmuş ResNet-50 → lojistik regresyon** | **%83,7** | **0,837** [0,828–0,846] | %97,9 |

| Tablo 2 · Nutrition5k (yalnızca RGB, test) | MAE (kcal) | PMAE | R² | ±100 kcal içinde | Tablo 3 F1 |
| --- | --- | --- | --- | --- | --- |
| Naif: train ortalaması | 168,4 [158,0–180,1] | %65,9 | 0,00 | %31,6 | 0,159 |
| HOG + renk → XGBoost | 103,9 [96,3–112,1] | %40,7 | 0,56 | %61,7 | 0,635 |
| Donmuş ResNet-50 → Random Forest | 88,9 [81,9–96,9] | %34,8 | 0,64 | %69,0 | 0,710 |
| Donmuş ResNet-50 → Ridge | 78,2 [72,0–85,6] | %30,6 | 0,71 | %74,4 | 0,743 |
| **Donmuş ResNet-50 → XGBoost** | **77,4** [70,9–84,8] | **%30,3** | 0,71 | %73,2 | 0,747 |

1. **Derin özellikler, elle tasarlanmış özellikleri ezici farkla geçiyor:** sınıflandırmada F1 0,84'e karşı 0,29; kaloride 77'ye karşı 104 kcal.
2. **Hiç ince ayar yapılmamış ResNet-50 + XGBoost bile 77,4 kcal'e iniyor.** Bu, en iyi sıfır-atış VLM'in (80,7) seviyesinde. Ancak güven aralığı 80,7'yi içerdiği için "geçtik" denemez. Literatürdeki "ImageNet ResNet-50, 117,4 kcal" satırının da çok altında, fakat o çalışmanın test kümesi belirsiz (§3 Açık konular).
3. **XGBoost ile Ridge arasındaki fark anlamlı değil:** `python -m src.compare` → −0,8 kcal [−4,2; +2,7].
4. **Adım 10–13 için çıta:** İnce ayarlı CNN'lerin **77 kcal**'i ve **F1 0,84**'ü geçmesi gerekiyor; geçip geçmediği `src.compare` ile kontrol edilmeli. Nutrition5k küçük (2.472 train) olduğu için uçtan uca ince ayar aşırı öğrenmeye açık; omurganın ilk katmanlarını dondurmak ve güçlü düzenlileştirme denenmeli.

### Transfer CNN / Transformer bulguları (Adım 10–13, `notebooks/10_egitim_kaggle.ipynb` + `11_egitim_kaggle_adim12-13.ipynb`, Kaggle T4)

Uçtan uca ince ayar, §6.1'deki ortak ayarlar (AdamW, omurga 1e-4 / başlık 1e-3, ısınma + cosine, val'e göre erken durdurma). Süre ölçümü T4'te, yalnızca model. Köşeli parantez: %95 bootstrap güven aralığı. Anlamlılık testleri `python -m src.compare` ile (eşleştirilmiş bootstrap).

| Tablo 1 · Food-101 (test) | Doğruluk | Macro-F1 | Top-5 | Parametre | ms/görüntü |
| --- | --- | --- | --- | --- | --- |
| ResNet-50 (ince ayar) | %91,3 | 0,912 [0,905–0,919] | %99,0 | 23,6 M | 1,31 |
| EfficientNet-B0 (ince ayar) | %91,7 | 0,917 [0,909–0,924] | %99,0 | 4,0 M | 0,70 |
| ConvNeXt-T (ince ayar) | %94,1 | 0,941 [0,935–0,947] | %99,4 | 27,8 M | 1,49 |
| **ViT-B/16 (ince ayar)** | **%95,1** | **0,951** [0,945–0,956] | %99,3 | 85,8 M | 2,40 |

| Tablo 2 · Nutrition5k (test) | MAE (kcal) | PMAE | R² | ±100 kcal içinde | Tablo 3 F1 | MAE balıksız / balıklı |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-50, timm başlatması (Adım 11, ilk run) | 59,7 [53,6–66,3] | %23,4 | 0,81 | %80,7 | 0,806 | 58,1 / 92,2 |
| ResNet-50, düzeltilmiş başlatma (erken durdu, 14/30) | 68,7 [62,0–75,5] | %26,9 | 0,76 | %78,3 | 0,790 | — |
| ResNet-50 çok görevli: kalori + kütle + makrolar (Adım 12) | 58,0 [52,5–63,9] | %22,7 | 0,83 | %81,5 | 0,799 | 56,2 / 96,2 |
| EfficientNet-B0, hatalı başlatma (ilk run) | 104,4 [97,0–112,4] | %40,9 | 0,58 | %57,6 | 0,600 | — |
| EfficientNet-B0, düzeltilmiş başlatma | 52,8 [47,9–58,3] | %20,7 | 0,86 | %85,0 | 0,817 | 51,2 / 86,6 |
| ViT-B/16 (Adım 13) | 47,3 [42,2–52,8] | %18,5 | 0,87 | %86,6 | 0,859 | 45,8 / 79,2 |
| **ConvNeXt-T (Adım 13)** | **41,4** [37,3–46,4] | **%16,2** | **0,89** | **%89,0** | **0,861** | 39,8 / 75,1 |

Balıklı tabak = malzeme listesinde `fish` geçen tabak; test kümesinde 23 tane (484 balıksız). Eski kalori sonuçları `results.csv`'de `*_timminit` adıyla duruyor.

**Adım 10–11 bulguları (ilk Kaggle run'ı):**

1. **İnce ayar iki görevde de klasik ML'yi anlamlı farkla geçiyor** (`src.compare`): sınıflandırmada F1 +0,079 [+0,070; +0,088] (EfficientNet-B0 ile donmuş ResNet-50 + LogReg karşılaştırması), kaloride −17,7 kcal [−24,1; −11,9] (ResNet-50 ile donmuş ResNet-50 + XGBoost karşılaştırması).
2. **Sınıflandırmada iki CNN arasındaki fark anlamlı değil:** −0,004 [−0,011; +0,003]. EfficientNet-B0'da 6 kat daha az parametre var ve 2 kat daha hızlı.
3. **Sınıflandırmada test, val'den yüksek** (0,91'e karşı 0,88). Val, gürültülü etiketli train'den ayrıldığı için zorlaşıyor; test elle temizlenmiş (Food-101 EDA). Model seçimi bundan etkilenmez, ama val skorları raporda "karamsar" diye not edilmeli.
4. **EfficientNet-B0'ın 104 kcal'i bir başlatma hatasıydı.** timm, EfficientNet'in son katmanını TF tarzı başlatıyor; tek çıktıda ağırlıklar ±1 oluyor. Bu yüzden ilk tahminler hedefin ~7 katı çıktı (ilk epoch val MAE'si 605 kcal) ve bozuk gradyanlar omurgayı da bozdu. Düzeltme: `init_regression_head`. Ağırlıklar 0, bias 1 olarak başlıyor, yani ilk tahmin train ortalaması. Düzeltmeyle EfficientNet-B0 **104,4 → 52,8 kcal** (aşağıda).

**Adım 12–13 bulguları (ikinci Kaggle run'ı, 29 Ekim):**

5. **Kaloride en iyi model ConvNeXt-T: 41,4 kcal, PMAE %16,2.** Fark ViT-B/16'ya göre −5,9 kcal [−8,9; −3,1], EfficientNet-B0'a göre −11,4 [−15,5; −7,7], klasik ML'nin en iyisi XGBoost'a göre −35,9 [−41,6; −30,6]; hepsi anlamlı. Her iki literatür hedefi (80,7 ve 70,6) artık bütün ince ayarlı modellerce geçildi; ConvNeXt-T, segmentasyon/derinlik kullanan çalışmaların 37–40 kcal aralığının hemen üstünde ve **yalnızca RGB** kullanıyor. Dikkat: test tabaklarının ~%22'sinin train'de "kardeşi" var (EDA); bu etki Adım 16'da ölçülmeden literatürle birebir karşılaştırma yapılmamalı.
6. **Sınıflandırmada en iyi model ViT-B/16: F1 0,951.** ConvNeXt-T'den +0,009 [+0,004; +0,015], EfficientNet-B0'dan +0,034 [+0,028; +0,041]; ikisi de anlamlı ama küçük. ViT'nin parametresi EfficientNet-B0'ın 21 katı, çıkarımı 3,4 kat yavaş. Demo için EfficientNet-B0 veya ConvNeXt-T daha mantıklı; son seçim Adım 18'de.
7. **ResNet-50 düzeltilmiş başlatmayla kötüleşti (59,7 → 68,7; +9,0 [+4,7; +13,8]) ve bunun sebebi başlatma değil, erken durdurma.** Yeni run 7. epoch'ta en iyi val'ini (63,6) gördü, 7 epoch iyileşme olmayınca 14. epoch'ta durdu; eski run 30 epoch'un tamamını koştu ve val'i 62,7'ydi. Val MAE'leri neredeyse aynı (63,6 ve 62,7) ama test 9 kcal farklı: **280 tabaklık val, model seçimi için gürültülü**, ve `--patience 7` cosine'ın ortasında kesebiliyor. Düzeltilmiş başlatmayla başlayan diğer üç kalori modeli (EfficientNet-B0, ConvNeXt-T, ViT) 24–29. epoch'ta en iyisini buldu ve sorun yaşamadı. Adım 17 için ders: kalori ablation'larında `--patience` ≥ 10 (veya sabit 30 epoch) ve mümkünse 2–3 tohumla tekrar.
8. **Çok görevli başlık (Adım 12) en az tek görevli kadar iyi; "daha iyi" demek için kanıt yetersiz.** ResNet-50 çok görevli 58,0 kcal: aynı run'daki tek görevli ResNet-50'den (68,7) anlamlı iyi (−10,7 [−14,9; −7,0]), ama eski tek görevli run'dan (59,7) farkı anlamlı değil (−1,6 [−5,4; +2,4]). Tek görevli run erken durduğu için karşılaştırma kirli (madde 7). Yan çıktılar (test MAE): kütle 41,4 g, yağ 4,3 g, karbonhidrat 6,5 g, protein 6,2 g. Temiz cevap için Adım 17'de ConvNeXt-T'ye çok görevli başlık eklenip aynı ayarlarla tek görevliyle karşılaştırılmalı; §3'teki araştırma sorusu önerisi buna bağlı.
9. **Balıklı tabaklar bütün modellerde 1,7–1,9 kat daha kötü** (ConvNeXt-T: balıksız 39,8, balıklı 75,1 kcal). EDA'daki bulguyla tutarlı: balık kalorisi veri setinde eksik kayıtlı, yani hata kısmen etiketten geliyor. Raporda Tablo 2'nin yanında ayrı sütun olarak verilmeli; Adım 16 hata analizinde "en kötü 20 tabak" listesinde balıklılar işaretlenmeli.
10. **Val–test ilişkisi beklendiği gibi.** Kaloride test, val'den iyi (ConvNeXt-T 46,0 → 41,4), kardeş-sızıntısı tahminiyle uyumlu; sınıflandırmada da test > val (ViT 0,930 → 0,951), madde 3'teki sebeple.
11. **Süre (T4, bir epoch):** kalori modelleri 27–30 sn (2472 görüntü); sınıflandırma EfficientNet-B0 52 sn, ResNet-50 73 sn, ConvNeXt-T 83 sn, ViT-B/16 145 sn (16 865 görüntü). Notebook 11 toplam ~2,5 saat sürdü (eğitim ~2,2 sa + değerlendirme).

### Değerlendirme bulguları (Adım 15, `python -m src.make_figures` + `python -m src.gradcam`)

Bütün figürler `results/preds/` ve `results/history/` dosyalarından üretilir; GPU ve checkpoint gerekmez (yalnızca Grad-CAM için `checkpoints/convnext_tiny_reg.pt`). Rapora yapıştırmaya hazır tablolar: `results/tablo1.md`, `tablo2.md` (precision/recall ve RMSE dahil), `tablo3_kalori_araligi.md`.

| Tablo 3 · Kalori aralığı (düşük < 116,1 ≤ orta < 313,7 ≤ yüksek kcal, test) | Doğruluk | Precision (macro) | Recall (macro) | Macro-F1 [%95 GA] |
| --- | --- | --- | --- | --- |
| Naif: train ortalaması | %31,4 | 0,104 | 0,333 | 0,159 [0,142–0,174] |
| Donmuş ResNet-50 → XGBoost | %74,2 | 0,771 | 0,742 | 0,747 [0,708–0,785] |
| ResNet-50 (timm başlatması) | %80,5 | 0,814 | 0,804 | 0,806 [0,768–0,839] |
| ResNet-50 çok görevli | %79,9 | 0,808 | 0,798 | 0,799 [0,763–0,832] |
| EfficientNet-B0 | %81,7 | 0,821 | 0,815 | 0,817 [0,781–0,848] |
| ViT-B/16 | %86,0 | 0,861 | 0,859 | 0,859 [0,827–0,889] |
| ConvNeXt-T | %86,2 | 0,864 | 0,861 | 0,861 [0,828–0,889] |

| Figür | Dosya (`results/figures/`) | Açıklama |
| --- | --- | --- |
| Karışıklık matrisi | `eval_karisiklik_matrisi.png` | ViT-B/16, Food-101 25 sınıf, test; satırlar gerçek sınıf, hücreler satır içi yüzde; yalnızca ≥ %2'lik karışmalar yazılı |
| Sınıf başına F1 | `eval_sinif_f1.png` | ViT-B/16 çubuk, ConvNeXt-T nokta; en zor sınıflar en altta (apple pie 0,89) |
| Tahmin–gerçek | `eval_tahmin_gercek.png` | ConvNeXt-T test tahminleri; gölgeli bant ±100 kcal; turuncu = 23 balıklı tabak |
| Hata histogramı | `eval_hata_histogrami.png` | İşaretli hata (tahmin − gerçek), ConvNeXt-T ve ViT-B/16; ortalama −12, medyan +1 kcal |
| Aralık başına MAE | `eval_aralik_mae.png` | Gerçek kalori aralığına göre test MAE, tüm kalori modelleri (`results/eval_aralik_mae.csv`) |
| Model karşılaştırma | `eval_model_karsilastirma.png` | Test MAE + %95 GA; literatür çizgileri 80,7 / 70,6 ve 37–40 bandı |
| Eğitim eğrileri | `eval_egitim_egrileri.png` | Val skoru – epoch; nokta = saklanan en iyi epoch |
| En kötü 20 tabak | `eval_en_kotu_tabaklar.png` | ConvNeXt-T, mutlak hata; liste `results/eval_en_kotu_20.csv` |
| Grad-CAM | `eval_gradcam.png` | ConvNeXt-T; en iyi 4 ve en kötü 4 test tabağı, kırmızı = tahmini yukarı çeken bölge |

1. **Hata kaloriyle birlikte büyüyor.** ConvNeXt-T'de aralık başına MAE: düşük 16, orta 32, yüksek 76 kcal (naif: 201 / 54 / 241). Modeller arasındaki fark en çok yüksek aralıkta açılıyor.
2. **En kötü 20 tabağın 18'i yüksek kalorili tabakların eksik tahmini** (gerçek 380–1013, tahmin 120–768 kcal). Ortak örüntü: yeşilliğin altına gizlenmiş 100–240 g tavuk/sığır/domuz (ör. 902 → 424 kcal, 205 g tavuk + brokoli). Tek fazla tahmin var (117 → 323). Grad-CAM bunu doğruluyor: kötü tahminlerde model yemeğin üstündeki yeşilliğe bakıyor, altındaki proteini görmüyor. Balık en kötü 20'nin yalnızca 3'ünde, yani ortalama balık hatasını yükseltiyor ama en büyük hataların sebebi değil.
3. **Kardeş tabaklar listede çift çift görünüyor:** üç çift (`1565811139/1565811061`, `1566414412/1566414342`, `1563389626/1563389600`) aynı tabağın art arda halleri. Bir tabak (`1558373159`, 1013 kcal) malzeme listesinde 142 g "deprecated" içeriyor → etiket kalitesi sorunu, raporda örnek olarak verilebilir.
4. **Sınıflandırmada hatalar beklenen çiftlerde toplanıyor:** apple pie → pancakes 7, caesar → greek salad 7, fried rice ↔ risotto 7+6, carbonara → bolognese 6, lasagna → omelette 6, pizza → lasagna 6 (`results/eval_karisan_ciftler.csv`). Apple pie en zayıf sınıf (recall %85) ve ilk 10 çiftin 4'ünde var; EDA'da bilerek bırakılan karışabilir çiftler tam da hatanın toplandığı yer.
5. **Tablo 3'te ConvNeXt-T ile ViT-B/16 ayırt edilemiyor** (0,861 ve 0,859, GA'lar üst üste); üç aralıklı sınıflandırma MAE'deki farkı görmeye yetmiyor.
6. **Adım 16–17 için seçilen teknikler (kanıta dayalı):** (a) yüksek kalori eksik tahmini → `log1p` hedef veya göreli kayıp ablation'ı (`train_reg.py --target log1p` hazır), kütle-farkındalı başlık (çok görevli ConvNeXt-T); (b) kardeş sızıntısı → test tabaklarını `time_groups` ile gruplayıp "dizinin ilk tabağı" ve "sonrakiler" için MAE'yi ayrı raporlamak; (c) balık MAE'si ayrı sütun olarak (yapıldı); (d) sınıflandırmada apple pie / tatlı karışmaları için hedefli augmentasyon veya daha yüksek çözünürlük (ViT 384) ablation'ı.

### Hedeflerimiz (Nutrition5k test, yalnızca RGB)

| Referans | Kalori MAE (kcal) | PMAE (%) | Bizim için anlamı |
| --- | --- | --- | --- |
| Ortalama tahmin baseline'ı | 150,8 | 60,2 | Mutlaka geçilmeli. **Bizim ölçümümüz: 168,4 / %65,9** (Açık konular'a bakın) |
| ResNet-50, ImageNet ön eğitimli (Andrade 2025) | 117,4 | 46,6 | İlk derin model bu civarda çıkabilir |
| En iyi sıfır-atış VLM (Gemini 3.0 Flash) | 80,7 | — | **1. hedef** ✅ geçildi (tüm ince ayarlı modeller) |
| Orijinal Nutrition5k RGB baseline'ı (JFT ön eğitimli) | 70,6 | 26,1 | **2. hedef** ✅ geçildi |
| Segmentasyon veya tahmini derinlikle güncel çalışmalar | 37–40 | 14,7–15,7 | İyileştirme fazında yaklaşılacak üst sınır. **ConvNeXt-T yalnızca RGB ile 41,4 / %16,2** (kardeş-sızıntısı etkisi Adım 16'da ölçülecek) |

---

## 3. Durum — şu an neredeyiz?

| Faz | İş | Durum |
| --- | --- | --- |
| 0 | Literatür taraması (39 makale) | ✅ Tamamlandı |
| 1 | Veri seti seçimi ve gerekçesi | ✅ Tamamlandı |
| — | Platform seçimi | ✅ Kaggle (eğitim) + Mac. Kaggle kurulumu ✅ (repo okuma/yazma doğrulandı) |
| 2 | Kurulum, veri indirme | ✅ Mac'te: repo, `metrics.py`, veri okuyucular, testler; Nutrition5k indirildi |
| 2 | Temizlik, EDA | ✅ Nutrition5k (`notebooks/01_eda_nutrition5k.ipynb`) · Food-101 (`notebooks/02_eda_food101.ipynb`); bulgular §2'de |
| 3 | Baseline modeller | ✅ Naif (Adım 8, `src/baselines.py`) · Klasik ML (Adım 9, `src/classic_ml.py`) · **Kontrol noktası 1 ✅** |
| 3 | CNN / Transformer eğitimi | ✅ Adım 10–13 (`notebooks/10_egitim_kaggle.ipynb`, `11_egitim_kaggle_adim12-13.ipynb`; bulgular ve Tablo 1–2 §2'de) |
| 4 | Değerlendirme + VLM referansı | Adım 15 ✅ (figürler, Tablo 1–3; bulgular §2) · Adım 14 ⏳ Gemini 3 Flash, 507 tabak çalışıyor (`src/vlm_eval.py`) |
| 5 | İyileştirme ve ablation | ⬜ Gün 6–8 |
| — | Tablolar, figürler, kodun temizlenmesi | Tablolar + figürler ✅ (`src/make_figures.py` ile yeniden üretilir) · kod temizliği ⬜ |
| 6–7 | Rapor ve sunum | ⬜ Geliştirmeden sonra |

**Kısaca:** Adım 1–13 ve 15 bitti; Adım 14 (VLM) çalışıyor. En iyiler: sınıflandırmada **ViT-B/16, F1 0,951**; kaloride **ConvNeXt-T, test MAE 41,4 kcal (PMAE %16,2)**. Her iki literatür hedefi geçildi (§2). Değerlendirme figürleri ve Tablo 1–3 hazır; hata analizi en büyük hataların yeşilliğin altına gizlenmiş proteinli yüksek kalorili tabaklarda olduğunu gösterdi (Grad-CAM doğruluyor). VLM referansı Gemini 3 Flash (literatürdeki modelin kendisi) ile `gemini-3-flash-preview` üzerinden alınıyor; 2.5 serisi yeni anahtarlara kapalı, `gemini-3.8-flash` ise metin isteğine bile yanıt vermedi (zaman aşımı). Sıradaki: **Adım 16–17**, §2'deki kanıta dayalı teknik listesiyle (madde 6).

### Açık konular (karar bekliyor)

| Konu | Durum |
| --- | --- |
| **Araştırma sorusu deneylerle uyuşmuyor** | §1'deki soru "yemek türü + kalori birlikte" diyor, ancak tür etiketi Food-101'de, kalori Nutrition5k'da; hiçbir fotoğrafta ikisi birden yok, bu yüzden soru hiçbir deneyde test edilmiyor. Öneri: soruyu Adım 12'ye uydurmak → "Kaloriyle birlikte kütle ve makroları da tahmin etmek, yalnızca kalori tahmininden daha mı iyi?" Food-101 sınıflandırması Tablo 1 için ayrı deney olarak kalır. **Adım 12 sonucu:** ResNet-50'de fark anlamlı değil (§2 madde 8); soru bu haliyle seçilirse Adım 17'de ConvNeXt-T üzerinde temiz bir tekrar şart. |
| Teslim tarihi | Belgede yok. Gün 1'in tarihini ve teslim gününü §4'e yazın |
| İş bölümü | Adım 5'ten sonra iki bağımsız hat var: **kalori** (Nutrition5k EDA → kalori modelleri → iyileştirme) ve **sınıflandırma** (Food-101 → klasik ML → CNN/ViT → VLM betiği). İki kişi paralel çalışırsa süre ve GPU kotası ikiye katlanır |
| VLM seçimi ve bütçe | ✅ Karar: Gemini 3 Flash (`gemini-3-flash-preview`), ücretsiz katman, anahtar traldo'da (`GEMINI_API_KEY`); tabak başına ~1,2 K giriş + ~1,7 K çıkış (düşünme) tokenı, ~11 sn. İkinci VLM isteğe bağlı (`--provider openai` hazır) |
| **Literatürdeki 150,8 kcal'lik ortalama baseline'ı tekrarlanamıyor** | Bizim 507 tabaklık test kümemizde ortalama tahmin 168,4 kcal MAE veriyor. Test kümesinin kendi medyanını bilen (hileli) en iyi sabit tahmin bile 167,0. Yani o sayı farklı bir test kümesinden ya da protokolden geliyor. Literatürü toplayan kişi kaynağı kontrol etmeli: hangi bölme, kaç tabak? Aynı soru §2'deki diğer referans sayılar için de geçerli. Tablo 2'de baseline satırı olarak kendi ölçümümüz kullanılır |
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

**Durum (30 Ekim):** Adım 1–13, 15 ✅ · Adım 14 ⏳ · Kontrol noktası 1 ✅ · Kontrol noktası 2 (Adım 15) ✅

**Sıradaki adım:** Adım 14 sonucu `results.csv`'ye düşünce (`vlm_gemini_3_flash_preview`) Tablo 2'ye VLM satırı eklenir. Sonra **Adım 16–17**: §2 "Değerlendirme bulguları" madde 6'daki teknikler tek tek ablation olarak (Kaggle, `train_reg.py`; kalori ablation'larında `--patience 10`).

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

**Kaggle'da kurulum** (bir kez, kişi başı):

1. kaggle.com → Settings → **Phone verification** (GPU ve internet için şart)
2. GitHub token'ı (**classic**): [github.com/settings/tokens/new](https://github.com/settings/tokens/new) → Note: `kaggle` · Expiration: 90 days · scope: **`repo`** → Generate token → `ghp_...` ile başlayan anahtarı kopyalayın. (Fine-grained token denendi; yalnızca okuma izni verdi ve push 403 hatası aldı.) Classic token hesabınızdaki tüm repolara erişir; yalnızca Kaggle Secrets'a yapıştırın.
3. Kaggle → Create → New Notebook → File → **Import Notebook** → `notebooks/00_kaggle_kurulum.ipynb`
4. Notebook ayarları: Accelerator **GPU T4 x2** · Internet **On** · Environment **"Pin to original environment"** (paket sürümleri oturumlar arasında değişmesin)
5. Add-ons → **Secrets** → Add secret: adı `GH_TOKEN`, değeri token → bu notebook için işaretleyin
6. İlk kod hücresinde adınızı/e-postanızı yazın → **Run All**. Kontrol hücrelerinin hepsi ✓ ise kurulum tamam.

Her yeni notebook'un başına `00_kaggle_kurulum.ipynb`'nin ilk iki kod hücresini (Kurulum + Veri) kopyalayın. Token hiçbir dosyaya yazılmaz, yalnızca oturumun hafızasında durur. Veri `/tmp/kalori-data`'ya iner; `/kaggle/working`'e girmediği için notebook çıktısını şişirmez. Her oturumda yeniden indirmek birkaç dakika sürer.

**Bilinen sorun (macOS):** torch ve xgboost aynı Python sürecinde yüklenirse çöker (segfault; ikisi de kendi OpenMP kütüphanesini getiriyor). `src/features.py` özellik çıkarımını ayrı süreçte yapar; `src/classic_ml.py` torch'u hiç yüklemez (bir test bunu denetler). Notebook'ta ikisini aynı kernel'de kullanmayın.

Kaggle'da torch/torchvision/timm zaten kurulu; `pip install -r requirements.txt` **yapmayın** (torch'u yeniden kurar). Eksik paket çıkarsa yalnızca onu kurun.

### 5.2 Repo yapısı

```
kalori-tahmini/
├── README.md              ← bu dosya
├── data/                  ← git'e EKLENMEZ (.gitignore)
│   ├── nutrition5k/
│   └── food101/
├── notebooks/
│   ├── 00_kaggle_kurulum.ipynb   ← Kaggle'ın ilk hücreleri (§5.1)  ✅
│   ├── 01_eda_nutrition5k.ipynb  ✅
│   ├── 02_eda_food101.ipynb      ✅
│   ├── 10_egitim_kaggle.ipynb    ← Adım 10–11 eğitimleri, Kaggle (sonuçları kendisi push'lar)  ✅
│   ├── 11_egitim_kaggle_adim12-13.ipynb ← Adım 12–13: çok görevli, ConvNeXt-T, ViT-B/16 (Kaggle)
│   └── ...
├── src/
│   ├── config.py          ← ORTAK sabitler (tohum, boyut, yollar, cihaz)       ✅
│   ├── data.py            ← veri yükleyiciler, val ayrımı, BAD_DISHES, Food-101 sınıf listesi ✅
│   ├── transforms.py      ← kalori / sınıflandırma dönüşümleri (§6.1)          ✅
│   ├── metrics.py         ← ORTAK: tüm metrikler + güven aralıkları             ✅
│   ├── results.py         ← results.csv'ye satır ekleme, tahmin kaydetme        ✅
│   ├── plot_style.py      ← ORTAK grafik stili: tüm figürler `apply()` ile      ✅
│   ├── compare.py         ← iki modeli eşleştirilmiş bootstrap ile karşılaştır  ✅
│   ├── baselines.py       ← naif baseline'lar (Adım 8)                          ✅
│   ├── features.py        ← HOG + renk / donmuş CNN özellikleri, önbellek       ✅
│   ├── classic_ml.py      ← klasik ML (Adım 9)                                  ✅
│   ├── models.py          ← omurga + başlıklar, checkpoint biçimi, tek foto tahmini ✅ (temel)
│   ├── eval_own_photos.py ← kendi fotoğraflarımızla dış test (§7, Faz 4)       ✅
│   ├── trainer.py         ← ortak eğitim döngüsü (AdamW, ısınma+cosine, erken durdurma, AMP) ✅
│   ├── train_cls.py       ← sınıflandırma eğitimi (Adım 10, 13)                 ✅
│   ├── train_reg.py       ← kalori eğitimi; --targets ile çok görevli (Adım 11–13) ✅
│   ├── vlm_eval.py        ← VLM sıfır-atış referansı (Gemini / OpenAI uyumlu, stdlib) ✅
│   ├── make_figures.py    ← Adım 15: tüm değerlendirme figürleri + Tablo 1–3 (md)  ✅
│   └── gradcam.py         ← Grad-CAM (checkpoint gerekir)                       ✅
├── scripts/
│   ├── download_nutrition5k.py  ← gsutil'siz indirme (Mac ve Kaggle)           ✅
│   ├── get_food101.py           ← Food-101 25 sınıflık alt küme: build / fetch   ✅
│   └── share_checkpoint.py      ← checkpoint'leri GitHub Release ile paylaş (§6.4) ✅
├── tests/                 ← python -m pytest -q
├── app.py                 ← Gradio demosu (Adım 19)                            ✅ (iskelet)
├── own_photos/            ← kendi fotoğraflarımız + labels.csv (§7, Faz 4)
├── results/
│   ├── results.csv        ← ORTAK: her deneyin sonucu buraya
│   ├── preds/             ← her modelin test tahminleri (git'te)
│   ├── history/           ← eğitim eğrileri: epoch başına kayıp ve val skoru (git'te)
│   ├── data_card_nutrition5k.csv, data_card_food101.csv ← veri kartları (EDA)
│   ├── vlm/               ← VLM ham yanıtları (jsonl, devam edilebilir; git'te)
│   ├── tablo1.md, tablo2.md, tablo3_kalori_araligi.md ← rapora hazır tablolar
│   ├── eval_*.csv         ← karışan çiftler, en kötü 20, aralık MAE
│   └── figures/           ← n5k_*, f101_* (EDA), eval_* (Adım 15)
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
| train (`rgb_train_ids.txt`) | 4059 | 2755 | 3 | **2752** → 2472 train + 280 val |
| test (`rgb_test_ids.txt`) | 709 | 507 | 0 | **507** |

Üstten görüntüsü olan tabaklar tam olarak `depth_{train,test}_ids.txt` listeleri. Literatürdeki "üstten RGB" sonuçları da bu 507 tabaklık test kümesiyle karşılaştırılmalı. Bölmeler yalnızca Cafe 1'i kapsıyor (Cafe 2 kullanılmıyor, bkz. `dish_ids/README`).

**Metadata:** Her satır bir tabak: `dish_id, total_calories, total_mass, total_fat, total_carb, total_protein`, ardından malzeme başına 7'li gruplar. Satır uzunlukları farklı olduğu için `pandas.read_csv` hata verir; `src/data.py` → `read_dish_metadata` bunu `csv` modülüyle okur.

**Görüntülerden öğrendiklerimiz:** Tabaklar her zaman ortada değil, bazıları kenara taşıyor (o yüzden kırpma yok, §6.1). EDA'da 3 hatalı tabak bulundu ve `BAD_DISHES`'a eklendi: telefon görüntüsü (313 kcal), "2991 g pirinç" yazım hatası (3943 kcal), domatesli ama "boş tabak" etiketli tabak (bkz. `results/figures/n5k_hatali_tabaklar.png`).

### 5.4 Food-101 (alt küme)

Tam Food-101 5 GB; bize 25 sınıf lazım. Alt küme bir kez çıkarıldı, görüntülerin kısa kenarı 256 px'e küçültüldü (eğitim 224 px kullanıyor; okuma çok hızlanır) ve repo'nun `data` Release'ine kondu. Herkes yalnızca alt kümeyi indirir:

```bash
python scripts/get_food101.py fetch                          # Mac → data/food101/
python scripts/get_food101.py fetch --out /tmp/kalori-data   # Kaggle (kurulum notebook'u bunu yapıyor)
```

Alt küme **25 sınıf** olarak sabitlendi: `src/data.py` → `FOOD101_CLASSES`. Liste sonradan değişirse önceki sınıflandırma sonuçları karşılaştırılamaz hale gelir. Listede bilerek karışabilir çiftler var (bolognese/carbonara, caesar/greek salad, pancakes/waffles); karışıklık matrisi ve hata analizi için iyi malzeme olur.

**Bölmeler:** `Food101Subset("train" | "val" | "test", transform)`. Resmi train'den her sınıfın %10'u sabit tohumla val'e ayrılır → sınıf başına **675 train / 75 val / 250 test**. EDA'da bulunan 11 sorunlu görüntü train/val'den çıkarıldı (`FOOD101_EXCLUDE`) → son sayılar **16.865 train / 1.874 val / 6.250 test**. Food-101'in resmi train'i **bilerek temizlenmemiştir** (etiket gürültüsü var, bkz. EDA); test elle temizlenmiştir.

### 5.5 Gün 1 bitiş kriterleri

- [x] Platform seçildi ve §3'e yazıldı; Kaggle kurulumu tamam
- [x] Repo açıldı, bu dosya `README.md` olarak eklendi
- [x] `src/metrics.py` yazıldı (§6.2) ve tüm deneyler onu kullanıyor
- [x] Nutrition5k RGB görüntüleri, metadata ve bölmeler indirildi; kullanılabilir train/test tabak sayıları not edildi
- [x] Food-101 alt kümesi hazır (`scripts/get_food101.py fetch`)

---

## 6. Ortak kurallar (sonuçların karşılaştırılabilir olması için)

### 6.1 Sabitler

| Ayar | Değer |
| --- | --- |
| Rastgelelik tohumu | `42` (torch, numpy, random) |
| Görüntü boyutu | 224 × 224 (iyileştirme fazında 384 denenebilir). Nutrition5k: 640×480'in **tamamı** 224×224'e küçültülür, kırpılmaz |
| Normalizasyon | Omurganın ön eğitimdeki değerleri (`model_norm()`, timm `pretrained_cfg`): ResNet / EfficientNet / ConvNeXt → ImageNet ortalama/std, **ViT → 0,5/0,5**. Eğitim betikleri otomatik alır ve checkpoint'e yazar |
| Doğrulama kümesi | Train'in ~%10'u, **zaman gruplarıyla** ayrılır: `make_val_split()`. Aralarında 10 dakikadan az olan tabaklar aynı gruptadır ve aynı kümeye düşer. Sonuç: 2472 train / 280 val. Gerekçe: tabaklar malzeme malzeme eklenerek art arda çekilmiş; ardışık tabak çiftlerinin %57'sinde malzeme listeleri iç içe. Rastgele ayırma neredeyse aynı tabağı hem train'e hem val'e koyar |
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
python -m src.compare results/preds/resnet50_cls.csv results/preds/vit_b16_cls.csv   # sınıflandırma → macro-F1
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

Betik hazır (`src/vlm_eval.py`, yalnızca standart kütüphane; ağsız testler `tests/test_vlm_eval.py`):

```bash
export GEMINI_API_KEY=...                           # sohbete/dosyaya yazmayın
python -m src.vlm_eval --model gemini-3-flash-preview --rpm 15        # 507 tabak, ~1,5 sa
python -m src.vlm_eval --n 300 --seed 42 --rpm 15                       # maliyet sorunsa
python -m src.vlm_eval --provider openai --model <ad> --base-url <url>  # başka sağlayıcı (OPENAI_API_KEY)
python -m src.vlm_eval --dry-run --n 5                                  # ağsız deneme
```

Kesinti veya kota hatasında **aynı komutu tekrar verin**: `results/vlm/<run>.jsonl`'daki bitmiş tabaklar atlanır. Çıktılar: `results/preds/vlm_<model>.csv` (ayrıştırılamayanlar atılmış), `results.csv`'de `calorie` + `calorie_bins` satırı (notlarda tabak sayısı, ayrıştırılamayan oranı, train ortalamasıyla yedek MAE, token toplamları, `--price-in/--price-out` verilirse maliyet). Model notu (30 Ekim): `gemini-2.5-*` yeni anahtarlara kapalı (404); `gemini-3.8-flash` ve `gemini-flash-latest` yanıt vermedi (75 sn zaman aşımı); `gemini-3-flash-preview`, `3.5-flash`, `3.6-flash` 1–2 sn.

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
| **Gün 3 sonu (Adım 9, kontrol noktası 1)** ✅ | Baseline sonuçları Tablo 1 ve 2'de | İyileştirme fazı 2 tekniğe iner (yukarıda) |
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
| 2026-10-30 | traldo | Adım 15 ✅ (figürler, Tablo 1–3, Grad-CAM) + Adım 14 betiği ✅ (çalıştırma sürüyor) | Adım 14 sonucunu Tablo 2'ye ekle → Adım 16–17 ablation'ları | `src/make_figures.py` (9 figür + 3 md tablo), `src/gradcam.py`, `src/vlm_eval.py` (stdlib, devam edilebilir). Bulgular §2 "Değerlendirme bulguları": en büyük hatalar yeşillik altındaki proteinli yüksek kalorili tabaklar; apple pie en zayıf sınıf. `checkpoints/convnext_tiny_reg.pt` Mac'e indirildi (git dışı) |
| 2026-10-30 | traldo | Adım 12–13 ✅ (Kaggle, notebook 11 ~2,5 sa) | Adım 14 (VLM: model + API anahtarı kararı bekliyor) · Adım 15 (figürler, `results/preds/`'ten Mac'te) | Sonuçlar §2 Tablo 1–2: ConvNeXt-T kalori 41,4 kcal, ViT-B/16 sınıflandırma F1 0,951. Düzeltilmiş başlatma EfficientNet-B0'ı 104 → 52,8'e indirdi; ResNet-50 ise `--patience 7` yüzünden 14. epoch'ta durup 68,7'ye kötüleşti (eski 59,7 run'ı `*_timminit` adıyla duruyor ve raporda kullanılabilir). Çok görevli ResNet-50 58,0 (tek görevliden anlamlı farklı değil). Checkpoint'lerin hepsi Release `checkpoints`'ta. Balıklı/balıksız MAE Tablo 2'ye eklendi |
| 2026-10-29 | traldo | Adım 10–11 ✅ (Kaggle) + Adım 12–13 hazırlığı | `notebooks/11_egitim_kaggle_adim12-13.ipynb`'yi Kaggle'da çalıştır (~4 sa) | Sonuçlar §2'de: ResNet-50 kalori 59,7 kcal, EfficientNet-B0 sınıflandırma F1 0,917. EfficientNet kalori başlatma hatası → `init_regression_head` (tüm kalori modelleri train ortalamasıyla başlıyor); eski kalori sonuçları `*_timminit` adıyla duruyor. `src.compare` artık sınıflandırma dosyalarında macro-F1 karşılaştırıyor. ViT'nin adı `vit_b16_*` |
| 2026-10-28 | traldo | Adım 10–11 hazırlığı: eğitim betikleri | `notebooks/10_egitim_kaggle.ipynb`'yi Kaggle'da çalıştır | `src/trainer.py` + `train_cls.py` + `train_reg.py` (Mac'te küçük denemelerle test edildi). Kalori hedefleri train ortalamasına bölünerek ölçekleniyor; `--targets` ile çok görevli (Adım 12 hazır). ViT kendi normalizasyonunu kullanıyor. `--limit` yalnızca `--dry-run` ile |
| 2026-10-28 | traldo | Adım 9: klasik ML · Kontrol noktası 1 ✅ | Adım 10–11: `train_cls.py` / `train_reg.py` yazılıp Kaggle'da ResNet-50 ve EfficientNet-B0 | `python -m src.classic_ml` → 8 model (val'de seçildi). Test: CNN + LogReg F1 0,837; CNN + XGBoost 77,4 kcal (Ridge 78,2; fark anlamlı değil). Özellik önbelleği `data/features/`. macOS'ta torch + xgboost aynı süreçte çöküyor → ayrı süreç (§5.1) |
| 2026-10-28 | traldo | Adım 5, 7, 8: Food-101 alt kümesi + EDA, naif baseline'lar | Adım 9: klasik ML | Food-101 alt kümesi `data` Release'inde (662 MB; `get_food101.py fetch`, kesintide kaldığı yerden devam eder). `FOOD101_EXCLUDE` 11 görüntü (9 test sızıntısı + 1 çelişkili çift). Naif baseline'lar `results.csv`'de: test MAE 168,4 (ortalama) / 167,3 (medyan). Literatürdeki 150,8'lik baseline tekrarlanamadı (§3 Açık konular) |
| 2026-10-28 | traldo | Adım 6: Nutrition5k EDA | Adım 5: Food-101 indir → Adım 7: Food-101 EDA → Adım 8 | `BAD_DISHES` 3 tabak; kullanılabilir 2472 train / 280 val / 507 test. Baseline hedefi **ham kcal + L1** (log1p ablation'da). `CALORIE_BIN_EDGES` sabitlendi. Ortak grafik stili: `src/plot_style.py`. Bulgular §2'de |
| 2026-10-28 | traldo | Adım 2, 3, 4 (+ Adım 1 Mac tarafı, Adım 5 sınıf listesi) | Adım 5: Food-101 indir → Adım 6: Nutrition5k EDA | **Yeni:** zaman gruplu val ayrımı, kalori için ölçeksiz artırma, bootstrap güven aralıkları, `src.compare`, kendi fotoğraf testi, demo iskeleti, checkpoint paylaşımı. Food-101 25 sınıf. `BAD_DISHES`'ta 1 tabak. Kullanılabilir veri: 2476 train / 278 val / 507 test. Kaggle kurulumu tamam (classic token; `notebooks/00_kaggle_kurulum.ipynb`). "Açık konular" tablosuna bakın (§3) |
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
