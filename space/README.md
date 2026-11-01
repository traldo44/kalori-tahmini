---
title: Kalori Tahmini
emoji: 🍽️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Yemek fotoğrafından kalori, kütle ve makro tahmini (ConvNeXt-T)
---

# Yemek fotoğrafından kalori tahmini

Telefondan tabağı **tam üstten** çekin veya galeriden yükleyin; model kaloriyi, kütleyi ve makroları tahmin eder.

- **Model:** ConvNeXt-T (ImageNet-12k ön eğitimli), çok görevli başlık, 224 px, yalnızca RGB
- **Eğitim verisi:** Nutrition5k (2 752 tabak, üst kamera)
- **Test (507 tabak):** MAE 40,3 kcal [36,2–45,3], PMAE %15,8, R² 0,90
- **Karşılaştırma:** sıfır-atış VLM 82,1 kcal · klasik ML 77,4 kcal · Nutrition5k RGB baseline'ı 70,6 kcal

Kod, deneyler ve rapor: [github.com/traldo44/kalori-tahmini](https://github.com/traldo44/kalori-tahmini).
Bu Space `python scripts/deploy_space.py` ile o repodan üretilir; burada elle düzenleme yapmayın.
