| Tablo 3 · Kalori aralığı (düşük < 116,1 ≤ orta < 313,7 ≤ yüksek kcal, test) | Doğruluk | Precision (macro) | Recall (macro) | Macro-F1 [%95 GA] |
| --- | --- | --- | --- | --- |
| Naif: train ortalaması | %31,4 | 0,104 | 0,333 | 0,159 [0,142–0,174] |
| Donmuş ResNet-50 → XGBoost | %74,2 | 0,771 | 0,742 | 0,747 [0,708–0,785] |
| ResNet-50 (timm başlatması) | %80,5 | 0,814 | 0,804 | 0,806 [0,768–0,839] |
| ResNet-50 (erken durdu, 14/30) | %79,1 | 0,804 | 0,790 | 0,790 [0,756–0,826] |
| ResNet-50 çok görevli | %79,9 | 0,808 | 0,798 | 0,799 [0,763–0,832] |
| EfficientNet-B0 (hatalı başlatma) | %59,8 | 0,611 | 0,595 | 0,600 [0,558–0,639] |
| EfficientNet-B0 | %81,7 | 0,821 | 0,815 | 0,817 [0,781–0,848] |
| ViT-B/16 | %86,0 | 0,861 | 0,859 | 0,859 [0,827–0,889] |
| ConvNeXt-T | %86,2 | 0,864 | 0,861 | 0,861 [0,828–0,889] |
| VLM sıfır-atış (Gemini 3.5 Flash-Lite) | %75,7 | 0,750 | 0,752 | 0,751 [0,713–0,784] |
