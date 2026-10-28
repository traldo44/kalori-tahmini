"""Ortak sabitler (§6.1). Bir değeri değiştirirseniz devir günlüğüne yazın."""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Kaggle'da veri /kaggle/input/... altındadır: KALORI_DATA ortam değişkeniyle gösterin.
DATA_DIR = Path(os.environ.get("KALORI_DATA", ROOT / "data"))
N5K_DIR = DATA_DIR / "nutrition5k"
FOOD101_DIR = DATA_DIR / "food101"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"

SEED = 42
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Doğrulama kümesi: train'in ~%10'u, ZAMAN GRUPLARIYLA ayrılır (bkz. data.make_val_split).
# Nutrition5k'da tabaklar malzeme malzeme eklenerek art arda çekildi; aralarında
# 10 dakikadan az olan tabaklar aynı gruptadır ve aynı kümeye düşer.
VAL_FRACTION = 0.1
VAL_GROUP_GAP_SEC = 600

# Tablo 3 aralık sınırları (kcal): resmi train'in (hatalılar hariç) üçte birlik dilimleri.
# EDA'da bir kez hesaplandı (notebooks/01_eda_nutrition5k.ipynb) ve SABİTLENDİ.
# düşük < 116.1 ≤ orta < 313.7 ≤ yüksek
CALORIE_BIN_EDGES = (116.1, 313.7)


def set_seed(seed=SEED):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")      # Kaggle
    if torch.backends.mps.is_available():
        return torch.device("mps")       # Apple Silicon Mac
    return torch.device("cpu")
