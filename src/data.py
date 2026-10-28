"""Veri yükleyiciler: Nutrition5k (kalori) ve Food-101 alt kümesi (sınıflandırma)."""
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import FOOD101_DIR, N5K_DIR, SEED, VAL_FRACTION, VAL_GROUP_GAP_SEC

# ---------------------------------------------------------------- Nutrition5k

TARGETS = ("calories", "mass", "fat", "carb", "protein")

# EDA'da bulunan hatalı tabaklar: eğitimden ve değerlendirmeden çıkarılır.
# Yeni bulduğunuzu gerekçesiyle buraya ekleyin ve devir günlüğüne yazın.
BAD_DISHES = {
    "dish_1559172356": "görüntüde yemek yok, telefon var (etiket 313 kcal)",
}


def read_dish_metadata(meta_dir=N5K_DIR / "metadata"):
    """{dish_id: {calories, mass, fat, carb, protein}}. Satır uzunlukları farklı olduğu için pandas yerine csv."""
    dishes = {}
    for path in sorted(Path(meta_dir).glob("dish_metadata_cafe*.csv")):
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0].startswith("dish_"):
                    continue  # başlık veya boş satır
                dishes[row[0]] = dict(zip(TARGETS, map(float, row[1:6])))
    return dishes


def read_split(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_n5k_split(split, root=N5K_DIR, metadata=None, verbose=True):
    """Resmi 'train' veya 'test' bölmesindeki, üstten görüntüsü olan ve hatalı olmayan tabaklar."""
    metadata = metadata if metadata is not None else read_dish_metadata(Path(root) / "metadata")
    ids = read_split(Path(root) / f"dish_ids/splits/rgb_{split}_ids.txt")
    no_image = [d for d in ids if not (Path(root) / "realsense_overhead" / d / "rgb.png").exists()]
    keep = [d for d in ids if d not in set(no_image) and d in metadata and d not in BAD_DISHES]
    if verbose:
        print(f"{split}: {len(ids)} kimlik → {len(keep)} kullanılabilir "
              f"({len(no_image)} görüntüsüz, {len(ids) - len(no_image) - len(keep)} hatalı/metadatasız)")
    return keep


def dish_time(dish_id):
    """Tabak kimliği aslında çekim zamanı (Unix saniye): dish_1556572657."""
    return int(dish_id.split("_")[1])


def time_groups(dish_ids, gap=VAL_GROUP_GAP_SEC):
    """Aralarında `gap` saniyeden az olan ardışık tabakları aynı gruba koyar."""
    groups = []
    for d in sorted(dish_ids, key=dish_time):
        if groups and dish_time(d) - dish_time(groups[-1][-1]) <= gap:
            groups[-1].append(d)
        else:
            groups.append([d])
    return groups


def make_val_split(train_ids, frac=VAL_FRACTION, gap=VAL_GROUP_GAP_SEC, seed=SEED):
    """Train'den doğrulama kümesi ayırır; aynı zaman grubu iki kümeye bölünmez.

    Neden: Nutrition5k'da tabaklar malzeme malzeme eklenerek art arda çekildi (ardışık
    tabakların %57'sinde malzemeler iç içe). Rastgele ayırma neredeyse aynı tabağı hem
    train'e hem val'e koyar ve doğrulama skoru olduğundan iyi görünür.
    """
    groups = time_groups(train_ids, gap)
    random.Random(seed).shuffle(groups)
    val, target = [], frac * len(train_ids)
    for g in groups:
        if len(val) >= target:
            break
        val += g
    val_set = set(val)
    return [d for d in train_ids if d not in val_set], [d for d in train_ids if d in val_set]


class Nutrition5kDataset(Dataset):
    """(görüntü, hedef, dish_id) döner. Hedef, `targets` sırasıyla bir float tensör."""

    def __init__(self, dish_ids, metadata, transform, targets=("calories",), root=N5K_DIR):
        self.ids, self.meta, self.transform = list(dish_ids), metadata, transform
        self.targets, self.root = tuple(targets), Path(root)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = self.ids[i]
        img = Image.open(self.root / "realsense_overhead" / d / "rgb.png").convert("RGB")
        y = torch.tensor([self.meta[d][t] for t in self.targets], dtype=torch.float32)
        return self.transform(img), y, d


# ---------------------------------------------------------------- Food-101 alt kümesi

# SABİT LİSTE: değişirse önceki sınıflandırma sonuçları karşılaştırılamaz.
# Seçim: yaygın, görsel olarak çeşitli yemekler + bilerek bırakılmış karışabilir çiftler
# (bolognese/carbonara, caesar/greek salad, pancakes/waffles) → karışıklık matrisi için malzeme.
FOOD101_CLASSES = [
    "apple_pie", "caesar_salad", "chicken_curry", "chicken_wings", "club_sandwich",
    "donuts", "dumplings", "french_fries", "fried_rice", "greek_salad",
    "hamburger", "hot_dog", "ice_cream", "lasagna", "omelette",
    "pancakes", "pizza", "ramen", "risotto", "spaghetti_bolognese",
    "spaghetti_carbonara", "steak", "sushi", "tacos", "waffles",
]


class Food101Subset(Dataset):
    """FOOD101_CLASSES'taki sınıflar; etiket = listedeki sıra. (görüntü, etiket) döner.

    Beklenen yapı (torchvision Food101 indirmesi veya Kaggle kopyası):
        <root>/food-101/meta/{train,test}.json   ve   <root>/food-101/images/<sınıf>/<id>.jpg
    """

    def __init__(self, split, transform, root=FOOD101_DIR, classes=FOOD101_CLASSES):
        base = Path(root) / "food-101"
        with open(base / "meta" / f"{split}.json") as f:
            meta = json.load(f)
        self.classes, self.transform = list(classes), transform
        self.samples = [(base / "images" / f"{name}.jpg", label)
                        for label, cls in enumerate(self.classes) for name in meta[cls]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label
