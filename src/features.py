"""Klasik ML için özellik çıkarımı (Adım 9). Sonuçlar data/features/<veri>_<özellik>_<bölme>.npz'e önbelleğe alınır.

hog_color — elle tasarlanmış: 128×128 gri görüntüde HOG (1764) + HSV renk histogramı (3 × 32 = 96) → 1860 boyut.
            Görüntü kırpılmadan küçültülür (Nutrition5k'da porsiyon bilgisi korunsun, §6.1).
cnn       — donmuş ImageNet ResNet-50'nin havuzlanmış çıktısı → 2048 boyut (ince ayar yok).

ÖNEMLİ: torch ile xgboost aynı süreçte yüklenince macOS'ta çöküyor (ikisi de kendi OpenMP kütüphanesini getiriyor,
hangi sırayla yüklenirse yüklensin biri segfault veriyor). Bu yüzden `load()` özellikleri AYRI bir süreçte
(`python -m src.features <veri> <özellik>`) hesaplar/denetler; çağıran süreç torch'u hiç yüklemez.
Bu modülün en üstünde torch veya src.data (torch yükler) içe aktarılmamalı.
"""
import argparse
import subprocess
import sys

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage.feature import hog

from .config import DATA_DIR, ROOT

FEATURE_DIR = DATA_DIR / "features"
SPLITS = ("train", "val", "test")
HOG_SIZE = 128
HIST_BINS = 32


def hog_color_features(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((HOG_SIZE, HOG_SIZE), Image.BILINEAR)
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255
    h = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    hsv = np.asarray(img.convert("HSV"))
    hist = [np.histogram(hsv[..., c], bins=HIST_BINS, range=(0, 256))[0] / hsv[..., c].size for c in range(3)]
    return np.concatenate([h, *hist]).astype(np.float32)


def _hog_color_file(path):
    with Image.open(path) as im:
        return hog_color_features(im)


def extract_hog_color(paths, n_jobs=-1):
    return np.stack(Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_hog_color_file)(p) for p in paths))


def extract_cnn(paths, transform, batch_size=64):
    import timm
    import torch
    from torch.utils.data import DataLoader, Dataset

    from .config import get_device

    class _Paths(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            with Image.open(paths[i]) as im:
                return transform(im.convert("RGB"))

    dev = get_device()
    model = timm.create_model("resnet50", pretrained=True, num_classes=0).to(dev).eval()  # num_classes=0 → 2048'lik havuz çıktısı
    with torch.no_grad():  # num_workers=0: macOS'ta takılmasın
        return torch.cat([model(x.to(dev)).float().cpu() for x in DataLoader(_Paths(), batch_size=batch_size)]).numpy()


def split_items(dataset):
    """{bölme: (görüntü yolları, etiketler, kimlikler)}. Bölme tanımları src/data.py'den (torch yükler)."""
    if dataset == "food101":
        from .data import Food101Subset
        out = {}
        for s in SPLITS:
            samples = Food101Subset(s, transform=None).samples
            out[s] = ([p for p, _ in samples], np.array([lab for _, lab in samples]),
                      [f"{p.parent.name}/{p.stem}" for p, _ in samples])
        return out
    from .config import N5K_DIR
    from .data import load_n5k_split, make_val_split, read_dish_metadata
    meta = read_dish_metadata()
    train, val = make_val_split(load_n5k_split("train", metadata=meta, verbose=False))
    test = load_n5k_split("test", metadata=meta, verbose=False)
    return {s: ([N5K_DIR / "realsense_overhead" / d / "rgb.png" for d in ids], np.array([meta[d]["calories"] for d in ids]), ids)
            for s, ids in (("train", train), ("val", val), ("test", test))}


def _path(dataset, feat, split):
    return FEATURE_DIR / f"{dataset}_{feat}_{split}.npz"


def compute(dataset, feat):
    """Eksik ya da eskimiş (örnek listesi değişmiş) önbellekleri hesaplar. torch yükleyebilir → ayrı süreçte çalıştırın."""
    for split, (paths, y, ids) in split_items(dataset).items():
        path = _path(dataset, feat, split)
        if path.exists() and np.array_equal(np.load(path)["ids"], np.asarray(ids)):
            continue
        if feat == "hogcolor":
            X = extract_hog_color(paths)
        else:
            from .transforms import calorie_transform, classification_transform
            X = extract_cnn(paths, calorie_transform(False) if dataset == "n5k" else classification_transform(False))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=X, y=y, ids=np.asarray(ids))
        print(f"  hesaplandı: {path.name} {X.shape}", flush=True)


def load(dataset, feat):
    """{bölme: (X, y, kimlikler)}. Önbelleği ayrı bir süreçte denetler/hesaplar, sonra yalnızca .npz okur (torch'suz)."""
    subprocess.run([sys.executable, "-m", "src.features", dataset, feat], check=True, cwd=ROOT)
    out = {}
    for s in SPLITS:
        z = np.load(_path(dataset, feat, s))
        out[s] = (z["X"], z["y"], list(z["ids"]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["food101", "n5k"])
    ap.add_argument("feat", choices=["hogcolor", "cnn"])
    args = ap.parse_args()
    compute(args.dataset, args.feat)
