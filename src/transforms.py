"""Görüntü dönüşümleri: KALORİ ve SINIFLANDIRMA için ayrı kurallar.

KALORİ (Nutrition5k) — ölçeği değiştiren artırma YASAK:
    Fotoğraflar sabit yükseklikteki üst kameradan çekildi; model porsiyonu tabağın fotoğrafta
    kapladığı alandan öğrenir. RandomResizedCrop / zoom / ölçekli kırpma aynı tabağı büyük ya da
    küçük gösterir ama etiket aynı kalır → model porsiyonu öğrenemez.
    Güvenli olanlar: yatay/dikey çevirme, 90° döndürme, renk/parlaklık değişimi.
    Kırpma da yok: tabaklar her zaman ortada değil, kenara taşanlar var. Görüntünün tamamı
    224×224'e küçültülür (640×480 → hafif yatay sıkışma; tüm görüntülere aynı uygulandığı için sorun değil).

SINIFLANDIRMA (Food-101) — ölçek önemli değil, standart artırmaların hepsi serbest.

Normalizasyon varsayılanı ImageNet ortalama/std; ön eğitimi farklı normalize edilmiş omurgalarda (ör. ViT: 0,5/0,5)
eğitim betikleri modelin kendi değerlerini (timm pretrained_cfg) mean/std olarak geçirir.
"""
import random

import torch
from PIL import Image
from torchvision import transforms as T

from .config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

N5K_ASPECT = 640 / 480  # Nutrition5k üst kamera görüntü oranı (4:3)


class RandomRot90:
    """0/90/180/270° döndürme. Kare tensörde çalışır; ölçek ve alan oranı korunur."""

    def __call__(self, x):
        return torch.rot90(x, random.randint(0, 3), dims=(-2, -1))


def crop_to_aspect(img: Image.Image, aspect=N5K_ASPECT) -> Image.Image:
    """Telefon fotoğrafını eğitim görüntüleriyle aynı orana (4:3) ortadan kırpar. Demo ve dış test için."""
    w, h = img.size
    if w / h > aspect:   # fazla geniş → yanlardan kırp
        new_w = round(h * aspect)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = round(w / aspect)  # fazla uzun (ör. dikey telefon fotoğrafı) → üstten/alttan kırp
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


def calorie_transform(train: bool, size=IMG_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    steps = [T.Resize((size, size)), T.ToTensor()]
    if train:
        steps += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRot90(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        ]
    steps.append(T.Normalize(mean, std))
    return T.Compose(steps)


def classification_transform(train: bool, size=IMG_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if train:
        steps = [T.RandomResizedCrop(size), T.RandomHorizontalFlip(), T.RandAugment()]
    else:
        steps = [T.Resize(int(size * 256 / 224)), T.CenterCrop(size)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(mean, std)])
