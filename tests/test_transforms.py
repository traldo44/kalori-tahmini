import numpy as np
import torch
from PIL import Image, ImageDraw

from src.config import IMAGENET_MEAN, IMAGENET_STD
from src.transforms import calorie_transform, classification_transform, crop_to_aspect


def plate_image():
    """640×480 siyah zemin üzerinde beyaz 'tabak' (kenara yakın, ortada değil)."""
    img = Image.new("RGB", (640, 480), "black")
    ImageDraw.Draw(img).rectangle((300, 100, 600, 380), fill="white")
    return img


def white_fraction(x):
    rgb = x * torch.tensor(IMAGENET_STD)[:, None, None] + torch.tensor(IMAGENET_MEAN)[:, None, None]
    return (rgb.mean(0) > 0.5).float().mean().item()


def test_calorie_augmentation_keeps_portion_size():
    """3. madde: kalori artırması tabağın kapladığı alanı DEĞİŞTİRMEMELİ."""
    img = plate_image()
    reference = white_fraction(calorie_transform(train=False)(img))
    torch.manual_seed(0)
    fractions = [white_fraction(calorie_transform(train=True)(img)) for _ in range(50)]
    assert max(abs(f - reference) for f in fractions) < 0.01


def test_classification_augmentation_does_change_scale():
    """Karşılaştırma: sınıflandırma artırması ölçeği değiştirir (orada sorun değil)."""
    img = plate_image()
    torch.manual_seed(0)
    fractions = [white_fraction(classification_transform(train=True)(img)) for _ in range(50)]
    assert np.std(fractions) > 0.05


def test_calorie_transform_keeps_whole_frame():
    """Kırpma yok: kenara taşan tabak kesilmemeli."""
    x = calorie_transform(train=False)(plate_image())
    assert x.shape == (3, 224, 224)
    expected = (300 * 280) / (640 * 480)
    assert abs(white_fraction(x) - expected) < 0.01


def test_crop_to_aspect_portrait_phone_photo():
    out = crop_to_aspect(Image.new("RGB", (3000, 4000)))
    assert out.size == (3000, 2250)
    assert crop_to_aspect(Image.new("RGB", (640, 480))).size == (640, 480)
    assert crop_to_aspect(Image.new("RGB", (1920, 1080))).size == (1440, 1080)
