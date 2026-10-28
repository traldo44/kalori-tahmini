"""Omurga + başlıklar ve checkpoint biçimi.

Şimdilik: timm omurgası + doğrusal çıktı katmanı (n_outputs = tahmin edilen hedef sayısı).
Çok görevli başlık (kalori + kütle + makrolar) Adım 12'de outputs listesi genişletilerek eklenir.
"""
import timm
import torch
from PIL import Image, ImageOps

from .transforms import calorie_transform, crop_to_aspect


def build_regressor(backbone="resnet50", n_outputs=1, pretrained=True):
    return timm.create_model(backbone, pretrained=pretrained, num_classes=n_outputs)


def encode_target(y, target):
    """Eğitimde hedefe uygulanan dönüşüm. 'log1p' sağa çarpık kalori dağılımı için (EDA kararı)."""
    return torch.log1p(y) if target == "log1p" else y


def decode_target(y, target):
    return torch.expm1(y) if target == "log1p" else y


def save_checkpoint(path, model, backbone, outputs=("calories",), target="log1p", **info):
    """info: ör. test_MAE=92.3 → demo bunu kullanıcıya 'tipik hata' olarak gösterir."""
    torch.save({"backbone": backbone, "outputs": list(outputs), "target": target,
                "state_dict": model.state_dict(), "info": info}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu")
    model = build_regressor(ckpt["backbone"], len(ckpt["outputs"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model.to(device).eval(), meta


@torch.no_grad()
def predict_image(model, meta, img: Image.Image, device="cpu"):
    """Tek fotoğraf → {"calories": ..., ...}. Telefon fotoğrafları önce eğitimdeki 4:3 orana kırpılır."""
    img = crop_to_aspect(ImageOps.exif_transpose(img).convert("RGB"))
    x = calorie_transform(train=False)(img).unsqueeze(0).to(device)
    y = decode_target(model(x)[0].float().cpu(), meta["target"])
    return dict(zip(meta["outputs"], y.tolist()))
