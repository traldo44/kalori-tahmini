"""Omurga + başlık, hedef kodlama ve checkpoint biçimi.

Omurga: timm modeli; başlık: num_classes = çıktı sayısı (sınıflandırmada sınıf, kaloride hedef sayısı).
Kalori hedefleri eğitimde ölçeklenir (encode_target): z = f(y) / scale, f = kimlik ('raw') veya log1p.
scale = hedefin train ortalaması → her hedef ~1 civarında; çok görevli kayıpta hedefler eşit ağırlıkta kalır.
"""
import timm
import torch
from PIL import Image, ImageOps

from .config import IMAGENET_MEAN, IMAGENET_STD
from .transforms import calorie_transform, crop_to_aspect


def build_model(backbone="resnet50", n_outputs=1, pretrained=True):
    return timm.create_model(backbone, pretrained=pretrained, num_classes=n_outputs)


build_regressor = build_model  # eski ad


def model_norm(model):
    """Omurganın ön eğitimde kullandığı normalizasyon (ör. ViT 0,5/0,5; ResNet ImageNet)."""
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    return tuple(cfg.get("mean", IMAGENET_MEAN)), tuple(cfg.get("std", IMAGENET_STD))


def encode_target(y, target, scale=1.0):
    """'raw' (baseline, EDA kararı) veya 'log1p' (ablation); ardından scale'e bölünür."""
    z = torch.log1p(y) if target == "log1p" else y
    return z / torch.as_tensor(scale, dtype=z.dtype, device=z.device)


def decode_target(z, target, scale=1.0):
    z = z * torch.as_tensor(scale, dtype=z.dtype, device=z.device)
    return torch.expm1(z) if target == "log1p" else z


def save_checkpoint(path, model, backbone, outputs=("calories",), target="raw", scale=1.0,
                    mean=IMAGENET_MEAN, std=IMAGENET_STD, **info):
    """outputs: hedef adları (kalori) ya da sınıf adları (target='cls').
    info: ör. test_MAE=92.3 → demo bunu kullanıcıya 'tipik hata' olarak gösterir."""
    scale = scale.tolist() if hasattr(scale, "tolist") else scale
    torch.save({"backbone": backbone, "outputs": list(outputs), "target": target, "scale": scale,
                "mean": list(mean), "std": list(std), "state_dict": model.state_dict(), "info": info}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["backbone"], len(ckpt["outputs"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    meta.setdefault("scale", 1.0)
    meta.setdefault("mean", list(IMAGENET_MEAN))
    meta.setdefault("std", list(IMAGENET_STD))
    return model.to(device).eval(), meta


@torch.no_grad()
def predict_image(model, meta, img: Image.Image, device="cpu"):
    """Tek fotoğraf → {"calories": ..., ...}. Telefon fotoğrafları önce eğitimdeki 4:3 orana kırpılır."""
    img = crop_to_aspect(ImageOps.exif_transpose(img).convert("RGB"))
    x = calorie_transform(train=False, mean=meta["mean"], std=meta["std"])(img).unsqueeze(0).to(device)
    y = decode_target(model(x)[0].float().cpu(), meta["target"], meta["scale"])
    return dict(zip(meta["outputs"], y.clamp(min=0).tolist()))
