import torch
from PIL import Image

from src.models import build_regressor, decode_target, encode_target, load_checkpoint, predict_image, save_checkpoint


def test_target_roundtrip():
    y = torch.tensor([0.0, 120.0, 950.0])
    assert torch.allclose(decode_target(encode_target(y, "log1p"), "log1p"), y, atol=1e-3)


def test_checkpoint_roundtrip_and_phone_photo_prediction(tmp_path):
    torch.manual_seed(0)
    model = build_regressor("resnet18", n_outputs=2, pretrained=False)
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, "resnet18", outputs=("calories", "mass"), target="log1p", test_MAE=90.0)
    loaded, meta = load_checkpoint(path)
    assert meta["outputs"] == ["calories", "mass"] and meta["info"]["test_MAE"] == 90.0
    phone_photo = Image.new("RGB", (3024, 4032), (120, 90, 60))   # dikey telefon fotoğrafı
    pred = predict_image(loaded, meta, phone_photo)
    assert set(pred) == {"calories", "mass"} and pred["calories"] > -1   # expm1 çıktısı


def test_regression_head_starts_at_train_mean():
    from src.models import init_regression_head
    model = init_regression_head(build_regressor("efficientnet_b0", n_outputs=3, pretrained=False)).eval()
    out = model(torch.randn(2, 3, 64, 64))
    assert torch.equal(out, torch.ones(2, 3))   # ölçeklenmiş hedefte 1 = train ortalaması
