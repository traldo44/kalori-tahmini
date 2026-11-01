"""Final checkpoint'i tarayıcıda çalışacak ONNX modeline çevir (Adım 19, web/ demosu → Vercel).

    python scripts/export_onnx.py                       # checkpoints/convnext_tiny_reg_mt.pt → web/model.onnx (int8, ~35 MB)
    python scripts/export_onnx.py --no-quant            # fp32 (111 MB; yalnızca karşılaştırma için)
    python scripts/export_onnx.py --check 100           # 100 test tabağında torch / fp32 ONNX / int8 ONNX farkını yaz

Model grafiğine ön işleme (ImageNet normalizasyonu) ve son işleme (ölçek, ≥ 0) gömülü:
girdi  image  float32 [1, 3, 224, 224], değerler 0–1 (RGB, 224×224'e yeniden boyutlanmış 4:3 kırpma)
çıktı  pred   float32 [1, 5] = kalori (kcal), kütle (g), yağ (g), karbonhidrat (g), protein (g)
Ağırlıklar dinamik int8 (MatMul/Gemm); doğruluk farkı --check ile ölçülür ve web/model.json'a yazılır.
"""
import argparse
import json
import time
from pathlib import Path

import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src.models import load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


class Wrapped(torch.nn.Module):
    def __init__(self, model, meta):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(meta["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(meta["std"]).view(1, 3, 1, 1))
        scale = meta["scale"] if isinstance(meta["scale"], list) else [meta["scale"]] * len(meta["outputs"])
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32).view(1, -1))
        self.log1p = meta["target"] == "log1p"

    def forward(self, image):
        z = self.model((image - self.mean) / self.std) * self.scale
        if self.log1p:
            z = torch.expm1(z)
        return z.clamp(min=0)


def test_images(n, seed=0):
    """Test kümesinden n tabak: 0–1 tensörü (kırpma + 224) ve gerçek kalori."""
    import pandas as pd
    from PIL import Image
    from src.config import N5K_DIR
    from src.transforms import calorie_transform, crop_to_aspect
    preds = pd.read_csv(ROOT / "results/preds/convnext_tiny_reg_mt.csv").sample(n, random_state=seed)
    tf = calorie_transform(train=False, mean=(0, 0, 0), std=(1, 1, 1))  # yalnızca resize + ToTensor
    xs, ys = [], []
    for d, y in zip(preds["id"], preds["y_true"]):
        img = Image.open(N5K_DIR / "realsense_overhead" / d / "rgb.png").convert("RGB")
        xs.append(tf(crop_to_aspect(img)).numpy())
        ys.append(y)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)


def run_onnx(path, xs):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    t = time.time()
    out = np.concatenate([sess.run(None, {"image": x[None]})[0] for x in xs])
    return out, (time.time() - t) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/convnext_tiny_reg_mt.pt"))
    ap.add_argument("--out", default=str(WEB / "model.onnx"))
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--check", type=int, default=50, help="kaç test tabağında karşılaştırılsın (0 = atla)")
    args = ap.parse_args()
    WEB.mkdir(exist_ok=True)

    model, meta = load_checkpoint(args.ckpt, "cpu")
    wrapped = Wrapped(model, meta).eval()
    fp32_path = Path(args.out).with_suffix(".fp32.onnx")
    torch.onnx.export(wrapped, torch.rand(1, 3, 224, 224), str(fp32_path), dynamo=False, opset_version=17,
                      input_names=["image"], output_names=["pred"], do_constant_folding=True)
    import onnx
    onnx.checker.check_model(str(fp32_path))
    print(f"fp32 ONNX: {fp32_path.name} {fp32_path.stat().st_size / 1e6:.1f} MB")

    out_path = Path(args.out)
    if args.no_quant:
        fp32_path.replace(out_path)
    else:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8)
        print(f"int8 ONNX: {out_path.name} {out_path.stat().st_size / 1e6:.1f} MB")

    info = {"outputs": meta["outputs"], "input": "image float32 [1,3,224,224] 0-1 RGB", "size_mb": round(out_path.stat().st_size / 1e6, 1),
            "quant": "int8 dynamic" if not args.no_quant else "fp32", "test_MAE_torch": meta["info"].get("test_MAE")}
    if args.check:
        xs, ys = test_images(args.check)
        with torch.no_grad():
            pt = wrapped(torch.from_numpy(xs)).numpy()
        if fp32_path.exists():
            o32, _ = run_onnx(fp32_path, xs)
            print(f"torch ↔ fp32 ONNX: kcal ort. |fark| {np.abs(pt[:, 0] - o32[:, 0]).mean():.2f}")
        oq, sec = run_onnx(out_path, xs)
        info.update({"check_n": int(args.check),
                     "kcal_abs_diff_vs_torch": round(float(np.abs(pt[:, 0] - oq[:, 0]).mean()), 2),
                     "MAE_torch_subset": round(float(np.abs(pt[:, 0] - ys).mean()), 1),
                     "MAE_onnx_subset": round(float(np.abs(oq[:, 0] - ys).mean()), 1),
                     "ms_per_image_cpu": round(1000 * sec, 1)})
        print(f"torch ↔ {info['quant']} ONNX: kcal ort. |fark| {info['kcal_abs_diff_vs_torch']} · "
              f"MAE aynı {args.check} tabakta torch {info['MAE_torch_subset']} / onnx {info['MAE_onnx_subset']} · {info['ms_per_image_cpu']} ms/görüntü")
    if not args.no_quant and fp32_path.exists():
        fp32_path.unlink()
    (WEB / "model.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out_path}  ·  {WEB / 'model.json'}")


if __name__ == "__main__":
    main()
