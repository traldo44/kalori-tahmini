"""Food-101'in 25 sınıflık alt kümesi (src/data.py → FOOD101_CLASSES).

Tam Food-101 5 GB; bize yalnızca 25 sınıf lazım. Alt küme bir kez çıkarılır, görüntülerin kısa kenarı
256 px'e küçültülür (eğitim 224 px kullanıyor; okuma çok hızlanır) ve repo'nun "data" Release'ine konur.

Herkes (Mac veya Kaggle) yalnızca alt kümeyi indirir:
    python scripts/get_food101.py fetch                          # → data/food101/
    python scripts/get_food101.py fetch --out /tmp/kalori-data   # Kaggle

Bir kez yapılır (tam Food-101 data/food101_raw/ altında olmalı; torchvision Food101(download=True)):
    python scripts/get_food101.py build --upload
"""
import argparse
import json
import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.share_checkpoint import download, repo_slug, upload  # noqa: E402
from src.data import FOOD101_CLASSES  # noqa: E402

ARCHIVE = "food101_subset25_256.tar"
SHORT_SIDE = 256


def shrink(src, dst):
    """Kısa kenarı SHORT_SIDE'a indirir (büyütmez), RGB JPEG olarak kaydeder. Orijinal (genişlik, yükseklik, mod) döner."""
    with Image.open(src) as im:
        info = (*im.size, im.mode)
        im = im.convert("RGB")
        scale = SHORT_SIDE / min(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=90)
    return info


def build(raw_root, out_root):
    src, dst = Path(raw_root) / "food-101", Path(out_root) / "food-101"
    jobs = []
    for split in ("train", "test"):
        with open(src / "meta" / f"{split}.json") as f:
            meta = {c: v for c, v in json.load(f).items() if c in FOOD101_CLASSES}
        assert set(meta) == set(FOOD101_CLASSES), "Sınıf listesinde Food-101'de olmayan ad var"
        (dst / "meta").mkdir(parents=True, exist_ok=True)
        with open(dst / "meta" / f"{split}.json", "w") as f:
            json.dump(meta, f)
        jobs += [n for names in meta.values() for n in names]
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        infos = list(tqdm(ex.map(lambda n: shrink(src / "images" / f"{n}.jpg", dst / "images" / f"{n}.jpg"), jobs),
                          total=len(jobs), desc="küçültülüyor"))
    originals = dict(zip(jobs, infos))
    # EDA için orijinal boyut ve renk modu (küçültülmüş kopyada bu bilgi kaybolur)
    with open(dst / "meta" / "original_sizes.json", "w") as f:
        json.dump(originals, f)
    archive = Path(out_root) / ARCHIVE
    with tarfile.open(archive, "w") as tar:
        tar.add(dst, arcname="food-101")
    print(f"{len(jobs)} görüntü → {archive} ({archive.stat().st_size / 1e6:.0f} MB)")
    return archive


def fetch(out_root):
    out_root = Path(out_root)
    if (out_root / "food-101" / "meta" / "train.json").exists():
        print(f"Zaten var: {out_root / 'food-101'}")
        return
    archive = download(repo_slug(), ARCHIVE, out_root, tag="data")
    with tarfile.open(archive) as tar:
        tar.extractall(out_root, filter="data")
    archive.unlink()
    print(f"Hazır: {out_root / 'food-101'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["build", "fetch"])
    ap.add_argument("--raw", default=str(ROOT / "data" / "food101_raw"), help="build: tam Food-101'in kökü")
    ap.add_argument("--out", default=None, help="varsayılan: $KALORI_DATA/food101 veya data/food101")
    ap.add_argument("--upload", action="store_true", help="build: arşivi 'data' Release'ine yükle")
    args = ap.parse_args()
    out = Path(args.out) / "food101" if args.out else Path(os.environ.get("KALORI_DATA", ROOT / "data")) / "food101"
    if args.action == "build":
        archive = build(args.raw, out)
        if args.upload:
            upload(repo_slug(), archive, tag="data")
    else:
        if "GH_TOKEN" not in os.environ:
            sys.exit("GH_TOKEN ortam değişkeni yok (bkz. README §5.1)")
        fetch(out)


if __name__ == "__main__":
    main()
