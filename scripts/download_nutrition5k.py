"""Nutrition5k'dan yalnızca gerekenleri indirir: metadata, bölmeler ve üstten RGB görüntüler.

gsutil gerekmez; herkese açık bucket'tan HTTPS ile indirir. Mac'te ve Kaggle'da aynı çalışır.
Yarıda kesilirse yeniden çalıştırın; inmiş dosyaları atlar.

    python scripts/download_nutrition5k.py                 # her şey (~1,4 GB)
    python scripts/download_nutrition5k.py --no-images     # yalnızca metadata + bölmeler
    python scripts/download_nutrition5k.py --out /kaggle/working/nutrition5k
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

BUCKET = "nutrition5k_dataset"
PREFIX = "nutrition5k_dataset"
HTTP = f"https://storage.googleapis.com/{BUCKET}/{PREFIX}"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"

SMALL_FILES = [
    "metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv",
    "metadata/ingredients_metadata.csv",
    "dish_ids/README",
    "dish_ids/dish_ids_cafe1.txt",
    "dish_ids/splits/rgb_train_ids.txt",
    "dish_ids/splits/rgb_test_ids.txt",
    "dish_ids/splits/depth_train_ids.txt",
    "dish_ids/splits/depth_test_ids.txt",
]


def fetch(url, dest, retries=3):
    """Dosyayı indirir; zaten varsa atlar. İndirildiyse True döner."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            return True
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def list_overhead_dishes():
    """Bucket'ta realsense_overhead klasörü olan tabak kimlikleri (≈3490)."""
    dishes, token = [], None
    while True:
        q = {"prefix": f"{PREFIX}/imagery/realsense_overhead/", "delimiter": "/",
             "fields": "prefixes,nextPageToken", "maxResults": "1000"}
        if token:
            q["pageToken"] = token
        page = json.load(urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}"))
        dishes += [p.rstrip("/").split("/")[-1] for p in page.get("prefixes", [])]
        token = page.get("nextPageToken")
        if not token:
            return sorted(dishes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/nutrition5k")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    for f in SMALL_FILES:
        fetch(f"{HTTP}/{f}", out / f)
    dishes = list_overhead_dishes()
    (out / "dish_ids/overhead_available.txt").write_text("\n".join(dishes) + "\n")
    print(f"Metadata ve bölmeler hazır. Üstten görüntüsü olan tabak: {len(dishes)}")
    if args.no_images:
        return

    def get(dish):
        return fetch(f"{HTTP}/imagery/realsense_overhead/{dish}/rgb.png",
                     out / "realsense_overhead" / dish / "rgb.png")

    with ThreadPoolExecutor(args.workers) as ex:
        new = sum(tqdm(ex.map(get, dishes), total=len(dishes), desc="rgb.png"))
    print(f"{new} yeni görüntü indirildi, {len(dishes) - new} zaten vardı.")


if __name__ == "__main__":
    main()
