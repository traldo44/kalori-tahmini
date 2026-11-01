"""Demoyu Hugging Face Space olarak yayınla / güncelle (Adım 19).

    python scripts/deploy_space.py                 # traldo/kalori-tahmini Space'ini oluşturur veya günceller
    python scripts/deploy_space.py --no-ckpt       # yalnızca kod/README değişti, 111 MB checkpoint'i yeniden yükleme
    python scripts/deploy_space.py --repo ad/space # başka hesap/ad

Gereken: `hf auth login` yapılmış olması (token ~/.cache/huggingface/token). Yüklenenler:
app.py, src/{__init__,config,transforms,models}.py, demo_ornekleri/, space/README.md → README.md,
space/requirements.txt → requirements.txt, checkpoints/convnext_tiny_reg_mt.pt (LFS).
Space ücretsiz CPU'da çalışır; ilk kurulum (torch CPU) 3–5 dk sürer.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "convnext_tiny_reg_mt.pt"
CODE_FILES = ["app.py", "src/__init__.py", "src/config.py", "src/transforms.py", "src/models.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="traldo/kalori-tahmini")
    ap.add_argument("--no-ckpt", action="store_true", help="checkpoint'i yeniden yükleme")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    me = api.whoami()["name"]
    url = api.create_repo(args.repo, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    print(f"Space: {url}  (hesap: {me})")

    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(ROOT / f)) for f in CODE_FILES]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(ROOT / "space" / "README.md")))
    ops.append(CommitOperationAdd(path_in_repo="requirements.txt", path_or_fileobj=str(ROOT / "space" / "requirements.txt")))
    for p in sorted((ROOT / "demo_ornekleri").iterdir()):
        if p.suffix in {".jpg", ".csv"}:
            ops.append(CommitOperationAdd(path_in_repo=f"demo_ornekleri/{p.name}", path_or_fileobj=str(p)))
    if not args.no_ckpt:
        if not CKPT.exists():
            sys.exit(f"{CKPT} yok. GH_TOKEN=$(gh auth token) python scripts/share_checkpoint.py download {CKPT.name}")
        ops.append(CommitOperationAdd(path_in_repo=f"checkpoints/{CKPT.name}", path_or_fileobj=str(CKPT)))
        print(f"checkpoint {CKPT.stat().st_size / 1e6:.0f} MB yüklenecek")

    api.create_commit(args.repo, repo_type="space", operations=ops,
                      commit_message="deploy: " + ("kod + checkpoint" if not args.no_ckpt else "kod"))
    print(f"Gönderildi: {len(ops)} dosya → https://huggingface.co/spaces/{args.repo}")
    print("Space birkaç dakika içinde kurulur; durumu sayfanın üstündeki 'Building/Running' etiketinden izleyin.")


if __name__ == "__main__":
    main()
