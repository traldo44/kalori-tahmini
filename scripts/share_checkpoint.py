"""Büyük dosya paylaşımı: dosyalar git'e değil, repo'nun GitHub Release'lerine yüklenir.

Varsayılan "checkpoints" Release'i: model dosyaları → checkpoints/ klasörü.
Analiz ve tablolar için checkpoint gerekmez; results/preds/*.csv yeter ve onlar git'te.
Checkpoint yalnızca final model (demo) ve Grad-CAM için paylaşılır.
"data" Release'i: veri arşivleri (ör. Food-101 alt kümesi, bkz. scripts/get_food101.py).

    export GH_TOKEN=...      # Kaggle'da: Add-ons → Secrets (bkz. README §5.1)
    python scripts/share_checkpoint.py upload checkpoints/resnet50_reg.pt
    python scripts/share_checkpoint.py download resnet50_reg.pt
    python scripts/share_checkpoint.py list
    python scripts/share_checkpoint.py list --tag data
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

API ="https://api.github.com"
ROOT = Path(__file__).resolve().parent.parent
RELEASE_NOTES = {
    "checkpoints": "Model dosyaları. scripts/share_checkpoint.py ile yükleyin/indirin.",
    "data": "Veri arşivleri. scripts/get_food101.py fetch ile indirin.",
}


def repo_slug():
    """origin adresinden 'kullanici/repo' (token'lı https adresleri dahil)."""
    url = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"],
                         capture_output=True, text=True, check=True).stdout
    return re.search(r"github\.com[:/](.+?)(?:\.git)?$", url.strip()).group(1)


def call(method, url, data=None, content_type="application/json"):
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "Content-Type": content_type,
    })
    with urllib.request.urlopen(req) as r:
        body = r.read()
    return json.loads(body) if body else None


def get_release(repo, tag, create=False):
    try:
        return call("GET", f"{API}/repos/{repo}/releases/tags/{tag}")
    except urllib.error.HTTPError as e:
        if e.code != 404 or not create:
            raise
    body = {"tag_name": tag, "name": tag.capitalize(), "prerelease": True,
            "body": RELEASE_NOTES.get(tag, "")}
    return call("POST", f"{API}/repos/{repo}/releases", json.dumps(body).encode())


def upload(repo, path, tag="checkpoints"):
    path = Path(path)
    release = get_release(repo, tag, create=True)
    for asset in release["assets"]:
        if asset["name"] == path.name:  # aynı adlı eski sürümü değiştir
            call("DELETE", f"{API}/repos/{repo}/releases/assets/{asset['id']}")
    upload_url = release["upload_url"].split("{")[0] + f"?name={path.name}"
    call("POST", upload_url, path.read_bytes(), content_type="application/octet-stream")
    print(f"Yüklendi: {path.name} ({path.stat().st_size / 1e6:.0f} MB) → '{tag}' Release")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _signed_url(repo, asset_id):
    """GitHub dosyayı birkaç dakika geçerli, imzalı bir adrese yönlendirir; o adrese token'sız gidilmeli."""
    req = urllib.request.Request(f"{API}/repos/{repo}/releases/assets/{asset_id}", headers={
        "Authorization": f"Bearer {os.environ['GH_TOKEN']}", "Accept": "application/octet-stream"})
    try:
        urllib.request.build_opener(_NoRedirect).open(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 307):
            return e.headers["Location"]
        raise
    sys.exit("Beklenmeyen yanıt: yönlendirme gelmedi.")


def download(repo, name, out_dir=ROOT / "checkpoints", tag="checkpoints", retries=5):
    """Bağlantı 60 sn veri göndermezse kesilir ve kaldığı yerden (HTTP Range) yeniden denenir."""
    asset = next((a for a in get_release(repo, tag)["assets"] if a["name"] == name), None)
    if asset is None:
        sys.exit(f"'{name}' bulunamadı. Mevcut dosyalar için: python scripts/share_checkpoint.py list --tag {tag}")
    out = Path(out_dir) / name
    part = out.with_name(out.name + ".part")
    out.parent.mkdir(parents=True, exist_ok=True)
    size = asset["size"]
    with tqdm(total=size, unit="B", unit_scale=True, desc=name) as bar:
        for attempt in range(1, retries + 1):
            have = part.stat().st_size if part.exists() else 0
            bar.n = have
            bar.refresh()
            if have == size:
                break
            req = urllib.request.Request(_signed_url(repo, asset["id"]), headers={"Range": f"bytes={have}-"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if r.status == 206 else "wb") as f:
                    if r.status != 206:  # sunucu kaldığı yerden devamı kabul etmedi → baştan
                        bar.n = 0
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            except (OSError, urllib.error.URLError) as e:
                bar.write(f"Bağlantı koptu ({e}); kaldığı yerden yeniden deneniyor ({attempt}/{retries})")
    if not part.exists() or part.stat().st_size != size:
        sys.exit(f"İndirme tamamlanamadı: {name}. Komutu tekrar çalıştırın, kaldığı yerden devam eder.")
    part.rename(out)
    print(f"İndirildi: {out}")
    return out


def list_assets(repo, tag="checkpoints"):
    for a in get_release(repo, tag)["assets"]:
        print(f"{a['name']:<40} {a['size'] / 1e6:>6.0f} MB  {a['updated_at'][:10]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["upload", "download", "list"])
    ap.add_argument("name", nargs="?", help="upload: dosya yolu · download: dosya adı")
    ap.add_argument("--tag", default="checkpoints", help="Release adı (checkpoints | data)")
    ap.add_argument("--out", default=str(ROOT / "checkpoints"), help="download: hedef klasör")
    args = ap.parse_args()
    if "GH_TOKEN" not in os.environ:
        sys.exit("GH_TOKEN ortam değişkeni yok.\n\n" + __doc__)
    if args.action != "list" and not args.name:
        ap.error(f"{args.action} için dosya adı gerekli")
    repo = repo_slug()
    if args.action == "upload":
        upload(repo, args.name, args.tag)
    elif args.action == "download":
        download(repo, args.name, args.out, args.tag)
    else:
        list_assets(repo, args.tag)


if __name__ == "__main__":
    main()
