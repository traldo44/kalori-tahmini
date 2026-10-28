"""Checkpoint paylaşımı: büyük model dosyaları git'e değil, repo'nun "checkpoints" Release'ine yüklenir.

Analiz ve tablolar için checkpoint gerekmez; results/preds/*.csv yeter ve onlar git'te.
Checkpoint yalnızca final model (demo) ve Grad-CAM için paylaşılır.

    export GH_TOKEN=...      # Kaggle'da: Add-ons → Secrets (bkz. README §5.1)
    python scripts/share_checkpoint.py upload checkpoints/resnet50_reg.pt
    python scripts/share_checkpoint.py download resnet50_reg.pt
    python scripts/share_checkpoint.py list
"""
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
TAG = "checkpoints"
OUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def repo_slug():
    """origin adresinden 'kullanici/repo' (token'lı https adresleri dahil)."""
    url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True).stdout
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


def get_release(repo, create=False):
    try:
        return call("GET", f"{API}/repos/{repo}/releases/tags/{TAG}")
    except urllib.error.HTTPError as e:
        if e.code != 404 or not create:
            raise
    body = {"tag_name": TAG, "name": "Checkpoints", "prerelease": True,
            "body": "Model dosyaları. scripts/share_checkpoint.py ile yükleyin/indirin."}
    return call("POST", f"{API}/repos/{repo}/releases", json.dumps(body).encode())


def upload(repo, path):
    path = Path(path)
    release = get_release(repo, create=True)
    for asset in release["assets"]:
        if asset["name"] == path.name:  # aynı adlı eski sürümü değiştir
            call("DELETE", f"{API}/repos/{repo}/releases/assets/{asset['id']}")
    upload_url = release["upload_url"].split("{")[0] + f"?name={path.name}"
    call("POST", upload_url, path.read_bytes(), content_type="application/octet-stream")
    print(f"Yüklendi: {path.name} ({path.stat().st_size / 1e6:.0f} MB)")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def download(repo, name):
    asset = next((a for a in get_release(repo)["assets"] if a["name"] == name), None)
    if asset is None:
        sys.exit(f"'{name}' bulunamadı. Mevcut dosyalar için: python scripts/share_checkpoint.py list")
    # GitHub dosyayı geçici bir adrese yönlendirir; o adrese token'sız gidilmeli.
    req = urllib.request.Request(f"{API}/repos/{repo}/releases/assets/{asset['id']}", headers={
        "Authorization": f"Bearer {os.environ['GH_TOKEN']}", "Accept": "application/octet-stream"})
    try:
        urllib.request.build_opener(_NoRedirect).open(req)
        sys.exit("Beklenmeyen yanıt: yönlendirme gelmedi.")
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 307):
            raise
        location = e.headers["Location"]
    OUT_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(location, OUT_DIR / name)
    print(f"İndirildi: checkpoints/{name}")


def list_assets(repo):
    for a in get_release(repo)["assets"]:
        print(f"{a['name']:<40} {a['size'] / 1e6:>6.0f} MB  {a['updated_at'][:10]}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("upload", "download", "list") or "GH_TOKEN" not in os.environ:
        sys.exit(__doc__)
    repo = repo_slug()
    {"upload": lambda: upload(repo, sys.argv[2]),
     "download": lambda: download(repo, sys.argv[2]),
     "list": lambda: list_assets(repo)}[sys.argv[1]]()
