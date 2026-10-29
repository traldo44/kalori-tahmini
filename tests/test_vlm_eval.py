"""vlm_eval testleri: ağ yok, gerçek veri yok. HTTP fonksiyonu taklit edilir."""
import base64
import csv
import json

import pytest
from PIL import Image

from src import vlm_eval
from src.vlm_eval import (append_jsonl, build_request, call_vlm, cost_usd, encode_image, fake_call, load_done,
                          parse_response, sanitize)

B64 = base64.b64encode(b"fakeimage").decode()


# ---------------------------------------------------------------- ayrıştırma

def test_parse_clean_json():
    r = parse_response('{"items":[{"food":"rice","grams":150,"kcal":195}],"total_kcal":320}')
    assert r["total_kcal"] == 320 and r["items"][0]["food"] == "rice"


def test_parse_fenced_json():
    r = parse_response('```json\n{"items": [], "total_kcal": 410.5}\n```')
    assert r["total_kcal"] == 410.5


def test_parse_prose_around_json():
    text = 'Sure! Here is my estimate:\n{"items":[{"food":"egg","grams":50,"kcal":78}],"total_kcal":78}\nHope this helps.'
    assert parse_response(text)["total_kcal"] == 78


def test_parse_number_only_fallback_when_json_broken():
    assert parse_response('{"items":[{"food":"x","grams":1,"kcal":5}, "total_kcal": 250')["total_kcal"] == 250


def test_parse_total_from_items_when_total_missing():
    assert parse_response('{"items":[{"food":"a","kcal":100},{"food":"b","kcal":"50 kcal"}]}')["total_kcal"] == 150


def test_parse_broken_returns_none():
    assert parse_response("I cannot determine the calories from this image.") is None
    assert parse_response("") is None
    assert parse_response(None) is None
    assert parse_response('{"items": [], "total_kcal": "unknown"}') is None


# ---------------------------------------------------------------- istek kurma

def capture(monkeypatch, responses):
    """http_post_json'u taklit eder: her çağrıda sıradaki (durum, gövde) döner, çağrıları kaydeder."""
    calls = []

    def fake_http(url, payload, headers=None, timeout=120):
        calls.append({"url": url, "payload": payload, "headers": headers})
        return responses.pop(0)

    monkeypatch.setattr(vlm_eval, "http_post_json", fake_http)
    monkeypatch.setattr(vlm_eval, "sleep", lambda s: None)
    return calls


GEMINI_OK = (200, {"candidates": [{"content": {"parts": [{"text": '{"items":[],"total_kcal":300}'}]}}],
                   "usageMetadata": {"promptTokenCount": 280, "candidatesTokenCount": 30, "thoughtsTokenCount": 10}})
OPENAI_OK = (200, {"choices": [{"message": {"content": '{"items":[],"total_kcal":120}'}}],
                   "usage": {"prompt_tokens": 900, "completion_tokens": 25}})


def test_gemini_request_has_inline_image_and_temperature_zero(monkeypatch):
    calls = capture(monkeypatch, [GEMINI_OK])
    r = call_vlm("gemini", "gemini-2.5-flash", "SECRET", B64, "image/jpeg")
    c = calls[0]
    assert "models/gemini-2.5-flash:generateContent" in c["url"] and "key=SECRET" in c["url"]
    parts = c["payload"]["contents"][0]["parts"]
    assert parts[0]["text"] == vlm_eval.PROMPT
    assert parts[1]["inline_data"] == {"mime_type": "image/jpeg", "data": B64}
    assert c["payload"]["generationConfig"]["temperature"] == 0
    assert r["error"] is None and r["text"].endswith("300}")
    assert (r["prompt_tokens"], r["output_tokens"]) == (280, 40)   # düşünme tokenları çıkışa dahil


def test_openai_request_has_data_url_and_bearer(monkeypatch):
    calls = capture(monkeypatch, [OPENAI_OK])
    r = call_vlm("openai", "gpt-4o-mini", "SECRET", B64, "image/png", base_url="https://example.com/v1/")
    c = calls[0]
    assert c["url"] == "https://example.com/v1/chat/completions"
    assert c["headers"]["Authorization"] == "Bearer SECRET"
    assert c["payload"]["temperature"] == 0 and c["payload"]["model"] == "gpt-4o-mini"
    content = c["payload"]["messages"][0]["content"]
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{B64}"
    assert (r["prompt_tokens"], r["output_tokens"]) == (900, 25) and r["error"] is None


def test_unknown_provider():
    with pytest.raises(ValueError):
        build_request("anthropic", "m", "k", B64, "image/jpeg")


# ---------------------------------------------------------------- yeniden deneme

def test_retry_on_429_then_success(monkeypatch):
    calls = capture(monkeypatch, [(429, {"error": "quota"}), (503, "busy"), GEMINI_OK])
    r = call_vlm("gemini", "m", "k", B64, "image/jpeg")
    assert len(calls) == 3 and r["tries"] == 3 and r["error"] is None


def test_gives_up_after_max_tries(monkeypatch):
    calls = capture(monkeypatch, [(429, "a"), (429, "b")])
    r = call_vlm("gemini", "m", "k", B64, "image/jpeg", max_tries=2)
    assert len(calls) == 2 and r["text"] is None and r["error"].startswith("HTTP 429")


def test_no_retry_on_400(monkeypatch):
    calls = capture(monkeypatch, [(400, {"error": {"message": "bad key"}})])
    r = call_vlm("gemini", "m", "k", B64, "image/jpeg")
    assert len(calls) == 1 and "bad key" in r["error"]


# ---------------------------------------------------------------- kayıt, devam etme, maliyet

def test_resume_skips_done_but_retries_errors(tmp_path):
    p = tmp_path / "run.jsonl"
    assert load_done(p) == {}
    append_jsonl(p, {"dish_id": "dish_1", "error": None, "text": "{}", "parsed_total": None, "prompt_tokens": 1, "output_tokens": 1})
    append_jsonl(p, {"dish_id": "dish_2", "error": "HTTP 503: busy", "text": None})
    done = load_done(p)
    assert set(done) == {"dish_1"}   # ayrıştırılamayan ama yanıtı alınmış tabak bitmiş sayılır, HTTP hatası sayılmaz
    todo = [d for d in ["dish_1", "dish_2", "dish_3"] if d not in done]
    assert todo == ["dish_2", "dish_3"]


def test_cost():
    assert cost_usd(1_000_000, 500_000, 0.30, 2.50) == pytest.approx(0.30 + 1.25)
    assert cost_usd(10, 10, 0, 0) is None


def test_sanitize_model_name():
    assert sanitize("gemini-2.5-flash") == "gemini_2_5_flash"
    assert sanitize("Qwen/Qwen2.5-VL-7B") == "qwen_qwen2_5_vl_7b"


def test_encode_image_downscales_to_jpeg(tmp_path):
    path = tmp_path / "rgb.png"
    Image.new("RGB", (640, 480), (200, 100, 50)).save(path)
    b64, mime = encode_image(path, short_side=120)
    img = Image.open(__import__("io").BytesIO(base64.b64decode(b64)))
    assert mime == "image/jpeg" and img.format == "JPEG" and img.size == (160, 120)
    b64, mime = encode_image(path, short_side=0, fmt="png")
    assert mime == "image/png" and Image.open(__import__("io").BytesIO(base64.b64decode(b64))).size == (640, 480)


def test_fake_call_is_deterministic_and_has_unparsable_case():
    assert fake_call("dish_1556572657") == fake_call("dish_1556572657")
    assert parse_response(fake_call("dish_1556572657")["text"]) is not None
    assert parse_response(fake_call("dish_1556572661")["text"]) is None   # 1556572661 % 7 == 0


# ---------------------------------------------------------------- uçtan uca (sahte veri kökü, --dry-run)

def make_fake_n5k(root, train_ids, test_ids):
    (root / "metadata").mkdir(parents=True)
    (root / "dish_ids" / "splits").mkdir(parents=True)
    with open(root / "metadata" / "dish_metadata_cafe1.csv", "w", newline="") as f:
        w = csv.writer(f)
        for i, d in enumerate(train_ids + test_ids):
            w.writerow([d, 100 + 50 * i, 200, 5, 20, 10, "ingr_1", "rice", 100, 130, 1, 28, 2])
    for split, ids in (("train", train_ids), ("test", test_ids)):
        (root / "dish_ids" / "splits" / f"rgb_{split}_ids.txt").write_text("\n".join(ids) + "\n")
        for d in ids:
            (root / "realsense_overhead" / d).mkdir(parents=True)
            Image.new("RGB", (64, 48), (120, 90, 60)).save(root / "realsense_overhead" / d / "rgb.png")


def test_dry_run_end_to_end(tmp_path, capsys):
    root, out = tmp_path / "n5k", tmp_path / "results"
    train = [f"dish_{1556572657 + 4000 * i}" for i in range(12)]
    test = [f"dish_{1556572661 + i}" for i in range(6)]   # ilki 7'ye bölünür → sahte "ayrıştırılamayan" yanıt
    make_fake_n5k(root, train, test)
    vlm_eval.main(["--dry-run", "--root", str(root), "--out-dir", str(out), "--model", "fake-vlm", "--price-in", "1", "--price-out", "2"])
    lines = (out / "vlm" / "vlm_fake_vlm.jsonl").read_text().splitlines()
    assert len(lines) == 6 and sum(json.loads(ln)["parsed_total"] is None for ln in lines) == 1
    with open(out / "preds" / "vlm_fake_vlm.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["id"] for r in rows] == test[1:]   # ayrıştırılamayan tabak atıldı
    assert all(float(r["y_pred"]) > 0 for r in rows)
    txt = capsys.readouterr().out
    assert "ayrıştırılamayan 1 (%16.7)" in txt and "results.csv'ye yazılmadı" in txt
    assert "maliyet $" in txt
    # ikinci çalıştırma: hepsi bitmiş, jsonl büyümez
    vlm_eval.main(["--dry-run", "--root", str(root), "--out-dir", str(out), "--model", "fake-vlm"])
    assert len((out / "vlm" / "vlm_fake_vlm.jsonl").read_text().splitlines()) == 6
    assert "6 önceden bitmiş, 0 sorulacak" in capsys.readouterr().out
