"""Adım 14 — VLM sıfır-atış referansı (§7): hazır bir görü-dil modeline Nutrition5k test tabaklarının kalorisini sordurur.

    GEMINI_API_KEY=... python -m src.vlm_eval                                  # 507 test tabağı, gemini-2.5-flash
    GEMINI_API_KEY=... python -m src.vlm_eval --n 300 --seed 42 --rpm 10       # maliyet sorunsa rastgele 300 tabak
    OPENAI_API_KEY=... python -m src.vlm_eval --provider openai --model gpt-4o-mini --base-url https://api.openai.com/v1
    python -m src.vlm_eval --dry-run --n 5                                     # ağ yok, sahte yanıt; boru hattını dener

Eğitim yok: sabit prompt, sıcaklık 0, JSON yanıt (önce görünen her yiyeceğin gramı, sonra toplam kcal).
Ham yanıtlar results/vlm/<run>.jsonl'a satır satır yazılır; betik yeniden çalışınca bitmiş tabaklar atlanır
(kesinti/kota sorunu olursa aynı komutu tekrar verin). Sonunda:
  - results/preds/<run>.csv  (id, y_true, y_pred; diğer modellerle aynı format, ayrıştırılamayanlar ATILIR)
  - results/results.csv'ye "calorie" ve "calorie_bins" satırı (notlarda model, tabak sayısı, hatalı yanıt oranı,
    token toplamı, maliyet ve "ayrıştırılamayan → train ortalaması" ile ikinci MAE)
API anahtarı YALNIZCA ortam değişkeninden okunur (GEMINI_API_KEY / OPENAI_API_KEY); repoya asla yazmayın.
Tüm HTTP trafiği tek fonksiyondan (`http_post_json`) geçer; testler onu taklit eder, ağ gerekmez.
"""
import argparse
import base64
import io
import json
import os
import random
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CALORIE_BIN_EDGES, N5K_DIR, RESULTS_DIR, SEED
from .data import load_n5k_split, make_val_split, read_dish_metadata
from .metrics import classification_metrics, regression_metrics, to_bins
from .results import log_result, save_preds

# Prompt İngilizce: VLM'ler İngilizce talimatla daha kararlı JSON üretiyor. Değiştirirseniz run adını da değiştirin.
PROMPT = (
    "You are a nutrition expert. The photo shows a single plate of food taken from directly above.\n"
    "Step 1: list every visible food item with an estimated weight in grams and its calories.\n"
    "Step 2: give the total calories of the whole plate in kcal.\n"
    "Answer with STRICT JSON only, no markdown, no explanation, exactly this shape:\n"
    '{"items":[{"food":"...","grams":0,"kcal":0}],"total_kcal":0}'
)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OPENAI_BASE_URL = "https://api.openai.com/v1"
RETRY_STATUS = {0, 408, 429, 500, 502, 503, 504}   # 0 = bağlantı hatası (DNS, zaman aşımı)

sleep = time.sleep   # testler bekleme süresini kapatmak için bunu değiştirir


# ---------------------------------------------------------------- görüntü

def encode_image(path, short_side=512, quality=85, fmt="jpeg"):
    """Görüntüyü küçültüp (kısa kenar ≤ short_side, yalnızca küçültme) base64 döner: (b64 metin, mime türü)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if short_side and min(w, h) > short_side:
        s = short_side / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG")
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    return base64.b64encode(buf.getvalue()).decode("ascii"), mime


# ---------------------------------------------------------------- HTTP (tek nokta)

def http_post_json(url, payload, headers=None, timeout=120):
    """JSON POST; (HTTP durum kodu, gövde) döner. Hata gövdeleri de döner, istisna fırlatmaz.
    Bağlantı kurulamazsa durum 0'dır. Testler bu fonksiyonu taklit eder."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            body = json.loads(body)
        except ValueError:
            pass
        return e.code, body
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, str(e)


# ---------------------------------------------------------------- istek kurma / yanıt çözme

def build_request(provider, model, api_key, image_b64, mime, base_url=None, thinking_budget=None):
    """(url, başlıklar, gövde). Gemini: inline_data; OpenAI uyumlu: data URL'li image_url. Sıcaklık 0."""
    if provider == "gemini":
        url = GEMINI_URL.format(model=model) + "?key=" + api_key
        gen = {"temperature": 0, "responseMimeType": "application/json"}
        if thinking_budget is not None:
            gen["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        payload = {"contents": [{"parts": [{"text": PROMPT}, {"inline_data": {"mime_type": mime, "data": image_b64}}]}],
                   "generationConfig": gen}
        return url, {}, payload
    if provider == "openai":
        url = (base_url or OPENAI_BASE_URL).rstrip("/") + "/chat/completions"
        payload = {"model": model, "temperature": 0,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": PROMPT},
                       {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}]}]}
        return url, {"Authorization": f"Bearer {api_key}"}, payload
    raise ValueError(f"bilinmeyen sağlayıcı: {provider}")


def extract_response(provider, body):
    """Yanıt gövdesinden (metin, giriş token, çıkış token) çıkarır. Çıkış tokenlarına düşünme tokenları dahildir."""
    if provider == "gemini":
        parts = body["candidates"][0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        u = body.get("usageMetadata", {})
        return text, int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0))
    text = body["choices"][0]["message"].get("content") or ""
    u = body.get("usage", {})
    return text, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))


def call_vlm(provider, model, api_key, image_b64, mime, base_url=None, thinking_budget=None, max_tries=5, timeout=120):
    """Bir görüntü için modeli çağırır; 429/5xx'te üstel bekleme ile en çok max_tries dener.
    Döner: {text, prompt_tokens, output_tokens, latency_s, tries, error}. error None değilse text None."""
    url, headers, payload = build_request(provider, model, api_key, image_b64, mime, base_url, thinking_budget)
    status, body, latency = None, None, 0.0
    for attempt in range(1, max_tries + 1):
        t0 = time.time()
        status, body = http_post_json(url, payload, headers, timeout)
        latency = time.time() - t0
        if status == 200:
            try:
                text, p_tok, o_tok = extract_response(provider, body)
            except (KeyError, IndexError, TypeError) as e:
                return {"text": None, "prompt_tokens": 0, "output_tokens": 0, "latency_s": latency, "tries": attempt,
                        "error": f"beklenmeyen yanıt yapısı ({e}): {str(body)[:300]}"}
            return {"text": text, "prompt_tokens": p_tok, "output_tokens": o_tok, "latency_s": latency,
                    "tries": attempt, "error": None}
        if status not in RETRY_STATUS or attempt == max_tries:
            break
        sleep(min(60, 2 ** attempt))   # 2, 4, 8, 16 s
    return {"text": None, "prompt_tokens": 0, "output_tokens": 0, "latency_s": latency, "tries": attempt,
            "error": f"HTTP {status}: {str(body)[:300]}"}


# ---------------------------------------------------------------- ayrıştırma

def _number(v):
    """350, 350.0, "350", "350 kcal" → 350.0; olmazsa None."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if np.isfinite(v) else None
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(v))
    return float(m.group().replace(",", ".")) if m else None


def parse_response(text):
    """Model metninden {"total_kcal": float, "items": [...]} çıkarır; olmazsa None (= ayrıştırılamayan yanıt).

    Sırasıyla: ```json çitlerini at → ilk {...} bloğunu JSON olarak oku (toplam yoksa kalemlerin kcal'ini topla)
    → son çare: "total_kcal" sözcüğünden sonraki ilk sayı.
    """
    if not text:
        return None
    s = re.sub(r"```[a-zA-Z]*", "", text).strip()
    i = s.find("{")
    if i != -1:
        obj = None
        try:
            obj, _ = json.JSONDecoder().raw_decode(s[i:])   # bloktan sonraki düz yazıyı yok sayar
        except ValueError:
            j = s.rfind("}")
            if j > i:
                try:
                    obj = json.loads(s[i:j + 1])
                except ValueError:
                    obj = None
        if isinstance(obj, dict):
            items = obj.get("items") if isinstance(obj.get("items"), list) else []
            total = _number(obj.get("total_kcal"))
            if total is None and items:
                kcals = [_number(it.get("kcal")) for it in items if isinstance(it, dict)]
                if kcals and all(k is not None for k in kcals):
                    total = float(sum(kcals))
            if total is not None:
                return {"total_kcal": total, "items": items}
    m = re.search(r"total_kcal\W{0,5}(-?\d+(?:[.,]\d+)?)", s)
    if m:
        return {"total_kcal": float(m.group(1).replace(",", ".")), "items": []}
    return None


# ---------------------------------------------------------------- kayıt (jsonl) ve maliyet

def load_done(jsonl_path):
    """Önceki çalıştırmadan yanıt alınmış tabaklar: {dish_id: kayıt}. HTTP hatası olanlar yeniden denenecek, alınmaz."""
    done = {}
    if not Path(jsonl_path).exists():
        return done
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("error") is None:
                    done[rec["dish_id"]] = rec
    return done


def append_jsonl(jsonl_path, rec):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cost_usd(prompt_tokens, output_tokens, price_in, price_out):
    """Fiyatlar USD / 1M token. İkisi de 0 ise None (fiyat bilinmiyor)."""
    if not price_in and not price_out:
        return None
    return prompt_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out


def sanitize(model):
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def fake_call(dish_id):
    """--dry-run için ağsız, deterministik sahte yanıt (her 7. tabakta bilerek ayrıştırılamaz metin)."""
    t = int(dish_id.split("_")[1])
    if t % 7 == 0:
        text = "I cannot determine the calories from this image."
    else:
        kcal = 80 + t % 500
        text = ("```json\n" + json.dumps({"items": [{"food": "rice", "grams": kcal // 2, "kcal": kcal}],
                                           "total_kcal": kcal}) + "\n```")
    return {"text": text, "prompt_tokens": 300, "output_tokens": 40, "latency_s": 0.0, "tries": 1, "error": None}


# ---------------------------------------------------------------- değerlendirme

def summarize(records, ids, meta, train_mean):
    """Kayıtlardan ana metrikleri (ayrıştırılamayanlar atılmış) ve yedek metrikleri (train ortalaması) hesaplar."""
    y_all = np.array([meta[d]["calories"] for d in ids])
    pred = np.array([records[d]["parsed_total"] if d in records and records[d].get("parsed_total") is not None
                     else np.nan for d in ids])
    n_error = sum(d not in records for d in ids)
    n_bad = int(np.isnan(pred).sum()) - n_error
    ok = ~np.isnan(pred)
    out = {"n": len(ids), "n_error": n_error, "n_unparsable": n_bad, "n_ok": int(ok.sum()),
           "fail_rate": 100 * n_bad / max(1, len(ids) - n_error),
           "prompt_tokens": sum(records[d]["prompt_tokens"] for d in ids if d in records),
           "output_tokens": sum(records[d]["output_tokens"] for d in ids if d in records),
           "ids_ok": [d for d, k in zip(ids, ok) if k], "y_ok": y_all[ok], "pred_ok": pred[ok]}
    if ok.sum() >= 2:
        out["metrics"] = regression_metrics(y_all[ok], pred[ok], ci=True)
        out["bins"] = classification_metrics(to_bins(y_all[ok], CALORIE_BIN_EDGES), to_bins(pred[ok], CALORIE_BIN_EDGES), ci=True)
        out["metrics_fallback"] = regression_metrics(y_all, np.where(ok, pred, train_mean))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="VLM sıfır-atış kalori referansı (Adım 14).")
    ap.add_argument("--provider", choices=["gemini", "openai"], default="gemini",
                    help="gemini: Google REST (GEMINI_API_KEY). openai: OpenAI uyumlu chat/completions (OPENAI_API_KEY, --base-url).")
    ap.add_argument("--model", default="gemini-2.5-flash",
                    help="Model kimliği. Güncel kimlikleri ve fiyatı sağlayıcının sayfasından doğrulayın; eski kimlikler 404 verir.")
    ap.add_argument("--base-url", default=None, help="openai sağlayıcısı için taban URL (varsayılan api.openai.com/v1).")
    ap.add_argument("--n", type=int, default=0, help="Rastgele alt küme boyutu (0 = tüm test, README: 300 önerilir).")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--img-size", type=int, default=512, help="Gönderilen görüntünün kısa kenarı (px, 0 = küçültme).")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--img-format", choices=["jpeg", "png"], default="jpeg")
    ap.add_argument("--rpm", type=float, default=0, help="Dakikadaki istek üst sınırı (0 = bekleme yok). Gemini ücretsiz katman düşüktür.")
    ap.add_argument("--max-tries", type=int, default=5, help="429/5xx için deneme sayısı (üstel bekleme).")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="Gemini düşünme bütçesi (0 = kapalı, ucuz). Verilmezse model varsayılanı.")
    ap.add_argument("--price-in", type=float, default=0, help="USD / 1M giriş tokenı (0 = bilinmiyor).")
    ap.add_argument("--price-out", type=float, default=0, help="USD / 1M çıkış tokenı (0 = bilinmiyor).")
    ap.add_argument("--run-name", default=None, help="Varsayılan vlm_<model>.")
    ap.add_argument("--root", default=N5K_DIR, help="Nutrition5k kökü.")
    ap.add_argument("--out-dir", default=None, help="results klasörü (varsayılan results/; --dry-run'da geçici klasör).")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="Ağ yok: sahte yanıt; results.csv'ye yazmaz, satırı ekrana basar.")
    args = ap.parse_args(argv)

    root = Path(args.root)
    meta = read_dish_metadata(root / "metadata")
    test_ids = load_n5k_split("test", root=root, metadata=meta)
    train_ids, _ = make_val_split(load_n5k_split("train", root=root, metadata=meta, verbose=False))
    train_mean = float(np.mean([meta[d]["calories"] for d in train_ids]))
    ids = sorted(test_ids)
    if args.n and args.n < len(ids):
        ids = sorted(random.Random(args.seed).sample(ids, args.n))

    run_name = args.run_name or f"vlm_{sanitize(args.model)}"
    out_dir = Path(args.out_dir) if args.out_dir else (Path(tempfile.mkdtemp(prefix="vlm_dryrun_")) if args.dry_run else RESULTS_DIR)
    (out_dir / "vlm").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "vlm" / f"{run_name}.jsonl"

    api_key = None
    if not args.dry_run:
        env = "GEMINI_API_KEY" if args.provider == "gemini" else "OPENAI_API_KEY"
        api_key = os.environ.get(env)
        if not api_key:
            sys.exit(f"{env} ortam değişkeni yok. Anahtarı dosyaya DEĞİL, ortama koyun: export {env}=...")

    done = load_done(jsonl_path)
    todo = [d for d in ids if d not in done]
    print(f"{run_name}: {len(ids)} tabak, {len(done)} önceden bitmiş, {len(todo)} sorulacak → {jsonl_path}"
          + ("  [DRY-RUN, ağ yok]" if args.dry_run else ""), flush=True)

    t_start, n_bad = time.time(), 0
    for i, d in enumerate(todo, 1):
        b64, mime = encode_image(root / "realsense_overhead" / d / "rgb.png", args.img_size, args.jpeg_quality, args.img_format)
        if args.dry_run:
            rec = fake_call(d)
        else:
            rec = call_vlm(args.provider, args.model, api_key, b64, mime, args.base_url, args.thinking_budget,
                           args.max_tries, args.timeout)
        parsed = parse_response(rec["text"]) if rec["error"] is None else None
        rec.update({"dish_id": d, "y_true": meta[d]["calories"], "model": args.model, "provider": args.provider,
                    "parsed_total": parsed["total_kcal"] if parsed else None, "items": parsed["items"] if parsed else None,
                    "time": datetime.now().isoformat(timespec="seconds")})
        append_jsonl(jsonl_path, rec)
        if rec["error"] is None:
            done[d] = rec
            n_bad += parsed is None
        else:
            print(f"  ✗ {d}: {rec['error']}", flush=True)
        if i % args.log_every == 0 or i == len(todo):
            p_tok = sum(r["prompt_tokens"] for r in done.values())
            o_tok = sum(r["output_tokens"] for r in done.values())
            print(f"  {i}/{len(todo)}  ayrıştırılamayan {n_bad}  token giriş/çıkış {p_tok}/{o_tok}  "
                  f"{time.time() - t_start:.0f} s", flush=True)
        if args.rpm and i < len(todo) and not args.dry_run:
            sleep(60 / args.rpm)

    s = summarize(done, ids, meta, train_mean)
    cost = cost_usd(s["prompt_tokens"], s["output_tokens"], args.price_in, args.price_out)
    cost_txt = f"${cost:.4f}" if cost is not None else "bilinmiyor (--price-in/--price-out verilmedi)"
    print(f"\n{run_name}: {s['n']} tabak, yanıt alınan {s['n'] - s['n_error']}, ayrıştırılamayan {s['n_unparsable']} "
          f"(%{s['fail_rate']:.1f}), HTTP hatası {s['n_error']}"
          + (" → aynı komutu tekrar çalıştırın" if s["n_error"] else ""))
    print(f"  token giriş {s['prompt_tokens']}  çıkış {s['output_tokens']}  tahmini maliyet {cost_txt}")
    if "metrics" not in s:
        sys.exit("Değerlendirilecek yeterli ayrıştırılmış yanıt yok.")
    m, b, fb = s["metrics"], s["bins"], s["metrics_fallback"]
    print(f"  test MAE {m['MAE']:.1f} [{m['MAE_ci_low']:.1f}–{m['MAE_ci_high']:.1f}]  PMAE %{m['PMAE']:.1f}  "
          f"RMSE {m['RMSE']:.1f}  R² {m['R2']:.2f}  ±100 kcal %{m['within_100kcal']:.1f}  | Tablo 3 F1 {b['f1']:.3f}")
    print(f"  ayrıştırılamayan → train ortalaması ({train_mean:.1f} kcal) ile: MAE {fb['MAE']:.1f}  PMAE %{fb['PMAE']:.1f}")

    config = f"{args.provider}:{args.model}_T0_{args.img_size}px" + (f"_n{args.n}_seed{args.seed}" if args.n else "")
    notes = (f"Adım 14 VLM sıfır-atış; {s['n']} tabak, {s['n_ok']} ayrıştırıldı, ayrıştırılamayan {s['n_unparsable']} "
             f"(%{s['fail_rate']:.1f}); metrikler ayrıştırılamayanlar atılarak, train ort. ({train_mean:.0f} kcal) ile "
             f"MAE {fb['MAE']:.1f} PMAE %{fb['PMAE']:.1f}; token giriş {s['prompt_tokens']} çıkış {s['output_tokens']}; "
             f"maliyet {cost_txt}" + (f" ({args.price_in:g}/{args.price_out:g} $/M)" if cost is not None else ""))
    preds_path = save_preds(run_name, s["ids_ok"], s["y_ok"], s["pred_ok"], path=out_dir / "preds" / f"{run_name}.csv")
    print(f"  tahminler: {preds_path}")
    if args.dry_run:
        print("  [DRY-RUN] results.csv'ye yazılmadı; satır:")
        print(f"  calorie,{run_name},{config},test,{ {k: round(v, 2) for k, v in m.items()} },{notes}")
        print(f"  calorie_bins,{run_name},{config},test,{ {k: round(v, 3) for k, v in b.items()} }")
    else:
        results_csv = out_dir / "results.csv"  # --out-dir verilmişse ortak results.csv'ye değil, oraya yazılır
        log_result("calorie", run_name, config, "test", m, notes=notes, path=results_csv)
        log_result("calorie_bins", run_name, config, "test", b, notes=notes, path=results_csv)
        print("  results/results.csv'ye yazıldı.")


if __name__ == "__main__":
    main()
