/* Kalori tahmini — tarayıcı içi çıkarım. Model worker.js içinde (onnxruntime-web) çalışır; bu dosya
   fotoğrafı hazırlar (4:3 kırpma → 224×224, 0–1, [1,3,224,224]) ve sonucu çizer.
   Model: web/model.onnx (scripts/export_onnx.py ile üretilir). Çıktı: [kcal, kütle g, yağ g, karbonhidrat g, protein g]. */
(() => {
  "use strict";
  const BOYUT = 224;
  const ORAN = 4 / 3;                 // Nutrition5k üst kamera oranı (src/transforms.py: crop_to_aspect)
  const TIPIK_HATA = 40;              // test MAE (kcal), convnext_tiny_reg_mt
  const GUNLUK = 2000;                // etiketlerdeki referans günlük enerji
  const ARALIK = [116.1, 313.7];      // Tablo 3 sınırları (src/config.py: CALORIE_BIN_EDGES)
  const MAKRO = [["Yağ", 2, 9], ["Karbonhidrat", 3, 4], ["Protein", 4, 4]]; // ad, çıktı indeksi, kcal/g
  const MODEL_URL = "model.onnx";
  const MODEL_MB = 28.2;              // içerik uzunluğu gelmezse ilerleme için (web/model.json ile aynı)
  const ONBELLEK = "kalori-model-v1"; // Cache API: model bir kez iner, sonraki açılışlarda yerelden

  const $ = (id) => document.getElementById(id);
  const durum = $("durum"), sonuc = $("sonuc"), bos = $("bos"), onizleme = $("onizleme"), kirpma = $("kirpma");
  const fmt = (x, b = 0) => Number(x).toLocaleString("tr-TR", { maximumFractionDigits: b, minimumFractionDigits: b });

  let worker = null, hazir = null, mesgul = false, bekleyenCevap = null;

  function setDurum(metin, hata = false) {
    durum.textContent = metin;
    durum.classList.toggle("hata", hata);
  }

  async function modelGetir() {
    // 1) Cache API'de varsa oradan (anında). 2) Yoksa indir, ilerlemeyi göster, önbelleğe koy.
    let onbellek = null;
    try { onbellek = await caches.open(ONBELLEK); } catch { /* özel mod vb. */ }
    if (onbellek) {
      const eski = await onbellek.match(MODEL_URL);
      if (eski) { setDurum("Model yerel önbellekten alınıyor…"); return eski.arrayBuffer(); }
    }
    const yanit = await fetch(MODEL_URL);
    if (!yanit.ok) throw new Error("model.onnx bulunamadı (" + yanit.status + ")");
    const toplam = Number(yanit.headers.get("content-length")) || Math.round(MODEL_MB * 1e6);
    const okuyucu = yanit.body.getReader();
    const parcalar = []; let alinan = 0;
    for (;;) {
      const { done, value } = await okuyucu.read();
      if (done) break;
      parcalar.push(value); alinan += value.length;
      setDurum(`Model ilk kez indiriliyor, ${fmt(MODEL_MB)} MB (bir kez; sonra telefonda kalır)… %${Math.min(99, Math.round(100 * alinan / toplam))}`);
    }
    const veri = new Uint8Array(alinan); let k = 0;
    for (const p of parcalar) { veri.set(p, k); k += p.length; }
    parcalar.length = 0;
    if (onbellek) {
      try { await onbellek.put(MODEL_URL, new Response(veri.slice(), { headers: { "Content-Type": "application/octet-stream" } })); } catch { /* yer yok */ }
    }
    return veri.buffer;
  }

  function modelYukle() {
    if (hazir) return hazir;
    hazir = new Promise(async (coz, reddet) => {
      try {
        if (!window.Worker) throw new Error("tarayıcı Web Worker desteklemiyor");
        worker = new Worker("worker.js");
        worker.onerror = (e) => { const m = "çıkarım iş parçacığı: " + (e.message || "yüklenemedi"); setDurum(m, true); reddet(new Error(m)); };
        worker.onmessage = (e) => {
          const m = e.data;
          if (m.tip === "hazir") { setDurum("Hazır. Fotoğraf çekin veya bir örneğe dokunun."); coz(); }
          else if (m.tip === "sonuc" && bekleyenCevap) { bekleyenCevap.coz(m); bekleyenCevap = null; }
          else if (m.tip === "hata") {
            if (bekleyenCevap) { bekleyenCevap.reddet(new Error(m.mesaj)); bekleyenCevap = null; }
            else { setDurum("Model yüklenemedi: " + m.mesaj, true); reddet(new Error(m.mesaj)); }
          }
        };
        const veri = await modelGetir();
        setDurum("Model hazırlanıyor (10–20 sn sürebilir)…");
        worker.postMessage({ tip: "yukle", veri }, [veri]);
      } catch (e) {
        setDurum("Model yüklenemedi: " + e.message, true);
        reddet(e);
      }
    });
    hazir.catch(() => { hazir = null; });
    return hazir;
  }

  function calistir(veri, sekil) {
    return new Promise((coz, reddet) => {
      bekleyenCevap = { coz, reddet };
      worker.postMessage({ tip: "tahmin", veri, sekil }, [veri.buffer]);
    });
  }

  async function gorselAc(kaynak) {
    // Dosya ya da URL → ImageBitmap (EXIF yönü uygulanmış); büyük telefon fotoğrafı küçültülerek açılır
    const blob = kaynak instanceof Blob ? kaynak : await (await fetch(kaynak)).blob();
    try {
      return await createImageBitmap(blob, { imageOrientation: "from-image", resizeWidth: 1024, resizeQuality: "high" });
    } catch {
      try {
        return await createImageBitmap(blob, { imageOrientation: "from-image" });
      } catch {
        return await new Promise((res, rej) => {
          const im = new Image();
          im.onload = () => res(im); im.onerror = () => rej(new Error("görsel açılamadı"));
          im.src = URL.createObjectURL(blob);
        });
      }
    }
  }

  function kirp(gorsel) {
    // crop_to_aspect ile aynı: fazla genişse yanlardan, fazla uzunsa üstten/alttan ortadan kırp
    const w = gorsel.width, h = gorsel.height;
    let sx = 0, sy = 0, sw = w, sh = h;
    if (w / h > ORAN) { sw = Math.round(h * ORAN); sx = Math.floor((w - sw) / 2); }
    else { sh = Math.round(w / ORAN); sy = Math.floor((h - sh) / 2); }
    return { sx, sy, sw, sh };
  }

  function tensorYap(gorsel) {
    const { sx, sy, sw, sh } = kirp(gorsel);
    kirpma.getContext("2d").drawImage(gorsel, sx, sy, sw, sh, 0, 0, kirpma.width, kirpma.height);   // önizleme (4:3)
    const c = document.createElement("canvas"); c.width = BOYUT; c.height = BOYUT;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(gorsel, sx, sy, sw, sh, 0, 0, BOYUT, BOYUT);   // eğitimdeki Resize((224,224)) gibi, oran korunmaz
    const px = ctx.getImageData(0, 0, BOYUT, BOYUT).data;
    const n = BOYUT * BOYUT, veri = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      veri[i] = px[4 * i] / 255; veri[n + i] = px[4 * i + 1] / 255; veri[2 * n + i] = px[4 * i + 2] / 255;
    }
    return veri;
  }

  function etiket(kcal) { return kcal < ARALIK[0] ? "düşük kalorili" : kcal < ARALIK[1] ? "orta kalorili" : "yüksek kalorili"; }

  function sonucYaz(p, ms) {
    const kcal = p[0];
    const enerji = MAKRO.map(([, i, f]) => p[i] * f), toplam = enerji.reduce((a, b) => a + b, 0) || 1;
    const satirlar = [`<tr><td class="ad">Porsiyon</td><td class="deger">${fmt(p[1])} g</td><td class="cubuk"></td></tr>`]
      .concat(MAKRO.map(([ad, i], k) =>
        `<tr><td class="ad">${ad}</td><td class="deger">${fmt(p[i])} g</td><td class="cubuk"><div><span style="width:${Math.round(100 * enerji[k] / toplam)}%"></span></div></td></tr>`));
    sonuc.innerHTML =
      `<div class="kcal">${fmt(kcal)}<small>kcal</small></div>` +
      `<div class="aralik">yaklaşık ${fmt(Math.max(kcal - TIPIK_HATA, 0))}–${fmt(kcal + TIPIK_HATA)} kcal arası · günlük ${GUNLUK} kcal'e göre %${fmt(100 * kcal / GUNLUK)}</div>` +
      `<span class="etiket">${etiket(kcal)}</span>` +
      `<table>${satirlar.join("")}</table>` +
      `<div class="not">Çubuklar enerjinin makrolara dağılımı (yağ 9, karbonhidrat ve protein 4 kcal/g); makrolardan hesaplanan enerji ≈ ${fmt(toplam)} kcal.</div>` +
      `<div class="sure">Cihazda hesaplandı, ${fmt(ms)} ms.</div>`;
    sonuc.hidden = false; bos.hidden = true;
  }

  async function tahmin(kaynak) {
    if (mesgul) return;
    mesgul = true;
    try {
      const gorsel = await gorselAc(kaynak);
      const veri = tensorYap(gorsel);
      if (gorsel.close) gorsel.close();
      onizleme.hidden = false;
      if (!hazir) modelYukle();
      await hazir;
      setDurum("Hesaplanıyor…");
      const { p, ms } = await calistir(veri, [1, 3, BOYUT, BOYUT]);
      sonucYaz(p, ms);
      setDurum("Hazır. Başka bir fotoğraf deneyebilirsiniz.");
      sonuc.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (e) {
      setDurum("Tahmin yapılamadı: " + e.message, true);
    } finally {
      mesgul = false;
    }
  }

  // Düğmeler: gerçek <button> → kullanıcı dokunuşuyla gizli dosya girişini açar (iOS/Android'de kamera, masaüstünde dosya seçici)
  function dosyaSec(ev) {
    const f = ev.target.files && ev.target.files[0];
    if (f) tahmin(f);
    ev.target.value = "";
  }
  $("kamera").addEventListener("change", dosyaSec);
  $("galeri").addEventListener("change", dosyaSec);
  $("kamera-dugme").addEventListener("click", () => $("kamera").click());
  $("galeri-dugme").addEventListener("click", () => $("galeri").click());

  async function ornekleriKur() {
    let liste = [];
    try { liste = await (await fetch("ornekler/ornekler.json")).json(); } catch { return; }
    const kutu = $("ornek-liste"), govde = $("ornek-tablo").querySelector("tbody");
    liste.forEach((o, i) => {
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("aria-label", `Örnek ${i + 1}`);
      b.innerHTML = `<img src="${o.file}" alt="Örnek tabak ${i + 1}" loading="lazy">`;
      b.addEventListener("click", () => tahmin(o.file));
      kutu.appendChild(b);
      govde.insertAdjacentHTML("beforeend", `<tr><td>${i + 1}</td><td>${o.gercek} kcal</td><td>${o.model} kcal</td></tr>`);
    });
    const secili = Number(new URLSearchParams(location.search).get("ornek"));
    if (secili >= 1 && secili <= liste.length) tahmin(liste[secili - 1].file);
  }

  fetch("model.json").then((r) => r.json()).then((m) => {
    $("model-bilgi").textContent = `Model int8 ONNX (${m.size_mb} MB) olarak tarayıcıda çalışır (onnxruntime-web). ` +
      `Küçültmenin etkisi: aynı ${m.check_n} test tabakta MAE ${fmt(m.MAE_torch_subset, 1)} → ${fmt(m.MAE_onnx_subset, 1)} kcal.`;
  }).catch(() => {});

  ornekleriKur();
  modelYukle();
})();
