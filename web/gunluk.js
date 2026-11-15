/* Günlük takip — profil, günlük hedef, yenenlerin kaydı, besin halkaları, öneriler ve alev serisi.
   Her şey bu cihazın tarayıcısında (localStorage) kalır; hiçbir sunucuya gönderilmez.
   Hedef: Mifflin-St Jeor bazal metabolizma × aktivite katsayısı ± hedef farkı.
   Makrolar: protein kg başına (hedefe göre 1,2–1,6 g), yağ enerjinin %30'u, kalan karbonhidrat.
   Gün "tamam" sayılır: kalori hedefin %90–115'i arasında VE protein hedefin en az %90'ı. */
(() => {
  "use strict";
  const ANAHTAR = "kalori-gunluk-v1";
  const AKTIVITE = { "1.2": "Hareketsiz (masa başı)", "1.375": "Hafif (haftada 1–3 gün spor)", "1.55": "Orta (haftada 3–5 gün)", "1.725": "Yüksek (her gün spor)" };
  const HEDEF = { ver: ["Kilo vermek", -500, 1.6], koru: ["Kilomu korumak", 0, 1.2], al: ["Kilo almak", 300, 1.6] };
  const TAMAM_ALT = 0.9, TAMAM_UST = 1.15, PROTEIN_ESIK = 0.9;
  const HALKALAR = [ // anahtar, ad, birim, renk değişkeni
    ["kcal", "Kalori", "kcal", "--h-kcal"],
    ["p", "Protein", "g", "--h-protein"],
    ["k", "Karbonhidrat", "g", "--h-karb"],
    ["y", "Yağ", "g", "--h-yag"],
  ];

  const fmt = (x, b = 0) => Number(x).toLocaleString("tr-TR", { maximumFractionDigits: b, minimumFractionDigits: b });
  const gunAnahtari = (t = new Date()) =>
    `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")}`;
  const gunOnce = (anahtar, n) => { const [y, a, g] = anahtar.split("-").map(Number); return gunAnahtari(new Date(y, a - 1, g - n)); };

  // ---------- kayıt ----------
  let bellek = { profil: null, gunler: {} };   // localStorage yoksa (özel mod) yalnızca bu oturumda
  function oku() {
    try { const s = JSON.parse(localStorage.getItem(ANAHTAR)); if (s && typeof s === "object") bellek = { profil: s.profil || null, gunler: s.gunler || {} }; } catch { /* yok/bozuk */ }
    return bellek;
  }
  function yaz() { try { localStorage.setItem(ANAHTAR, JSON.stringify(bellek)); } catch { /* yer yok / özel mod */ } }

  // ---------- hesaplar ----------
  function hedefHesapla(pr) {
    const bmr = 10 * pr.kilo + 6.25 * pr.boy - 5 * pr.yas + (pr.cinsiyet === "e" ? 5 : -161);
    const [, fark, pKatsayi] = HEDEF[pr.hedef] || HEDEF.koru;
    const taban = pr.cinsiyet === "e" ? 1500 : 1200;          // güvenli alt sınır
    const kcal = Math.max(taban, Math.round((bmr * Number(pr.aktivite) + fark) / 10) * 10);
    const p = Math.round(pr.kilo * pKatsayi);
    const y = Math.round(kcal * 0.30 / 9);
    const k = Math.max(0, Math.round((kcal - p * 4 - y * 9) / 4));
    return { kcal, p, k, y, bmr: Math.round(bmr) };
  }

  function toplam(kayitlar) {
    return (kayitlar || []).reduce((s, e) => ({ kcal: s.kcal + e.kcal, p: s.p + e.p, k: s.k + e.k, y: s.y + e.y }), { kcal: 0, p: 0, k: 0, y: 0 });
  }

  function gunTamamMi(t, h) {
    return t.kcal >= TAMAM_ALT * h.kcal && t.kcal <= TAMAM_UST * h.kcal && t.p >= PROTEIN_ESIK * h.p;
  }

  function seri(gunler, h, bugun = gunAnahtari()) {
    // Bugün tamamsa bugünden, değilse dünden geriye ardışık tamam günleri sayılır (bugün henüz bitmedi).
    const tamam = (g) => gunTamamMi(toplam(gunler[g]), h);
    const bugunTamam = tamam(bugun);
    let n = 0, g = bugunTamam ? bugun : gunOnce(bugun, 1);
    while (gunler[g] && tamam(g)) { n++; g = gunOnce(g, 1); }
    return { n, bugunTamam };
  }

  function oneriler(t, h, saat = new Date().getHours()) {
    const kalan = h.kcal - t.kcal, pKalan = h.p - t.p;
    const out = [];
    if (t.kcal === 0) {
      out.push(saat < 11 ? "Güne başlamak için kahvaltını fotoğraflayıp ekle." : "Bugün henüz bir şey eklemedin; ilk öğününü fotoğrafla.");
      return out;
    }
    if (kalan < -0.15 * h.kcal) out.push(`Hedefi ${fmt(-kalan)} kcal aştın. Kalan öğünleri hafif tut: sebze, çorba, yoğurt.`);
    else if (kalan < 0) out.push(`Hedefi ${fmt(-kalan)} kcal geçtin, hâlâ kabul edilebilir aralıktasın. Bugünlük yeter.`);
    else if (kalan <= 0.1 * h.kcal) out.push(`Hedefe çok yakınsın: ${fmt(kalan)} kcal kaldı. Küçük bir ara öğün yeter.`);
    else if (saat >= 18 && kalan <= 700) out.push(`Akşam için hafif bir şey: yaklaşık ${fmt(kalan)} kcal'lik bir öğün seni hedefe taşır.`);
    else if (saat >= 18) out.push(`Daha ${fmt(kalan)} kcal eksik; akşam yemeğini doyurucu ve dengeli yap.`);
    else out.push(`Günlük hedefe ${fmt(kalan)} kcal kaldı.`);
    if (pKalan >= 15 && kalan > 0) {
      const oneri = pKalan >= 40 ? "tavuk göğsü, ton balığı, köfte" : pKalan >= 25 ? "yumurta, lor, mercimek, tavuk" : "yoğurt, ayran, peynir";
      out.push(`Proteinden ${fmt(pKalan)} g eksiğin var: ${oneri} iyi seçenekler.`);
    }
    if (t.y > 1.2 * h.y) out.push("Yağ hedefini aştın; kızartma ve soslu yemeklere bugün ara ver.");
    return out;
  }

  // ---------- çizim ----------
  const $ = (id) => document.getElementById(id);
  let elBugun, elProfil;

  function halkaSvg(t, h) {
    const R0 = 90, ADIM = 15, KAL = 11;
    const halkalar = HALKALAR.map(([a, , , renk], i) => {
      const r = R0 - i * ADIM, c = 2 * Math.PI * r, oran = h[a] ? t[a] / h[a] : 0;
      const dolu = Math.min(oran, 1), asim = a === "kcal" && oran > TAMAM_UST;
      return `<circle cx="100" cy="100" r="${r}" fill="none" stroke="var(--cubuk-zemin)" stroke-width="${KAL}"/>` +
        `<circle class="dolgu" cx="100" cy="100" r="${r}" fill="none" stroke="var(${asim ? "--h-asim" : renk})" stroke-width="${KAL}" stroke-linecap="round"` +
        ` stroke-dasharray="${(dolu * c).toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 100 100)"/>`;
    }).join("");
    const kalan = h.kcal - t.kcal;
    return `<svg viewBox="0 0 200 200" class="halka" role="img" aria-label="Kalori ${fmt(t.kcal)}/${fmt(h.kcal)}, protein ${fmt(t.p)}/${fmt(h.p)} g">${halkalar}` +
      `<text x="100" y="98" text-anchor="middle" class="h-buyuk">${fmt(Math.abs(kalan))}</text>` +
      `<text x="100" y="116" text-anchor="middle" class="h-kucuk">${kalan >= 0 ? "kcal kaldı" : "kcal fazla"}</text></svg>`;
  }

  function haftaNoktalari(gunler, h) {
    const bugun = gunAnahtari(), adlar = ["Pz", "Pt", "Sa", "Ça", "Pe", "Cu", "Ct"];
    let s = "";
    for (let i = 6; i >= 0; i--) {
      const g = gunOnce(bugun, i), [y, a, d] = g.split("-").map(Number);
      const t = toplam(gunler[g]), durum = !gunler[g]?.length ? "yok" : gunTamamMi(t, h) ? "tamam" : "eksik";
      s += `<li class="${durum}${i === 0 ? " bugun" : ""}" title="${d}.${a}: ${fmt(t.kcal)} kcal"><span></span>${adlar[new Date(y, a - 1, d).getDay()]}</li>`;
    }
    return `<ol class="hafta" aria-label="Son 7 gün">${s}</ol>`;
  }

  function ciz(kutla = false) {
    if (!elBugun) return;
    const pr = bellek.profil;
    if (!pr) {
      elBugun.innerHTML = `<div class="bugun-bos"><strong>Günlük hedefini belirle</strong>` +
        `<p>Yaş, boy, kilo ve hedefini gir; günlük kalori ve protein ihtiyacını hesaplayıp yediklerini takip edelim.</p>` +
        `<button type="button" class="buton birincil" id="profil-ac">Hedefimi hesapla</button></div>`;
      $("profil-ac").addEventListener("click", () => { elProfil.open = true; elProfil.scrollIntoView({ behavior: "smooth", block: "start" }); });
      return;
    }
    const h = hedefHesapla(pr), bugun = gunAnahtari(), kayit = bellek.gunler[bugun] || [], t = toplam(kayit);
    const { n, bugunTamam } = seri(bellek.gunler, h);
    const lejant = HALKALAR.map(([a, ad, birim, renk]) =>
      `<li><i style="background:var(${renk})"></i><span>${ad}</span><b>${fmt(t[a])} / ${fmt(h[a])} ${birim}</b><em>%${fmt(h[a] ? 100 * t[a] / h[a] : 0)}</em></li>`).join("");
    const liste = kayit.length ? kayit.map((e, i) =>
      `<li><span>${e.saat}${e.porsiyon !== 1 ? ` · ×${fmt(e.porsiyon, 1)}` : ""}</span><b>${fmt(e.kcal)} kcal</b><small>P ${fmt(e.p)} · K ${fmt(e.k)} · Y ${fmt(e.y)} g</small>` +
      `<button type="button" class="sil" data-i="${i}" aria-label="${e.saat} kaydını sil">×</button></li>`).join("") : "";
    elBugun.innerHTML =
      `<div class="bugun-ust"><h2>Bugün</h2>` +
      `<div class="seri${bugunTamam ? " yandi" : ""}${kutla ? " kutla" : ""}" title="${bugunTamam ? "Bugünün hedefi tamam" : "Bugün hedefi tuttur, seriyi sürdür"}">` +
      `<span class="alev" aria-hidden="true">🔥</span><b>${n}</b><small>gün seri</small></div></div>` +
      `<div class="bugun-govde">${halkaSvg(t, h)}<ul class="lejant">${lejant}</ul></div>` +
      `<ul class="oneriler">${oneriler(t, h).map((o) => `<li>${o}</li>`).join("")}</ul>` +
      (kutla ? `<p class="tebrik">Günlük hedefini tutturdun! Serin ${n} gün oldu. 🔥</p>` : "") +
      haftaNoktalari(bellek.gunler, h) +
      (kayit.length ? `<details class="kayitlar"><summary>Bugün eklenenler (${kayit.length})</summary><ul>${liste}</ul></details>` : "") +
      `<p class="kucuk">Gün tamam sayılır: kalori hedefin %${fmt(TAMAM_ALT * 100)}–${fmt(TAMAM_UST * 100)}'i ve protein en az %${fmt(PROTEIN_ESIK * 100)}'i. Tahminler ±40 kcal hatalıdır; tıbbi tavsiye değildir.</p>`;
    elBugun.querySelectorAll(".sil").forEach((b) => b.addEventListener("click", () => {
      kayit.splice(Number(b.dataset.i), 1);
      if (!kayit.length) delete bellek.gunler[bugun];
      yaz(); ciz();
    }));
  }

  function profilFormu() {
    const pr = bellek.profil || { cinsiyet: "k", yas: "", boy: "", kilo: "", aktivite: "1.375", hedef: "koru" };
    const sec = (ad, secenekler, deger) => `<select name="${ad}">${Object.entries(secenekler).map(([v, e]) =>
      `<option value="${v}"${String(v) === String(deger) ? " selected" : ""}>${Array.isArray(e) ? e[0] : e}</option>`).join("")}</select>`;
    const h = bellek.profil ? hedefHesapla(bellek.profil) : null;
    elProfil.innerHTML =
      `<summary>Hedefim${h ? ` · ${fmt(h.kcal)} kcal, ${fmt(h.p)} g protein` : ""}</summary>` +
      `<form id="profil-form" class="profil">` +
      `<label>Cinsiyet ${sec("cinsiyet", { k: "Kadın", e: "Erkek" }, pr.cinsiyet)}</label>` +
      `<label>Yaş <input name="yas" type="number" inputmode="numeric" min="14" max="100" required value="${pr.yas}"></label>` +
      `<label>Boy (cm) <input name="boy" type="number" inputmode="numeric" min="120" max="230" required value="${pr.boy}"></label>` +
      `<label>Kilo (kg) <input name="kilo" type="number" inputmode="decimal" min="30" max="250" step="0.1" required value="${pr.kilo}"></label>` +
      `<label class="genis">Hareket düzeyi ${sec("aktivite", AKTIVITE, pr.aktivite)}</label>` +
      `<label class="genis">Hedef ${sec("hedef", HEDEF, pr.hedef)}</label>` +
      `<button type="submit" class="buton birincil genis">Kaydet</button>` +
      (h ? `<p class="kucuk genis">Bazal metabolizma ${fmt(h.bmr)} kcal → günlük ${fmt(h.kcal)} kcal · protein ${fmt(h.p)} g · karbonhidrat ${fmt(h.k)} g · yağ ${fmt(h.y)} g.</p>` : "") +
      `<p class="kucuk genis">Bilgiler yalnızca bu cihazda saklanır.${bellek.profil ? ` <button type="button" class="baglanti" id="sifirla">Tüm kayıtları sil</button>` : ""}</p>` +
      `</form>`;
    $("profil-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const f = new FormData(ev.target);
      bellek.profil = { cinsiyet: f.get("cinsiyet"), yas: Number(f.get("yas")), boy: Number(f.get("boy")), kilo: Number(f.get("kilo")), aktivite: f.get("aktivite"), hedef: f.get("hedef") };
      yaz(); profilFormu(); ciz();
      elBugun.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    const sifirla = $("sifirla");
    if (sifirla) sifirla.addEventListener("click", () => {
      if (!confirm("Profil ve tüm günlük kayıtlar bu cihazdan silinsin mi?")) return;
      bellek = { profil: null, gunler: {} }; yaz(); profilFormu(); ciz();
    });
  }

  // Sonuç kartına eklenen "Günlüğe ekle" kutusu. p = [kcal, kütle g, yağ g, karbonhidrat g, protein g]
  function ekleKutusu(p) {
    const kutu = document.createElement("div");
    kutu.className = "ekle";
    const oranlar = [0.5, 1, 1.5, 2];
    let porsiyon = 1;
    kutu.innerHTML = `<div class="porsiyon" role="group" aria-label="Ne kadarını yedin?"><span>Yediğin:</span>` +
      oranlar.map((o) => `<button type="button" data-o="${o}" class="${o === 1 ? "secili" : ""}" aria-pressed="${o === 1}">${o === 1 ? "Tamamı" : o === 0.5 ? "Yarısı" : "×" + fmt(o, 1)}</button>`).join("") +
      `</div><button type="button" class="buton birincil ekle-dugme">Günlüğe ekle · ${fmt(p[0])} kcal</button>`;
    const dugme = kutu.querySelector(".ekle-dugme");
    kutu.querySelectorAll(".porsiyon button").forEach((b) => b.addEventListener("click", () => {
      porsiyon = Number(b.dataset.o);
      kutu.querySelectorAll(".porsiyon button").forEach((x) => { x.classList.toggle("secili", x === b); x.setAttribute("aria-pressed", x === b); });
      dugme.textContent = `Günlüğe ekle · ${fmt(p[0] * porsiyon)} kcal`;
    }));
    dugme.addEventListener("click", () => {
      const once = bellek.profil ? seri(bellek.gunler, hedefHesapla(bellek.profil)).bugunTamam : false;
      const g = gunAnahtari(), simdi = new Date();
      (bellek.gunler[g] ||= []).push({
        saat: `${String(simdi.getHours()).padStart(2, "0")}:${String(simdi.getMinutes()).padStart(2, "0")}`, porsiyon,
        kcal: Math.round(p[0] * porsiyon), g: Math.round(p[1] * porsiyon), y: +(p[2] * porsiyon).toFixed(1), k: +(p[3] * porsiyon).toFixed(1), p: +(p[4] * porsiyon).toFixed(1),
      });
      yaz();
      const sonra = bellek.profil ? seri(bellek.gunler, hedefHesapla(bellek.profil)).bugunTamam : false;
      ciz(!once && sonra);
      dugme.disabled = true; dugme.textContent = "Eklendi ✓";
      kutu.querySelectorAll(".porsiyon button").forEach((x) => { x.disabled = true; });
      elBugun.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    return kutu;
  }

  function kur() {
    elBugun = $("bugun"); elProfil = $("profil");
    if (!elBugun || !elProfil) return;
    oku(); profilFormu(); ciz();
    // Gece yarısı geçince yeni güne geç
    let son = gunAnahtari();
    setInterval(() => { const g = gunAnahtari(); if (g !== son) { son = g; ciz(); } }, 60000);
  }

  const hedefKcal = () => (bellek.profil ? hedefHesapla(bellek.profil).kcal : null);
  window.Gunluk = { kur, ekleKutusu, hedefKcal, _test: { hedefHesapla, toplam, gunTamamMi, seri, oneriler, gunOnce, gunAnahtari } };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", kur); else kur();
})();
