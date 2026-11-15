/* NeYedim — öğün ve alışkanlık takibi.
   Profil → günlük hedef (Mifflin-St Jeor × hareket ± hedef farkı; protein kg başına, yağ %30, kalan karbonhidrat).
   Kayıtlar öğün öğün (kahvaltı, öğle, akşam, ara öğün) bu cihazın tarayıcısında (localStorage) tutulur, hiçbir yere gönderilmez.
   Gün "tamam": kalori hedefin %90–115'i ve protein en az %90'ı. Art arda tamam günler alev serisini büyütür. */
(() => {
  "use strict";
  const ANAHTAR = "kalori-gunluk-v1";
  const AKTIVITE = { "1.2": ["Az", "masa başı"], "1.375": ["Hafif", "haftada 1–3 gün spor"], "1.55": ["Orta", "haftada 3–5 gün"], "1.725": ["Yüksek", "her gün spor"] };
  const HEDEF = { ver: ["Kilo vermek", -500, 1.6], koru: ["Korumak", 0, 1.2], al: ["Kilo almak", 300, 1.6] };
  const OGUNLER = [["kahvalti", "Kahvaltı", "☕", "Kahvaltıya"], ["ogle", "Öğle", "🥗", "Öğle yemeğine"], ["aksam", "Akşam", "🍲", "Akşam yemeğine"], ["ara", "Ara öğün", "🍎", "Ara öğüne"]];
  const OGUN_RENK = { kahvalti: "var(--h-karb)", ogle: "var(--vurgu)", aksam: "var(--h-yag)", ara: "var(--h-protein)" };
  const TAMAM_ALT = 0.9, TAMAM_UST = 1.15, PROTEIN_ESIK = 0.9;
  const MAKRO = [["kcal", "Kalori", "kcal", "--h-kcal"], ["p", "Protein", "g", "--h-protein"], ["k", "Karbonhidrat", "g", "--h-karb"], ["y", "Yağ", "g", "--h-yag"]];

  const $ = (id) => document.getElementById(id);
  const fmt = (x, b = 0) => Number(x).toLocaleString("tr-TR", { maximumFractionDigits: b, minimumFractionDigits: b });
  const gunAnahtari = (t = new Date()) => `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")}`;
  const tarihOf = (a) => { const [y, m, d] = a.split("-").map(Number); return new Date(y, m - 1, d); };
  const gunOnce = (a, n) => { const t = tarihOf(a); t.setDate(t.getDate() - n); return gunAnahtari(t); };
  const ogunSaatten = (h) => (h >= 5 && h < 11 ? "kahvalti" : h >= 11 && h < 16 ? "ogle" : h >= 16 && h < 22 ? "aksam" : "ara");
  const ogunOf = (e) => e.ogun || ogunSaatten(parseInt(e.saat, 10) || 12);
  const ogunAd = (k) => (OGUNLER.find((o) => o[0] === k) || OGUNLER[3]);

  // ---------- kayıt ----------
  let bellek = { profil: null, gunler: {} };
  function oku() {
    try { const s = JSON.parse(localStorage.getItem(ANAHTAR)); if (s && typeof s === "object") bellek = { profil: s.profil || null, gunler: s.gunler || {} }; } catch { /* yok ya da bozuk */ }
  }
  function yaz() { try { localStorage.setItem(ANAHTAR, JSON.stringify(bellek)); } catch { /* özel mod / yer yok */ } }

  // ---------- hesaplar ----------
  function hedefHesapla(pr) {
    const bmr = 10 * pr.kilo + 6.25 * pr.boy - 5 * pr.yas + (pr.cinsiyet === "e" ? 5 : -161);
    const [, fark, pKat] = HEDEF[pr.hedef] || HEDEF.koru;
    const kcal = Math.max(pr.cinsiyet === "e" ? 1500 : 1200, Math.round((bmr * Number(pr.aktivite) + fark) / 10) * 10);
    const p = Math.round(pr.kilo * pKat), y = Math.round(kcal * 0.3 / 9);
    return { kcal, p, y, k: Math.max(0, Math.round((kcal - p * 4 - y * 9) / 4)), bmr: Math.round(bmr) };
  }
  const hedef = () => (bellek.profil ? hedefHesapla(bellek.profil) : null);
  const toplam = (k) => (k || []).reduce((s, e) => ({ kcal: s.kcal + e.kcal, p: s.p + e.p, k: s.k + e.k, y: s.y + e.y }), { kcal: 0, p: 0, k: 0, y: 0 });
  const gunTamamMi = (t, h) => t.kcal >= TAMAM_ALT * h.kcal && t.kcal <= TAMAM_UST * h.kcal && t.p >= PROTEIN_ESIK * h.p;

  function seri(gunler, h, bugun = gunAnahtari()) {
    const tamam = (g) => gunler[g] && gunTamamMi(toplam(gunler[g]), h);
    const bugunTamam = !!tamam(bugun);
    let n = 0, g = bugunTamam ? bugun : gunOnce(bugun, 1);
    while (tamam(g)) { n++; g = gunOnce(g, 1); }
    return { n, bugunTamam };
  }
  function enUzunSeri(gunler, h) {
    const gunlerSirali = Object.keys(gunler).sort();
    let en = 0, su = 0, onceki = null;
    for (const g of gunlerSirali) {
      if (gunTamamMi(toplam(gunler[g]), h)) { su = onceki && gunOnce(g, 1) === onceki ? su + 1 : 1; onceki = g; en = Math.max(en, su); }
      else { su = 0; onceki = null; }
    }
    return en;
  }

  function oneriler(t, h, saat = new Date().getHours()) {
    const kalan = h.kcal - t.kcal, pKalan = h.p - t.p, out = [];
    if (t.kcal === 0) return [["🍽️", saat < 11 ? "Güne kahvaltınla başla: tabağını çek, öğününe ekle." : "Bugün henüz bir şey eklemedin. İlk öğününün fotoğrafını çek."]];
    if (kalan < -0.15 * h.kcal) out.push(["🥬", `Hedefi ${fmt(-kalan)} kcal aştın. Kalan öğünleri sebze, çorba ya da yoğurtla hafif tut.`]);
    else if (kalan < 0) out.push(["👌", `Hedefi ${fmt(-kalan)} kcal geçtin ama hâlâ makul aralıktasın. Bugünlük bu kadarı yeter.`]);
    else if (kalan <= 0.1 * h.kcal) out.push(["🎯", `Hedefe ${fmt(kalan)} kcal kaldı; küçük bir ara öğün yeter.`]);
    else if (saat >= 18 && kalan <= 700) out.push(["🌙", `Akşam için yaklaşık ${fmt(kalan)} kcal'lik hafif bir öğün seni hedefe taşır.`]);
    else if (saat >= 18) out.push(["🍲", `Daha ${fmt(kalan)} kcal eksik. Akşam yemeğini doyurucu ve dengeli yap.`]);
    else out.push(["📍", `Günlük hedefe ${fmt(kalan)} kcal kaldı.`]);
    if (pKalan >= 15 && kalan > 0) {
      const ne = pKalan >= 40 ? "tavuk, balık ya da köfte" : pKalan >= 25 ? "yumurta, lor, mercimek" : "yoğurt, ayran ya da peynir";
      out.push(["🥚", `Proteinden ${fmt(pKalan)} g eksiğin var; ${ne} iyi gider.`]);
    }
    if (t.y > 1.2 * h.y) out.push(["🍳", "Yağ hedefini aştın; bugün kızartma ve soslu yemeklere ara ver."]);
    // Alev için ne eksik? (gün tamam değilse)
    if (!gunTamamMi(t, h)) {
      const kEksik = Math.ceil(TAMAM_ALT * h.kcal - t.kcal), pEksik = Math.ceil(PROTEIN_ESIK * h.p - t.p), eksik = [];
      if (kEksik > 0) eksik.push(`${fmt(kEksik)} kcal`);
      if (pEksik > 0) eksik.push(`${fmt(pEksik)} g protein`);
      if (eksik.length) out.unshift(["🔥", `Bugünün alevi için ${eksik.join(" ve ")} daha ekle.`]);
      else if (t.kcal > TAMAM_UST * h.kcal) out.unshift(["🔥", `Kalori hedefin %${fmt(TAMAM_UST * 100)}'ini geçtiği için bugün alev yanmıyor.`]);
    }
    return out;
  }

  function aliskanlikVerisi(gunler, h, bugun = gunAnahtari()) {
    const gunler7 = Array.from({ length: 7 }, (_, i) => gunOnce(bugun, 6 - i));
    const kayitli = gunler7.filter((g) => gunler[g]?.length);
    const ogunKcal = { kahvalti: 0, ogle: 0, aksam: 0, ara: 0 };
    let geceGun = 0, kahvaltisiz = 0, toplamKcal = 0, toplamP = 0, tamamGun = 0;
    for (const g of kayitli) {
      const k = gunler[g], t = toplam(k);
      toplamKcal += t.kcal; toplamP += t.p;
      k.forEach((e) => { ogunKcal[ogunOf(e)] += e.kcal; });
      if (!k.some((e) => ogunOf(e) === "kahvalti")) kahvaltisiz++;
      if (k.some((e) => { const s = parseInt(e.saat, 10); return s >= 22 || s < 4; })) geceGun++;
      if (h && gunTamamMi(t, h)) tamamGun++;
    }
    const n = kayitli.length || 1;
    return { gunler7, kayitli, ogunKcal, geceGun, kahvaltisiz, ortKcal: toplamKcal / n, ortP: toplamP / n, tamamGun, toplamKcal };
  }

  function icgoruler(v, h) {
    const out = [], n = v.kayitli.length;
    if (!n) return out;
    const pay = (k) => (v.toplamKcal ? v.ogunKcal[k] / v.toplamKcal : 0);
    if (v.kahvaltisiz >= 2) out.push(["☕", `Kayıt tuttuğun ${n} günün ${v.kahvaltisiz}'${ek(v.kahvaltisiz)} kahvaltı yok. Sabah yiyenler gün içinde daha az atıştırma eğiliminde.`]);
    if (pay("aksam") > 0.45) out.push(["🌆", `Kalorinin %${fmt(100 * pay("aksam"))}'i akşam yemeğinden geliyor. Bir kısmını öğlene kaydırmayı dene.`]);
    if (pay("ara") > 0.3) out.push(["🍪", `Ara öğünler kalorinin %${fmt(100 * pay("ara"))}'ini oluşturuyor. Atıştırmalıkları meyve, yoğurt veya kuruyemişle değiştir.`]);
    if (v.geceGun >= 2) out.push(["🌙", `${v.geceGun} gün saat 22'den sonra yemek yedin. Geç saatte yemek uykuyu ve ertesi sabahki iştahı etkiler.`]);
    if (h) {
      if (v.ortP < 0.8 * h.p) out.push(["🥚", `Günlük protein ortalaman ${fmt(v.ortP)} g, hedefin %${fmt(100 * v.ortP / h.p)}'i. Her öğüne bir protein kaynağı ekle.`]);
      if (v.ortKcal > 1.15 * h.kcal) out.push(["📈", `Kayıtlı günlerde ortalama ${fmt(v.ortKcal)} kcal yedin; hedefin ${fmt(v.ortKcal - h.kcal)} kcal üstünde.`]);
      else if (v.ortKcal < 0.8 * h.kcal) out.push(["📉", `Kayıtlı günlerde ortalama ${fmt(v.ortKcal)} kcal yedin; hedefin ${fmt(h.kcal - v.ortKcal)} kcal altında. Öğün atlıyor ya da eklemeyi unutuyor olabilirsin.`]);
    }
    if (!out.length) out.push(["🌿", "Öğün düzenin dengeli görünüyor. Böyle devam et."]);
    return out;
  }
  function ek(n) { // Türkçe bulunma eki: 1'inde, 2'sinde, 3'ünde…
    const s = { 1: "inde", 2: "sinde", 3: "ünde", 4: "ünde", 5: "inde", 6: "sında", 7: "sinde" };
    return s[n] || "inde";
  }

  // ---------- sekmeler ----------
  const SEKMELER = ["bugun", "ekle", "aliskanlik", "profil"];
  let bekleyenOgun = null;
  function git(ad, kaydir = true) {
    if (!SEKMELER.includes(ad)) ad = "bugun";
    document.querySelectorAll("[data-panel]").forEach((p) => { p.hidden = p.dataset.panel !== ad; });
    document.querySelectorAll(".menu a[data-sekme]").forEach((a) => {
      if (a.dataset.sekme === ad) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    if (location.hash.slice(1) !== ad) history.replaceState(null, "", "#" + ad);
    if (ad === "aliskanlik") cizAliskanlik();
    if (kaydir) window.scrollTo({ top: 0 });
  }

  // ---------- Bugün ----------
  function tabakSvg(t, h) {
    const halkalar = MAKRO.map(([a, , , renk], i) => {
      const r = 93 - i * 12, c = 2 * Math.PI * r, oran = h[a] ? t[a] / h[a] : 0;
      const asim = a === "kcal" && oran > TAMAM_UST;
      return `<circle cx="100" cy="100" r="${r}" fill="none" stroke="var(--iz)" stroke-width="8.5"/>` +
        `<circle class="dolgu" cx="100" cy="100" r="${r}" fill="none" stroke="var(${asim ? "--h-asim" : renk})" stroke-width="8.5" stroke-linecap="round"` +
        ` stroke-dasharray="${(Math.min(oran, 1) * c).toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 100 100)"/>`;
    }).join("");
    const kalan = h.kcal - t.kcal;
    return `<div class="tabak" role="img" aria-label="Kalori ${fmt(t.kcal)} / ${fmt(h.kcal)}, protein ${fmt(t.p)} / ${fmt(h.p)} gram">` +
      `<svg viewBox="0 0 200 200" aria-hidden="true">${halkalar}</svg>` +
      `<div class="orta${kalan < 0 ? " fazla" : ""}"><b>${fmt(Math.abs(kalan))}</b><span>${kalan >= 0 ? "kcal kaldı" : "kcal fazla"}</span></div></div>`;
  }

  function cizSeri(kutla = false) {
    const el = $("seri"), h = hedef();
    if (!h) { el.hidden = true; return; }
    const { n, bugunTamam } = seri(bellek.gunler, h);
    el.hidden = false;
    el.className = "seri" + (bugunTamam ? " yandi" : "") + (kutla ? " kutla" : "");
    el.title = bugunTamam ? "Bugünün hedefi tamam" : "Bugün hedefini tuttur, seriyi sürdür";
    el.innerHTML = `<span class="alev" aria-hidden="true">🔥</span><b>${n}</b><small>gün seri</small>`;
  }

  function cizBugun(kutla = false) {
    $("bugun-tarih").textContent = new Date().toLocaleDateString("tr-TR", { weekday: "long", day: "numeric", month: "long" });
    cizSeri(kutla);
    const kutu = $("bugun-icerik"), h = hedef();
    if (!h) {
      kutu.innerHTML = `<div class="kart bos-tabak"><div class="tabak" aria-hidden="true"><div class="orta"><b>?</b><span>kcal</span></div></div>` +
        `<h2 style="margin-top:18px">Önce günlük hedefini belirleyelim</h2><p>Yaşını, boyunu, kilonu ve hedefini gir; ne kadar yemen gerektiğini hesaplayıp öğünlerini bu tabakta toplayalım.</p>` +
        `<a class="buton birincil" href="#profil">Hedefimi hesapla</a></div>`;
      return;
    }
    const bugun = gunAnahtari(), kayit = bellek.gunler[bugun] || [], t = toplam(kayit);
    const makrolar = MAKRO.map(([a, ad, birim, renk]) => {
      const oran = h[a] ? t[a] / h[a] : 0;
      return `<li><i style="background:var(${renk})"></i><span>${ad}</span><b>${fmt(t[a])} <small>/ ${fmt(h[a])} ${birim}</small></b>` +
        `<div class="cubuk"><em style="width:${Math.min(100, 100 * oran).toFixed(0)}%;background:var(${a === "kcal" && oran > TAMAM_UST ? "--h-asim" : renk})"></em></div></li>`;
    }).join("");
    const tebrik = kutla ? `<li class="tebrik" data-ikon="🔥">Günlük hedefini tutturdun! Serin ${seri(bellek.gunler, h).n} gün oldu.</li>` : "";
    const oneri = oneriler(t, h).map(([i, m]) => `<li data-ikon="${i}">${m}</li>`).join("");
    const ogunler = OGUNLER.map(([k, ad, ikon]) => {
      const kalemler = kayit.map((e, i) => [e, i]).filter(([e]) => ogunOf(e) === k);
      const kc = kalemler.reduce((s, [e]) => s + e.kcal, 0);
      const liste = kalemler.map(([e, i]) =>
        `<li><div>${e.saat}${e.porsiyon !== 1 ? `, ${porsiyonAd(e.porsiyon).toLowerCase()}` : ""}<br><span>P ${fmt(e.p)} g, K ${fmt(e.k)} g, Y ${fmt(e.y)} g</span></div>` +
        `<b>${fmt(e.kcal)} kcal</b><button type="button" class="sil" data-i="${i}" aria-label="${ad}, ${e.saat} kaydını sil">×</button></li>`).join("");
      return `<article class="ogun${kalemler.length ? "" : " bos"}"><div class="ogun-bas"><span class="ogun-ikon" aria-hidden="true">${ikon}</span>` +
        `<div><h3>${ad}</h3><p>${kalemler.length ? `${kalemler.length} tabak` : "Henüz eklenmedi"}</p></div>` +
        (kalemler.length ? `<span class="ogun-kcal">${fmt(kc)} <small>kcal</small></span>` : `<button type="button" class="ogun-ekle" data-ogun="${k}">Ekle</button>`) +
        `</div>${liste ? `<ul class="kalemler">${liste}</ul>` : ""}</article>`;
    }).join("");
    kutu.innerHTML =
      `<div class="bugun-sol"><div class="kart tabak-kart">${tabakSvg(t, h)}<ul class="makrolar">${makrolar}</ul></div>` +
      `<ul class="oneriler">${tebrik}${oneri}</ul></div>` +
      `<div><h2 style="margin:0 0 10px">Öğünlerin</h2><div class="ogunler">${ogunler}</div>` +
      `<p class="kucuk">Gün tamam sayılır: kalori hedefin %${fmt(TAMAM_ALT * 100)}–${fmt(TAMAM_UST * 100)}'i, protein en az %${fmt(PROTEIN_ESIK * 100)}'i. Tahminler ortalama ±40 kcal sapar; tıbbi tavsiye değildir.</p></div>`;
    kutu.querySelectorAll(".sil").forEach((b) => b.addEventListener("click", () => {
      kayit.splice(Number(b.dataset.i), 1);
      if (!kayit.length) delete bellek.gunler[bugun];
      yaz(); cizBugun();
    }));
    kutu.querySelectorAll(".ogun-ekle").forEach((b) => b.addEventListener("click", () => { bekleyenOgun = b.dataset.ogun; git("ekle"); }));
  }

  // ---------- Alışkanlıklar ----------
  function cizAliskanlik() {
    const kutu = $("aliskanlik-icerik"), h = hedef(), v = aliskanlikVerisi(bellek.gunler, h);
    if (!v.kayitli.length) {
      kutu.innerHTML = `<div class="kart bos-tabak"><h2>Henüz alışkanlık çıkaracak kadar kayıt yok</h2>` +
        `<p>Birkaç gün öğünlerini ekle; ne zaman ve ne kadar yediğini burada göreceksin.</p><a class="buton birincil" href="#ekle">İlk öğününü ekle</a></div>`;
      return;
    }
    const enCok = Math.max(...v.gunler7.map((g) => toplam(bellek.gunler[g]).kcal), h ? h.kcal * 1.2 : 0, 1);
    const adlar = ["Paz", "Pzt", "Sal", "Çar", "Per", "Cum", "Cmt"];
    const sutunlar = v.gunler7.map((g, i) => {
      const t = toplam(bellek.gunler[g]), d = tarihOf(g);
      const durum = !bellek.gunler[g]?.length ? "yok" : h && gunTamamMi(t, h) ? "tamam" : "";
      return `<div class="gun-sutun ${durum}${i === 6 ? " bugun" : ""}" title="${d.getDate()}.${d.getMonth() + 1}: ${fmt(t.kcal)} kcal">` +
        `<div class="bar" style="height:${(100 * t.kcal / enCok).toFixed(1)}%"></div></div>`;
    }).join("");
    const etiketler = v.gunler7.map((g, i) => `<small${i === 6 ? ' class="bugun"' : ""}>${i === 6 ? "Bugün" : adlar[tarihOf(g).getDay()]}</small>`).join("");
    const hedefCizgi = h ? `<div class="hedef-cizgi" style="bottom:${(100 * h.kcal / enCok).toFixed(1)}%"><span>hedef ${fmt(h.kcal)}</span></div>` : "";
    const dagilim = OGUNLER.map(([k]) => `<span style="width:${(100 * v.ogunKcal[k] / (v.toplamKcal || 1)).toFixed(1)}%;background:${OGUN_RENK[k]}"></span>`).join("");
    const dagilimListe = OGUNLER.map(([k, ad, ikon]) => `<li><i style="background:${OGUN_RENK[k]}"></i>${ikon} ${ad}<b>%${fmt(100 * v.ogunKcal[k] / (v.toplamKcal || 1))}</b></li>`).join("");
    const olcut = (ad, deger, not) => `<li><span>${ad}</span><b>${deger}</b>${not ? `<small>${not}</small>` : ""}</li>`;
    const olcutler = [
      olcut("Kayıt tuttuğun gün", `${v.kayitli.length} / 7`, v.kayitli.length < 5 ? "Her gün eklersen alışkanlıkların daha net görünür." : ""),
      olcut("Günlük ortalama", `${fmt(v.ortKcal)} kcal`, h ? `Hedefin ${fmt(h.kcal)} kcal` : ""),
      h ? olcut("Ortalama protein", `${fmt(v.ortP)} g`, `Hedefin %${fmt(100 * v.ortP / h.p)}'i`) : "",
      h ? olcut("Hedefi tutturduğun gün", `${v.tamamGun} / 7`, "") : "",
      h ? olcut("En uzun seri", `🔥 ${enUzunSeri(bellek.gunler, h)} gün`, "") : "",
    ].join("");
    kutu.innerHTML = `<div class="aliskanlik-izgara">` +
      `<section class="kart genis"><h2>Son 7 gün</h2><p class="aciklama">Günlük kalori; turuncu günlerde hedefi tutturdun.</p>` +
      `<div class="hafta-grafik">${hedefCizgi}${sutunlar}</div><div class="hafta-etiket">${etiketler}</div>` +
      `<div class="lejant-satir"><span><i style="background:var(--alev)"></i>Hedef tamam</span><span><i style="background:color-mix(in srgb, var(--vurgu) 55%, var(--iz))"></i>Kayıt var</span><span><i style="background:var(--iz)"></i>Kayıt yok</span></div></section>` +
      `<section class="kart"><h2>Ne fark ettik</h2><p class="aciklama">Son 7 günün kayıtlarından.</p><ul class="icgoru">${icgoruler(v, h).map(([i, m]) => `<li><span aria-hidden="true">${i}</span><div>${m}</div></li>`).join("")}</ul></section>` +
      `<section class="kart"><h2>Kalori hangi öğünden geliyor</h2><p class="aciklama">Son 7 günün toplamı.</p><div class="dagilim" role="img" aria-label="Öğünlere göre kalori dağılımı">${dagilim}</div><ul class="dagilim-liste">${dagilimListe}</ul>` +
      `<ul class="olcutler" style="margin-top:16px">${olcutler}</ul></section>` +
      `</div>`;
  }

  // ---------- Hedefim ----------
  function cizProfil() {
    const kutu = $("profil-form-kutu"), pr = bellek.profil || { cinsiyet: "k", yas: "", boy: "", kilo: "", aktivite: "1.375", hedef: "koru" }, h = hedef();
    const radyo = (ad, baslik, secenekler, deger, sinif = "") => `<fieldset class="secenek-grup ${sinif}"><legend>${baslik}</legend><div class="secenekler">` +
      Object.entries(secenekler).map(([v, e]) => `<label><input type="radio" name="${ad}" value="${v}"${String(v) === String(deger) ? " checked" : ""}>` +
        `<span>${Array.isArray(e) ? e[0] : e}${Array.isArray(e) && typeof e[1] === "string" ? `<small>${e[1]}</small>` : ""}</span></label>`).join("") + `</div></fieldset>`;
    kutu.innerHTML = `<form id="profil-form" class="profil">` +
      radyo("cinsiyet", "Cinsiyet", { k: "Kadın", e: "Erkek" }, pr.cinsiyet, "dort") +
      `<label>Yaş<input name="yas" type="number" inputmode="numeric" min="14" max="100" required value="${pr.yas}"></label>` +
      `<label>Boy (cm)<input name="boy" type="number" inputmode="numeric" min="120" max="230" required value="${pr.boy}"></label>` +
      `<label class="genis">Kilo (kg)<input name="kilo" type="number" inputmode="decimal" min="30" max="250" step="0.1" required value="${pr.kilo}"></label>` +
      radyo("aktivite", "Ne kadar hareketlisin?", AKTIVITE, pr.aktivite, "dort") +
      radyo("hedef", "Hedefin ne?", Object.fromEntries(Object.entries(HEDEF).map(([k, v]) => [k, v[0]])), pr.hedef) +
      (h ? `<div class="hedef-ozet" aria-label="Günlük hedefin"><div><b>${fmt(h.kcal)}</b><small>kcal</small></div><div><b>${fmt(h.p)} g</b><small>protein</small></div>` +
        `<div><b>${fmt(h.k)} g</b><small>karbonhidrat</small></div><div><b>${fmt(h.y)} g</b><small>yağ</small></div></div>` : "") +
      `<button type="submit" class="buton birincil genis">${h ? "Hedefimi güncelle" : "Hedefimi hesapla"}</button>` +
      (bellek.profil ? `<p class="kucuk genis" style="margin:0">Bazal metabolizman ${fmt(h.bmr)} kcal. <button type="button" class="baglanti" id="sifirla">Tüm kayıtları sil</button></p>` : "") +
      `</form>`;
    $("profil-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const f = new FormData(ev.target);
      bellek.profil = { cinsiyet: f.get("cinsiyet"), yas: Number(f.get("yas")), boy: Number(f.get("boy")), kilo: Number(f.get("kilo")), aktivite: f.get("aktivite"), hedef: f.get("hedef") };
      yaz(); cizProfil(); cizBugun(); git("bugun");
    });
    const s = $("sifirla");
    if (s) s.addEventListener("click", () => {
      if (!confirm("Profilin ve tüm öğün kayıtların bu cihazdan silinsin mi?")) return;
      bellek = { profil: null, gunler: {} }; yaz(); cizProfil(); cizBugun();
    });
  }

  // ---------- tahmin sonucuna "Öğüne ekle" ----------
  const PORSIYON = [[0.5, "Yarısı"], [1, "Tamamı"], [1.5, "Bir buçuk"], [2, "İki porsiyon"]];
  const porsiyonAd = (o) => (PORSIYON.find((x) => x[0] === o) || [o, `×${fmt(o, 1)}`])[1];
  function ekleKutusu(p) { // p = [kcal, kütle g, yağ g, karbonhidrat g, protein g]
    const kutu = document.createElement("div");
    kutu.className = "ekle-kutu";
    let ogun = bekleyenOgun || ogunSaatten(new Date().getHours()), porsiyon = 1;
    bekleyenOgun = null;
    const grup = (ad, baslik, liste, secili) => `<div class="secim" role="group" aria-label="${baslik}" data-grup="${ad}"><span>${baslik}</span>` +
      liste.map(([v, e]) => `<button type="button" data-v="${v}" aria-pressed="${String(v) === String(secili)}">${e}</button>`).join("") + `</div>`;
    kutu.innerHTML = grup("ogun", "Hangi öğün?", OGUNLER.map(([k, ad, i]) => [k, `${i} ${ad}`]), ogun) +
      grup("porsiyon", "Ne kadarını yedin?", PORSIYON, porsiyon) +
      `<button type="button" class="buton birincil buyuk ekle-dugme"></button>`;
    const dugme = kutu.querySelector(".ekle-dugme");
    const etiket = () => { dugme.textContent = `${ogunAd(ogun)[3]} ekle (${fmt(p[0] * porsiyon)} kcal)`; };
    kutu.querySelectorAll(".secim button").forEach((b) => b.addEventListener("click", () => {
      const g = b.parentElement.dataset.grup;
      if (g === "ogun") ogun = b.dataset.v; else porsiyon = Number(b.dataset.v);
      b.parentElement.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b));
      etiket();
    }));
    etiket();
    dugme.addEventListener("click", () => {
      const h = hedef(), once = h ? seri(bellek.gunler, h).bugunTamam : false;
      const g = gunAnahtari(), s = new Date();
      (bellek.gunler[g] ||= []).push({
        saat: `${String(s.getHours()).padStart(2, "0")}:${String(s.getMinutes()).padStart(2, "0")}`, ogun, porsiyon,
        kcal: Math.round(p[0] * porsiyon), g: Math.round(p[1] * porsiyon), y: +(p[2] * porsiyon).toFixed(1), k: +(p[3] * porsiyon).toFixed(1), p: +(p[4] * porsiyon).toFixed(1),
      });
      yaz();
      const sonra = h ? seri(bellek.gunler, h).bugunTamam : false;
      dugme.disabled = true; dugme.textContent = `${ogunAd(ogun)[3]} eklendi ✓`;
      kutu.querySelectorAll(".secim button").forEach((x) => { x.disabled = true; });
      cizBugun(!once && sonra);
      setTimeout(() => git("bugun"), 650);
    });
    return kutu;
  }

  function kur() {
    oku();
    cizProfil(); cizBugun();
    window.addEventListener("hashchange", () => git(location.hash.slice(1)));
    git(location.hash.slice(1) || "bugun", false);
    let son = gunAnahtari();
    setInterval(() => { const g = gunAnahtari(); if (g !== son) { son = g; cizBugun(); } }, 60000);
  }

  window.Gunluk = {
    kur, ekleKutusu, git, hedefKcal: () => (hedef() ? hedef().kcal : null),
    _test: { hedefHesapla, toplam, gunTamamMi, seri, enUzunSeri, oneriler, aliskanlikVerisi, icgoruler, gunOnce, gunAnahtari, ogunSaatten },
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", kur); else kur();
})();
