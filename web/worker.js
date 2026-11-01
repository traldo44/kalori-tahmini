/* Çıkarım iş parçacığı: model yüklemesi ve tahmin burada çalışır, sayfa donmaz.
   Mesajlar: {tip:"yukle", veri:ArrayBuffer} → {tip:"hazir"} | {tip:"hata", mesaj}
             {tip:"tahmin", veri:Float32Array, sekil:[1,3,224,224]} → {tip:"sonuc", p:[kcal, g, yağ, karb, protein], ms} */
"use strict";
const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
importScripts(ORT_CDN + "ort.min.js");
ort.env.wasm.wasmPaths = ORT_CDN;
ort.env.wasm.numThreads = 1;   // SharedArrayBuffer gerektirmesin; tek iş parçacığı telefonda da güvenli

let oturum = null;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.tip === "yukle") {
      oturum = await ort.InferenceSession.create(new Uint8Array(m.veri), { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      self.postMessage({ tip: "hazir" });
    } else if (m.tip === "tahmin") {
      if (!oturum) throw new Error("model henüz hazır değil");
      const t = performance.now();
      const cikti = await oturum.run({ image: new ort.Tensor("float32", m.veri, m.sekil) });
      self.postMessage({ tip: "sonuc", p: Array.from(cikti.pred.data), ms: performance.now() - t });
    }
  } catch (err) {
    self.postMessage({ tip: "hata", mesaj: String(err && err.message || err) });
  }
};
