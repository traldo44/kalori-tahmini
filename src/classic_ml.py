"""Adım 9 — klasik ML: Tablo 1 ve 2'nin klasik satırları.

    python -m src.classic_ml                 # hepsi (özellikler önbelleğe alınır: data/features/)
    python -m src.classic_ml --only cls      # yalnızca Food-101
    python -m src.classic_ml --only calorie  # yalnızca Nutrition5k

Sınıflandırma (Food-101): HOG + renk → SVM (RBF), Random Forest, k-NN · CNN gömmesi → lojistik regresyon
Kalori (Nutrition5k):    CNN gömmesi → Ridge, Random Forest, XGBoost · HOG + renk → XGBoost
Her modelin hiperparametresi YALNIZCA val'de seçilir (sınıflandırma: macro-F1, kalori: MAE); seçilen ayarla
train'de eğitilmiş model val ve test'te ölçülür, results.csv'ye yazılır.
"""
import argparse
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBRegressor

# Bu süreç torch'u YÜKLEMEMELİ (xgboost ile çakışır, bkz. src/features.py). src.data / src.transforms içe aktarılmaz;
# özellikler ayrı süreçte hesaplanıp .npz'den okunur.
from .config import CALORIE_BIN_EDGES, SEED
from .features import load
from .metrics import classification_metrics, regression_metrics, to_bins
from .results import log_result, save_preds

NOTE = "Adım 9 klasik ML"
PCA_DIM = 256


# ---------------------------------------------------------------- değerlendirme

def scores_of(model, X):
    return model.predict_proba(X) if hasattr(model, "predict_proba") else model.decision_function(X)


def run_classifier(name, data, grid, make):
    """grid: [(config_adı, parametreler)]. Val macro-F1'e göre en iyisini seçer, val+test'i kaydeder."""
    (Xtr, ytr, _), (Xva, yva, ids_va), (Xte, yte, ids_te) = data["train"], data["val"], data["test"]
    best, tried = None, []
    for cfg, params in grid:
        t = time.time()
        model = make(**params).fit(Xtr, ytr)
        f1 = classification_metrics(yva, model.predict(Xva))["f1"]
        tried.append(f"{cfg}→{f1:.3f}")
        print(f"    {name} {cfg}: val F1 {f1:.3f} ({time.time() - t:.0f} sn)", flush=True)
        if best is None or f1 > best[0]:
            best = (f1, cfg, model)
    _, cfg, model = best
    for split, X, y, ids in (("val", Xva, yva, ids_va), ("test", Xte, yte, ids_te)):
        m = classification_metrics(y, model.predict(X), scores_of(model, X), ci=True)
        log_result("cls", name, cfg, split, m, notes=f"{NOTE}; val F1: {', '.join(tried)}")
        save_preds(name, ids, y, model.predict(X), split=split)
        print(f"  ✓ {name} [{cfg}] {split}: acc {m['accuracy']:.3f}  F1 {m['f1']:.3f} "
              f"[{m['f1_ci_low']:.3f}–{m['f1_ci_high']:.3f}]  top5 {m['top5']:.3f}", flush=True)


def run_regressor(name, data, grid, make, fit_kwargs=None):
    """Val MAE'ye göre en iyi ayarı seçer. Kalori negatif olamaz → tahminler 0'dan kırpılır."""
    (Xtr, ytr, _), (Xva, yva, ids_va), (Xte, yte, ids_te) = data["train"], data["val"], data["test"]
    best, tried = None, []
    for cfg, params in grid:
        t = time.time()
        model = make(**params)
        model.fit(Xtr, ytr, **(fit_kwargs(Xva, yva) if fit_kwargs else {}))
        mae = regression_metrics(yva, np.clip(model.predict(Xva), 0, None))["MAE"]
        tried.append(f"{cfg}→{mae:.1f}")
        print(f"    {name} {cfg}: val MAE {mae:.1f} ({time.time() - t:.0f} sn)", flush=True)
        if best is None or mae < best[0]:
            best = (mae, cfg, model)
    _, cfg, model = best
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        cfg += f",n_trees={model.best_iteration + 1}"
    for split, X, y, ids in (("val", Xva, yva, ids_va), ("test", Xte, yte, ids_te)):
        pred = np.clip(model.predict(X), 0, None)
        m = regression_metrics(y, pred, ci=True)
        log_result("calorie", name, cfg, split, m, notes=f"{NOTE}; val MAE: {', '.join(tried)}")
        bins = classification_metrics(to_bins(y, CALORIE_BIN_EDGES), to_bins(pred, CALORIE_BIN_EDGES), ci=True)
        log_result("calorie_bins", name, cfg, split, bins, notes=NOTE)
        save_preds(name, ids, y, pred, split=split)
        print(f"  ✓ {name} [{cfg}] {split}: MAE {m['MAE']:.1f} [{m['MAE_ci_low']:.1f}–{m['MAE_ci_high']:.1f}]  "
              f"PMAE %{m['PMAE']:.1f}  R² {m['R2']:.2f}  | Tablo 3 F1 {bins['f1']:.3f}", flush=True)


# ---------------------------------------------------------------- deneyler

def classification():
    print("Food-101 · HOG + renk", flush=True)
    hog = load("food101", "hogcolor")
    pca = lambda: PCA(PCA_DIM, random_state=SEED)  # noqa: E731
    run_classifier("hogcolor_svm_rbf", hog, [(f"C={c}", {"C": c}) for c in (1, 10)],
                   lambda C: make_pipeline(StandardScaler(), pca(), SVC(C=C, gamma="scale", random_state=SEED)))
    run_classifier("hogcolor_rf", hog, [(f"min_leaf={k}", {"k": k}) for k in (1, 3)],
                   lambda k: RandomForestClassifier(500, min_samples_leaf=k, n_jobs=-1, random_state=SEED))
    run_classifier("hogcolor_knn", hog, [(f"k={k}", {"k": k}) for k in (5, 15, 31)],
                   lambda k: make_pipeline(StandardScaler(), pca(), KNeighborsClassifier(k, weights="distance")))
    print("Food-101 · CNN gömmesi", flush=True)
    cnn = load("food101", "cnn")
    run_classifier("cnn_logreg", cnn, [(f"C={c}", {"C": c}) for c in (0.01, 0.1, 1)],
                   lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)))


def calorie():
    print("Nutrition5k · CNN gömmesi", flush=True)
    cnn = load("n5k", "cnn")
    run_regressor("cnn_ridge", cnn, [(f"alpha={a:g}", {"a": a}) for a in (10, 100, 1000, 10000, 100000)],
                  lambda a: make_pipeline(StandardScaler(), Ridge(alpha=a)))
    run_regressor("cnn_rf", cnn, [(f"min_leaf={k}", {"k": k}) for k in (1, 5)],
                  lambda k: RandomForestRegressor(500, min_samples_leaf=k, max_features=0.33, n_jobs=-1, random_state=SEED))
    xgb = lambda d: XGBRegressor(objective="reg:absoluteerror", n_estimators=3000, learning_rate=0.03, max_depth=d,  # noqa: E731
                                 subsample=0.8, colsample_bytree=0.5, early_stopping_rounds=100, eval_metric="mae",
                                 random_state=SEED, n_jobs=-1)
    early_stop = lambda Xva, yva: {"eval_set": [(Xva, yva)], "verbose": False}  # noqa: E731
    run_regressor("cnn_xgb", cnn, [(f"max_depth={d}", {"d": d}) for d in (4, 6)], xgb, early_stop)
    print("Nutrition5k · HOG + renk", flush=True)
    hog = load("n5k", "hogcolor")
    run_regressor("hogcolor_xgb", hog, [(f"max_depth={d}", {"d": d}) for d in (4, 6)], xgb, early_stop)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cls", "calorie"])
    args = ap.parse_args()
    if args.only != "cls":
        calorie()
    if args.only != "calorie":
        classification()
