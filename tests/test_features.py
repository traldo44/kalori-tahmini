import numpy as np
from PIL import Image

from src.features import HIST_BINS, hog_color_features


def test_hog_color_shape_and_histograms():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8))
    f = hog_color_features(img)
    assert f.shape == (1764 + 3 * HIST_BINS,) and f.dtype == np.float32
    hist = f[-3 * HIST_BINS:].reshape(3, HIST_BINS)
    assert np.allclose(hist.sum(1), 1)          # her kanalın histogramı olasılık dağılımı
    assert np.array_equal(f, hog_color_features(img))  # deterministik


def test_grayscale_image_is_handled():
    assert hog_color_features(Image.new("L", (300, 200), 128)).shape == (1764 + 3 * HIST_BINS,)


def test_classic_ml_never_imports_torch():
    """torch + xgboost aynı süreçte macOS'ta çöküyor; classic_ml torch'u hiç yüklememeli."""
    import subprocess
    import sys
    code = "import sys, src.classic_ml; assert 'torch' not in sys.modules, 'torch yüklendi!'"
    subprocess.run([sys.executable, "-c", code], check=True)
