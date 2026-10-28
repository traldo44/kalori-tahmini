"""ORTAK grafik stili: rapordaki tüm figürler aynı görünsün.

    from src.plot_style import apply, SERIES, INK, savefig
    apply()

Kurallar: renkler SERIES sırasıyla atanır (atlanmaz, döngüye girmez); saçılım grafiklerinde en
fazla 3 seri (renk körü okuyucular için yalnızca ilk 3 renk birbirinden ayırt edilebilir);
tek seride lejant yok; çift y-ekseni yok; metinler seri renginde değil, mürekkep renginde.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from .config import RESULTS_DIR

SURFACE = "#fcfcfb"
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781"}
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Renk körlüğü testinden geçmiş kategorik palet (sıra önemli)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Tek renkli büyüklük skalası (ısı haritaları): açık → koyu mavi
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "text.color": INK["primary"], "axes.labelcolor": INK["secondary"],
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 2, "legend.frameon": False,
    })


def sequential_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)


def savefig(fig, name, dpi=None):
    """results/figures/<name>.png olarak kaydeder. Fotoğraf ızgaraları için dpi=110 yeter (dosya küçük kalır)."""
    path = RESULTS_DIR / "figures" / f"{name}.png"
    fig.savefig(path, dpi=dpi)  # None → rcParams["savefig.dpi"] (200)
    return path
