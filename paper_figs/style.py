"""Common figure style for the paper (Copernicus manuscripts are set in Times, so figures use the Times-compatible Nimbus Roman / Times New Roman).
Usage: import style; style.apply(); ...; style.panel_labels(axes)  (axes in row-major order)."""
import string
import matplotlib.pyplot as plt

FONT = ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"]


def apply(size=10, tick=None, legend=None):
    """Times-compatible font, ticks pointing inside on all four sides. size scales all text; tick and legend default to 0.9 and 0.75 of size."""
    tick = tick or 0.9 * size; legend = legend or 0.75 * size
    plt.rcParams.update({
        "font.family": "serif", "font.serif": FONT, "mathtext.fontset": "stix",
        "font.size": size, "axes.labelsize": size, "axes.titlesize": size, "xtick.labelsize": tick, "ytick.labelsize": tick, "legend.fontsize": legend,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.minor.top": True, "ytick.minor.right": True,
    })


def panel_labels(axes, size=None, x=-0.02, y=1.02):
    """a), b), c), ... above the top-left corner of each axes (row-major order)."""
    size = size or plt.rcParams["font.size"] * 1.2
    for k, a in enumerate(axes):
        a.text(x, y, string.ascii_lowercase[k] + ")", transform=a.transAxes, va="bottom", ha="left", fontsize=size)
