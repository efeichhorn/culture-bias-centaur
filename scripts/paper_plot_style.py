"""Shared tueplots typography and palette for the manuscript figures."""

from tueplots import bundles


COLORS = {
    "us": "#4878A8",
    "china": "#CF866F",
    "pooled": "#8E82A8",
    "frozen": "#B9BEC4",
}


def paper_style(height: float, nrows: int = 1, ncols: int = 2) -> dict:
    """NeurIPS typography on a 6.3-inch-wide, two-column manuscript canvas."""
    style = bundles.neurips2024(
        usetex=False, family="serif", nrows=nrows, ncols=ncols,
    )
    style.update({
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "figure.figsize": (6.3, height),
        "figure.constrained_layout.w_pad": .055,
        "figure.constrained_layout.h_pad": .055,
        "figure.constrained_layout.wspace": .06,
        "figure.constrained_layout.hspace": .06,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "black",
        "axes.linewidth": .5,
        "axes.grid": False,
        "axes.labelcolor": "black",
        "axes.titleweight": "normal",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": .5,
        "ytick.major.width": .5,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })
    return style
