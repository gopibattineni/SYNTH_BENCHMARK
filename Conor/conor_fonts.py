"""Shared Times New Roman / Times-compatible serif font for Conor figures.

Prefers Times New Roman when installed; otherwise Liberation Serif
(metric-compatible Times substitute), then Nimbus Roman / FreeSerif.
"""

from __future__ import annotations

from matplotlib import font_manager, pyplot as plt

_CANDIDATES = (
    "Times New Roman",
    "TimesNewRoman",
    "Liberation Serif",
    "Nimbus Roman",
    "Nimbus Roman No9 L",
    "FreeSerif",
    "DejaVu Serif",
)


def configure_times_font() -> str:
    """Set global matplotlib rcParams to a Times-like serif face. Returns chosen name."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in _CANDIDATES if c in available), "DejaVu Serif")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [chosen, "Liberation Serif", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    return chosen


def apply_font_to_figure(fig: plt.Figure, font_name: str | None = None) -> None:
    """Force every text artist on a figure onto the chosen serif face."""
    if font_name is None:
        font_name = configure_times_font()
    for ax in fig.axes:
        for text in ax.texts:
            text.set_fontfamily("serif")
            text.set_fontname(font_name)
        if ax.title is not None:
            ax.title.set_fontfamily("serif")
            ax.title.set_fontname(font_name)
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            label.set_fontfamily("serif")
            label.set_fontname(font_name)
        xlab = ax.xaxis.label
        ylab = ax.yaxis.label
        if xlab is not None:
            xlab.set_fontfamily("serif")
            xlab.set_fontname(font_name)
        if ylab is not None:
            ylab.set_fontfamily("serif")
            ylab.set_fontname(font_name)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontfamily("serif")
                text.set_fontname(font_name)
            if legend.get_title() is not None:
                legend.get_title().set_fontfamily("serif")
                legend.get_title().set_fontname(font_name)
    for text in fig.texts:
        text.set_fontfamily("serif")
        text.set_fontname(font_name)
