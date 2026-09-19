"""Shared drawing helpers for the QGNN-v4 final report's architecture/flow
diagrams (matplotlib patches, not a diagramming library) -- kept simple
and consistent so every figure in the report shares the same visual
language (box style, arrow style, color palette, font sizes)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# Palette -- colorblind-safe-ish, consistent across every diagram.
COLOR_DATA = "#E8EEF7"       # light blue: data / input
COLOR_DATA_EDGE = "#4C72B0"
COLOR_CLASSICAL = "#E9F5EC"  # light green: classical components
COLOR_CLASSICAL_EDGE = "#55A868"
COLOR_QUANTUM = "#FDF0E3"    # light orange: quantum components
COLOR_QUANTUM_EDGE = "#DD8452"
COLOR_FROZEN = "#EFEFEF"     # light grey: frozen / non-trainable
COLOR_FROZEN_EDGE = "#7F7F7F"
COLOR_OUTPUT = "#F5E9F5"     # light purple: output / decision
COLOR_OUTPUT_EDGE = "#8172B3"
COLOR_TEXT = "#1a1a1a"


# Canvas is drawn smaller than the printed size so that fonts (fixed in points)
# stay legible when the figure is scaled to the A4 text width.
FIG_SCALE = 0.88
HEADER_BAND = 11  # extra y-units above the 0-100 drawing area, reserved for title/subtitle


def new_fig(width=8.5, height=11, dpi=200):
    fig, ax = plt.subplots(figsize=(width * FIG_SCALE, height * FIG_SCALE * (100 + HEADER_BAND) / 100), dpi=dpi)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 + HEADER_BAND)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, face=COLOR_DATA, edge=COLOR_DATA_EDGE, fontsize=9.5, fontweight="normal", textcolor=COLOR_TEXT, linewidth=1.6):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                        facecolor=face, edgecolor=edge, linewidth=linewidth)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=textcolor, wrap=True, linespacing=1.35)
    return b


def arrow(ax, x1, y1, x2, y2, color="#444444", style="-|>", lw=1.6, connectionstyle=None):
    kwargs = dict(arrowstyle=style, mutation_scale=13, color=color, linewidth=lw)
    if connectionstyle:
        kwargs["connectionstyle"] = connectionstyle
    a = FancyArrowPatch((x1, y1), (x2, y2), **kwargs)
    ax.add_patch(a)
    return a


def down_arrow(ax, x, y_top, y_bottom, color="#444444", lw=1.6):
    return arrow(ax, x, y_top, x, y_bottom, color=color, lw=lw)


def label(ax, x, y, text, fontsize=8.5, ha="center", va="center", style="normal", color=COLOR_TEXT, fontweight="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, style=style, color=color, fontweight=fontweight)


def title(ax, text, x=50, y=None, fontsize=13):
    # Always drawn in the reserved header band above the diagram area.
    ax.text(x, 100 + HEADER_BAND - 0.5, text, ha="center", va="top", fontsize=fontsize, fontweight="bold")


def subtitle(ax, text, fontsize=8.5):
    ax.text(50, 100 + HEADER_BAND - 5.2, text, ha="center", va="top", fontsize=fontsize, color=COLOR_TEXT, linespacing=1.3)


def legend_swatch(ax, x, y, w, h, face, edge, text, fontsize=8):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.5", facecolor=face, edgecolor=edge, linewidth=1.2)
    ax.add_patch(b)
    ax.text(x + w + 1.2, y + h / 2, text, ha="left", va="center", fontsize=fontsize)


def save(fig, path):
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
