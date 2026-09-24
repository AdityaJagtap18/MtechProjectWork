"""Slide figures grounded in the QGNN research report.

  system_diagram.png  <- report Fig. 1 (overall architecture), redrawn for slides
  scm_graph.png       <- report Fig. 3 / Sec. 4 (heterogeneous SCM graph), drawn node-level

Run from the project root:
    .venv/bin/python dissertation_ppt/make_diagrams.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = "dissertation_ppt/TemplatePPT/"
INK = "#1a1a1a"

# report palette (fill, edge)
BLUE = ("#E8EEF8", "#4C72B0")
GREEN = ("#E8F3EC", "#55A868")
ORANGE = ("#FCEEE1", "#DD8452")
PURPLE = ("#F3E9F5", "#8172B3")
GREY = ("#EEEEEE", "#808080")


# ------------------------------------------------------------------ system diagram
def system_diagram():
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=250)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    def box(cx, cy, w, h, text, col, bold=False, fs=15):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.18",
                                    fc=col[0], ec=col[1], lw=2.6, zorder=3))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, zorder=4,
                linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=22, lw=2.4, color="#444", zorder=2))

    box(5, 6.75, 8.6, 1.0,
        "Synthetic supply-chain data  →  graph construction", BLUE)
    arrow(5, 6.22, 5, 5.72)
    box(5, 5.2, 8.6, 1.0,
        "Node and edge features\n(static attributes + leakage-safe rolling aggregates)", BLUE)
    arrow(5, 4.67, 5, 4.17)
    box(5, 3.65, 8.6, 1.0,
        "GraphSAGE encoder\nSupplier embedding  (frozen)", GREEN, bold=True)
    arrow(2.9, 3.12, 2.9, 2.72)
    arrow(7.1, 3.12, 7.1, 2.72)
    box(2.9, 2.2, 3.8, 1.0, "Classical head\nMLP classifier", GREEN)
    box(7.1, 2.2, 3.8, 1.0, "QGNN head\nvariational quantum circuit", ORANGE)
    arrow(2.9, 1.67, 4.2, 1.22)
    arrow(7.1, 1.67, 5.8, 1.22)
    box(5, 0.75, 8.6, 0.9,
        "Risk probability  →  evaluation on temporal and severity (OOD) splits",
        PURPLE, bold=True, fs=14)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
    fig.savefig(OUT + "system_diagram.png", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ SCM graph
def scm_graph():
    fig, ax = plt.subplots(figsize=(10, 6.3), dpi=250)
    ax.set_xlim(0, 10)
    ax.set_ylim(1.15, 7.45)
    ax.set_aspect("equal")  # circles stay circles
    ax.axis("off")

    C = {  # type -> (colour, legend label)
        "S": ("#4C72B0", "Supplier (300)"),
        "M": ("#55A868", "Material (100)"),
        "P": ("#DD8452", "Plant (50)"),
        "D": ("#8172B3", "Product (200)"),
        "R": ("#C44E52", "Region (20)"),
        "O": ("#CCB974", "Procurement order (2,000)"),
    }
    pos = {}
    for i, y in enumerate((5.5, 4.6, 3.7, 2.8)):
        pos[f"S{i}"] = (1.2, y)
    for i, y in enumerate((5.0, 4.1, 3.2)):
        pos[f"M{i}"] = (3.9, y)
        pos[f"P{i}"] = (6.4, y)
        pos[f"D{i}"] = (8.8, y)
    pos["R0"], pos["R1"] = (2.6, 6.9), (7.6, 6.9)
    for i, x in enumerate((2.2, 4.4, 6.6, 8.6)):
        pos[f"O{i}"] = (x, 1.75)

    edges = (
        # supplier -> material -> plant -> product
        [("S0", "M0"), ("S1", "M0"), ("S1", "M1"), ("S2", "M1"), ("S2", "M2"), ("S3", "M2"),
         ("M0", "P0"), ("M1", "P1"), ("M2", "P1"), ("M2", "P2"),
         ("P0", "D0"), ("P1", "D1"), ("P1", "D2"), ("P2", "D2")]
        # region links
        + [("S0", "R0"), ("S1", "R0"), ("P0", "R1"), ("P1", "R1"), ("D0", "R1"), ("D1", "R1")]
        # procurement orders
        + [("S2", "O0"), ("S3", "O1"),
           ("O0", "M2"), ("O1", "M2"), ("O2", "M1"),
           ("O1", "P2"), ("O2", "P2"), ("O3", "P1")]
    )
    r = 0.33
    for a_, b_ in edges:
        ax.add_patch(FancyArrowPatch(pos[a_], pos[b_], arrowstyle="-|>", mutation_scale=14,
                                     lw=1.6, color="#9a9a9a", shrinkA=r * 72 * 0.62 * 2.2 / 2.2,
                                     shrinkB=r * 72 * 0.62 * 2.2 / 2.2, zorder=1))
    for n, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), r, fc=C[n[0]][0], ec="white", lw=2.4, zorder=3))

    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
    fig.savefig(OUT + "scm_graph.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    system_diagram()
    scm_graph()
    print("saved system_diagram.png, scm_graph.png")
