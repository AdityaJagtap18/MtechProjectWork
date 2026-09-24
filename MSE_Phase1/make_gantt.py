"""Gantt chart for the Timeline slide (July - December, project work only).

Run from the project root:
    .venv/bin/python dissertation_ppt/make_gantt.py
Edit TASKS to change bars; start/end are in months from 1 July (0.0 = Jul 1, 6.0 = Dec 31).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "dissertation_ppt/TemplatePPT/timeline_gantt.png"

MONTHS = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# (label, start, end)  -- months from 1 July
TASKS = [
    ("Literature Review", 0.0, 1.5),
    ("Problem Understanding", 0.5, 1.5),
    ("Dataset & Graph Setup", 1.5, 2.5),
    ("Classical Baseline", 2.0, 3.0),
    ("QGNN Implementation", 2.5, 4.0),
    ("Experiments & Comparison", 3.5, 5.0),
    ("Analysis & Evaluation", 4.5, 5.5),
    ("Report Writing", 5.0, 6.0),
]

BAR = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dcdcd8"
BAND = "#f3f3f1"

fig, ax = plt.subplots(figsize=(13, 7.2), dpi=250)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

n = len(TASKS)
ax.set_xlim(0, 6)
ax.set_ylim(n - 0.4, -1.1)  # top row reserved for month header

# alternating month bands + header labels
for i, m in enumerate(MONTHS):
    if i % 2 == 0:
        ax.axvspan(i, i + 1, ymin=0, ymax=1, color=BAND, zorder=0)
    ax.text(i + 0.5, -0.75, m, ha="center", va="center", fontsize=21,
            fontweight="bold", color=INK)
for x in range(7):
    ax.axvline(x, color=GRID, lw=1.2, zorder=1)
ax.axhline(-0.4, color=INK, lw=1.6, zorder=2)

# bars: thin, rounded, 2px surface gap between neighbours
h = 0.46
for row, (label, s, e) in enumerate(TASKS):
    ax.add_patch(FancyBboxPatch(
        (s + 0.02, row - h / 2), (e - s) - 0.04, h,
        boxstyle="round,pad=0,rounding_size=0.12",
        mutation_aspect=1 / 6, fc=BAR, ec="none", zorder=3))

ax.set_yticks(range(n))
ax.set_yticklabels([t[0] for t in TASKS], fontsize=19, color=INK)
ax.tick_params(axis="y", length=0, pad=14)
ax.set_xticks([])
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)

fig.subplots_adjust(left=0.315, right=0.985, top=0.97, bottom=0.03)
fig.savefig(OUT, facecolor="white")
print("saved", OUT)
