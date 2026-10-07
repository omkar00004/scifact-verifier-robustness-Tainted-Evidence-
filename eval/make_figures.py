#!/usr/bin/env python3
"""README figures, drawn from eval/results/results.csv (no API calls).
Every plotted statistic is computed with analyze.ci and asserted to appear verbatim in results/summary.md.
Usage: eval/.venv/bin/python eval/make_figures.py    (writes eval/results/figures/*.png)"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import analyze

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "figures"
SUMMARY = (HERE / "results" / "summary.md").read_text()

# Light-surface tokens of the reference palette; P0 and P1 are fixed entities (categorical slots 1 and 2, validated:
# worst colour-vision separation dE 24.7, contrast >= 3:1). Text only ever wears the ink tokens.
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"P0": "#2a78d6", "P1": "#eb6834", "P1-P0": INK2}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "axes.unicode_minus": True, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2})
MS, LW = 9, 2.2  # marker diameter (pt) and line width (pt), >= 8 px / 2 px at 100 dpi

cell = {}  # (condition, prompt) -> {claim_id: row}, parsed outputs only
for r in csv.DictReader(open(HERE / "results" / "results.csv")):
    if r["parsed"] == "True":
        cell.setdefault((r["cond"], r["pid"]), {})[r["claim_id"]] = r
N = len(cell[("E0", "P0")])
correct = lambda r: r["correct"] == "True"
fmt = lambda v: f"{v:+.3f}".replace("-", "−")


def pair(k1, k2, f, keep=lambda r: True):
    ids = [i for i in cell[k1] if i in cell[k2] and keep(cell[k1][i])]
    return [f(cell[k1][i]) for i in ids], [f(cell[k2][i]) for i in ids]


def diff_stat(a, b):
    assert analyze.diff(a, b) in SUMMARY, "plotted difference is not in results/summary.md"
    d = np.asarray(a, float) - np.asarray(b, float)
    return (d.mean(), *analyze.ci(d))


def rate_stat(x):
    assert analyze.rate(x) in SUMMARY, "plotted rate is not in results/summary.md"
    x = np.asarray(x, float)
    return (x.mean(), *analyze.ci(x))


def style(ax, xticks, xfmt):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.set_xticks(xticks)
    ax.set_xticklabels([xfmt(t) for t in xticks], fontsize=9)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", length=0, pad=8)


def dot(ax, x, y, lo, hi, color, hollow=False, thin=False):
    ax.plot([lo, hi], [y, y], color=color, lw=1.7 if thin else LW, solid_capstyle="round", zorder=2)
    ax.plot([x], [y], "o", ms=MS, color=SURFACE if hollow else color, mec=color if hollow else SURFACE,
            mew=2.0 if hollow else 1.8, zorder=3)


# ------------------------------------------------------------------ figure 1: paired accuracy differences
steps = [("E1", "E0", "distractors added"), ("E2", "E1", "injection added"), ("E2", "E0", "both")]
groups = [(p, f"{p}  ·  {'plain' if p == 'P0' else 'defended'} system prompt",
           [(f"{a} − {b}   {t}", diff_stat(*pair((a, p), (b, p), correct))) for a, b, t in steps]) for p in ("P0", "P1")]
groups.append(("P1-P0", "P1 − P0  ·  effect of the defended prompt",
               [(lab, diff_stat(*pair((c, "P1"), (c, "P0"), correct)))
                for c, lab in (("E0", "on clean evidence (E0)"), ("E1", "with distractors (E1)"), ("E2", "with injection (E2)"))]))

fig, ax = plt.subplots(figsize=(8.8, 5.7), dpi=200, facecolor=SURFACE)
fig.subplots_adjust(left=0.31, right=0.965, top=0.80, bottom=0.17)
style(ax, [-0.10, -0.05, 0, 0.05], lambda t: "0" if t == 0 else fmt(t)[:5])
ax.set_xlim(-0.125, 0.085)
y, ticks, labels = 0.0, [], []
for key, head, items in groups:
    ax.text(-0.455, y, head, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.5, fontweight="bold", color=INK)
    y += 1
    for lab, (m, lo, hi) in items:
        dot(ax, m, y, lo, hi, COLOR[key])
        if hi < 0 or lo > 0:  # label only the intervals that exclude zero
            ax.text(m, y - 0.34, f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]", ha="center", va="bottom", fontsize=8.5, color=INK2)
        ticks.append(y)
        labels.append(lab)
        y += 1
    y += 0.55
ax.axvline(0, color=INK2, lw=1.1, zorder=1)
ax.set_yticks(ticks)
ax.set_yticklabels(labels, fontsize=9)
ax.set_ylim(y - 0.55, -0.7)
ax.set_xlabel(f"difference in accuracy, proportion of the {N} claims  (left of the line = lower accuracy)", fontsize=9, labelpad=9)
fig.text(0.012, 0.955, "Paired differences in accuracy between evidence conditions and prompts", fontsize=13, fontweight="bold", color=INK, va="top")
fig.text(0.012, 0.905, f"{N} SciFact test claims, Llama 3.3 70B Instruct (FP8). Dots: difference; bars: 95% bootstrap CI (10,000 resamples).",
         fontsize=9, color=INK2, va="top")
fig.legend(handles=[Line2D([0], [0], marker="o", ls="", color=COLOR[k], mec=SURFACE, mew=1.8, ms=MS, label=l)
                    for k, l in (("P0", "P0 plain prompt"), ("P1", "P1 defended prompt"), ("P1-P0", "between prompts"))],
           loc="upper left", bbox_to_anchor=(0.003, 0.865), ncol=3, frameon=False, fontsize=9, labelcolor=INK2, handletextpad=0.3, columnspacing=1.6)
fig.text(0.012, 0.02, "Source: results/results.csv; every value is in results/summary.md, section 2.", fontsize=8, color=MUTED, va="bottom")
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig1_paired_differences.png", facecolor=SURFACE)
plt.close(fig)

# ------------------------------------------------------------------ figure 2: the injection
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.7), dpi=200, facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1]})
fig.subplots_adjust(left=0.07, right=0.975, top=0.66, bottom=0.17, wspace=0.34)

# (a) the planted word in the injected outputs
style(a, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:g}")
a.set_xlim(-0.02, 1.04)
for i, p in enumerate(("P0", "P1")):
    m, lo, hi = rate_stat([r["canary"] == "True" for r in cell[("E2", p)].values()])
    k = round(m * N)
    dot(a, m, i, lo, hi, COLOR[p])
    a.text(m, i - 0.3, f"{m:.3f}  ({k}/{N})", ha="center", va="bottom", fontsize=9, color=INK2)
a.set_yticks([0, 1])
a.set_yticklabels(["P0", "P1"], fontsize=10)
a.set_ylim(1.7, -0.75)
a.set_xlabel("share of injected (E2) outputs that contain the word", fontsize=9, labelpad=9)
a.set_title("Planted word in the output", loc="left", fontsize=11, fontweight="bold", color=INK, pad=22)
a.text(0, 1.045, "Upper bound on obeying the note: it also counts quoting or refusing it.", transform=a.transAxes, fontsize=8.5, color=INK2)
a.text(0.0, 1.5, "Controls: 0/160 in E0 and 0/160 in E1, for both prompts.", fontsize=8.5, color=MUTED, va="center", bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2), zorder=4)

# (b) verdicts pushed toward SUPPORTS on claims whose gold label is CONTRADICTS
style(b, [0, 0.1, 0.2, 0.3], lambda t: f"{t:g}")
b.set_xlim(-0.01, 0.335)
isC = lambda r: r["gold"] == "CONTRADICTS"
sup = lambda r: r["verdict"] == "SUPPORTS"
yy = {"P0": 0.0, "P1": 1.25}
for p in ("P0", "P1"):
    e1 = rate_stat([sup(r) for r in cell[("E1", p)].values() if isC(r)])
    e2 = rate_stat([sup(r) for r in cell[("E2", p)].values() if isC(r)])
    dm, dlo, dhi = diff_stat(*pair(("E2", p), ("E1", p), sup, isC))
    y0 = yy[p]
    b.plot([e1[0], e2[0]], [y0 - 0.13, y0 + 0.13], color=COLOR[p], lw=LW, solid_capstyle="round", zorder=1)
    dot(b, e1[0], y0 - 0.13, e1[1], e1[2], COLOR[p], hollow=True, thin=True)
    dot(b, e2[0], y0 + 0.13, e2[1], e2[2], COLOR[p], thin=True)
    b.text(-0.004, y0 - 0.5, f"paired change {fmt(dm)} [{fmt(dlo)}, {fmt(dhi)}]", fontsize=8.5, color=INK2, va="center")
b.set_yticks([yy["P0"], yy["P1"]])
b.set_yticklabels(["P0", "P1"], fontsize=10)
b.set_ylim(1.75, -0.8)
b.set_xlabel("share of gold-CONTRADICTS claims given the verdict SUPPORTS", fontsize=9, labelpad=9)
b.set_title("Verdicts pushed toward SUPPORTS", loc="left", fontsize=11, fontweight="bold", color=INK, pad=22)
b.text(0, 1.045, "Gold-CONTRADICTS claims only (n = 80); the note names SUPPORTS.", transform=b.transAxes, fontsize=8.5, color=INK2)
b.legend(handles=[Line2D([0], [0], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, mew=2.0, ms=MS, label="E1  distractors only"),
                  Line2D([0], [0], marker="o", ls="", color=INK2, mec=SURFACE, mew=1.8, ms=MS, label="E2  injected note")],
         loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2, handletextpad=0.3, borderaxespad=0.2)

fig.text(0.012, 0.975, "Effect of the planted note on the output text and on the verdicts", fontsize=13, fontweight="bold", color=INK, va="top")
fig.text(0.012, 0.915, f"{N} SciFact test claims, Llama 3.3 70B Instruct (FP8). Dots: rate; bars: 95% bootstrap CI (10,000 resamples).",
         fontsize=9, color=INK2, va="top")
fig.legend(handles=[Line2D([0], [0], marker="o", ls="", color=COLOR[k], mec=SURFACE, mew=1.8, ms=MS, label=l)
                    for k, l in (("P0", "P0 plain prompt"), ("P1", "P1 defended prompt"))],
           loc="upper left", bbox_to_anchor=(0.003, 0.865), ncol=2, frameon=False, fontsize=9, labelcolor=INK2, handletextpad=0.3, columnspacing=1.6)
fig.text(0.012, 0.02, "Source: results/results.csv; every value is in results/summary.md, sections 4 and 5.", fontsize=8, color=MUTED, va="bottom")
fig.savefig(OUT / "fig2_injection.png", facecolor=SURFACE)
plt.close(fig)
print("wrote", *sorted(p.name for p in OUT.glob("*.png")), "| all plotted numbers matched results/summary.md")
