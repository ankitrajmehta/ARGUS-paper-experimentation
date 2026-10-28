"""Generate the report figures from the recorded results.

Colours come from the validated categorical palette (slots 1-3). Every chart is
printed, so nothing relies on hover: series are direct-labelled and the ink
colours stay separate from the series colours.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b6b0"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {name}")


def box(ax, x, y, w, h, text, fc, tc="white", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=0, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, weight="bold", linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.4, color=color))


# ---------------------------------------------------------------- fig 1
def fig_threat_model():
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.6); ax.axis("off")

    box(ax, 0.1, 1.5, 1.7, 0.8, "Camera\n(road scene)", BLUE)
    box(ax, 2.2, 1.5, 1.9, 0.8, "Vision-language\nmodel", BLUE)
    box(ax, 4.5, 1.5, 1.7, 0.8, "Text decision\n\"brake\"", BLUE)
    box(ax, 6.6, 1.5, 1.6, 0.8, "Controller", BLUE)
    box(ax, 8.4, 1.5, 1.4, 0.8, "Vehicle\nmoves", BLUE)
    for x in (1.8, 4.1, 6.2, 8.2):
        arrow(ax, x, 1.9, x + 0.4, 1.9)

    box(ax, 0.1, 0.2, 1.7, 0.75, "Attacker puts\na sign here", ORANGE)
    arrow(ax, 0.95, 0.95, 0.95, 1.5, ORANGE)
    ax.text(5.2, 0.55, "The model reads the sign as a command.\n"
                       "The decision changes to \"proceed\".",
            ha="center", va="center", color=INK, fontsize=9)
    ax.set_title("Figure 1  The threat: a sign in the camera view changes the decision",
                 color=INK, fontsize=10, weight="bold", loc="left", pad=8)
    save(fig, "fig1_threat_model.png")


# ---------------------------------------------------------------- fig 2
def fig_chai_pipeline():
    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")

    ax.text(0.1, 2.95, "Stage 1  Make a word list", color=INK, fontsize=9.5, weight="bold")
    box(ax, 0.1, 1.95, 2.0, 0.75, "Attacker LLM\nwrites a phrase", AQUA, tc="#06281d")
    box(ax, 2.5, 1.95, 2.0, 0.75, "Put phrase on\na test image", AQUA, tc="#06281d")
    box(ax, 4.9, 1.95, 2.0, 0.75, "Ask the target\nmodel", AQUA, tc="#06281d")
    box(ax, 7.3, 1.95, 2.4, 0.75, "Keep the score.\nWrite a new phrase.", AQUA, tc="#06281d")
    for x in (2.1, 4.5, 6.9):
        arrow(ax, x, 2.32, x + 0.4, 2.32)
    # The loop arrow is routed under the row; a straight return cuts through
    # the boxes.
    ax.annotate("", xy=(1.1, 1.92), xytext=(8.5, 1.92),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.2,
                                connectionstyle="arc3,rad=-0.05"))
    ax.text(4.9, 1.38, "repeat  ->  a list of 9-10 phrases", ha="center",
            color=INK2, fontsize=8, style="italic")

    ax.text(0.1, 1.08, "Stage 2  Choose the best sign", color=INK, fontsize=9.5, weight="bold")
    box(ax, 0.1, 0.2, 2.3, 0.75, "Pick a phrase\nfrom the list", ORANGE)
    box(ax, 2.8, 0.2, 2.3, 0.75, "Pick colours,\nsize, position", ORANGE)
    box(ax, 5.5, 0.2, 2.1, 0.75, "Test on many\nimages", ORANGE)
    box(ax, 8.0, 0.2, 1.7, 0.75, "Keep the\nbest sign", ORANGE)
    for x in (2.4, 5.1, 7.6):
        arrow(ax, x, 0.57, x + 0.4, 0.57)
    ax.set_title("Figure 2  The CHAI attack has two stages",
                 color=INK, fontsize=10, weight="bold", loc="left", pad=8)
    save(fig, "fig2_chai_pipeline.png")


# ---------------------------------------------------------------- fig 3
def fig_our_pipeline():
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.1); ax.axis("off")

    steps = [
        ("1  Get CHAI\nfiles", BLUE), ("2  Put the sign\non the image", BLUE),
        ("3  Ask the\nmodel", BLUE), ("4  Count the\nattacks", BLUE),
    ]
    for i, (text, colour) in enumerate(steps):
        box(ax, 0.1 + i * 2.45, 2.0, 2.1, 0.8, text, colour)
        if i < 3:
            arrow(ax, 2.2 + i * 2.45, 2.4, 2.55 + i * 2.45, 2.4)
    ax.text(9.85, 2.4, "attack\nworks", ha="right", va="center",
            color=INK2, fontsize=8, style="italic")

    steps2 = [
        ("5  Record the\nmodel state", AQUA), ("6  Train a\nlinear probe", AQUA),
        ("7  Build the\ngate", AQUA), ("8  Move the\nstate", AQUA),
    ]
    for i, (text, colour) in enumerate(steps2):
        box(ax, 0.1 + i * 2.45, 0.5, 2.1, 0.8, text, colour, tc="#06281d")
        if i < 3:
            arrow(ax, 2.2 + i * 2.45, 0.9, 2.55 + i * 2.45, 0.9)
    ax.text(9.85, 0.9, "no effect", ha="right", va="center",
            color=ORANGE, fontsize=8, weight="bold")

    ax.text(0.1, 1.55, "Attack  (steps 1-4)", color=INK2, fontsize=8.5, weight="bold")
    ax.text(0.1, 0.05, "Defense  (steps 5-8)", color=INK2, fontsize=8.5, weight="bold")
    ax.set_title("Figure 3  Our eight steps",
                 color=INK, fontsize=10, weight="bold", loc="left", pad=8)
    save(fig, "fig3_our_pipeline.png")


# ---------------------------------------------------------------- fig 4
def fig_asr():
    runs = {}
    for path in (ROOT / "results" / "runs_2b_tiles6").glob("*.json"):
        d = json.loads(path.read_text())
        runs[(d["split"], d["label"])] = d["summary"]["asr"] * 100

    groups = ["Train split\n(15 scenes)", "Test split\n(8 scenes)"]
    clean = [runs[("train", "baseline")], runs[("test", "baseline")]]
    attacked = [runs[("train", "attack")], runs[("test", "attack")]]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = range(len(groups))
    w = 0.34
    b1 = ax.bar([i - w / 2 for i in x], clean, w, color=BLUE, label="No sign")
    b2 = ax.bar([i + w / 2 for i in x], attacked, w, color=ORANGE, label="With CHAI sign")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{bar.get_height():.0f}%", ha="center", color=INK, fontsize=9, weight="bold")

    ax.set_xticks(list(x)); ax.set_xticklabels(groups)
    ax.set_ylabel("Attack success rate")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=MUTED, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title("Figure 4  The sign changes the decision on the train split",
                 color=INK, fontsize=10, weight="bold", loc="left", pad=8)
    save(fig, "fig4_asr.png")


# ---------------------------------------------------------------- fig 5
def fig_probe():
    data = json.loads((ROOT / "results" / "probes_train.json").read_text())
    fig, ax = plt.subplots(figsize=(8.0, 3.4))

    series = [("scene", BLUE, "New road scene"), ("scene+phrase", ORANGE, "New scene and new words")]
    # Annotation offsets are tuned per series so the callouts clear both lines.
    offsets = {"scene": (0.6, 5.5), "scene+phrase": (0.6, -9)}
    for key, colour, label in series:
        rows = [r for r in data.get(key, []) if r["accuracy"] == r["accuracy"]]
        xs = [r["layer"] for r in rows]
        ys = [r["accuracy"] * 100 for r in rows]
        ax.plot(xs, ys, color=colour, linewidth=2, marker="o", markersize=4, label=label)
        best = max(rows, key=lambda r: r["accuracy"])
        dx, dy = offsets[key]
        ax.annotate(f"{best['accuracy']:.0%} at layer {best['layer']}",
                    xy=(best["layer"], best["accuracy"] * 100),
                    xytext=(best["layer"] + dx, best["accuracy"] * 100 + dy),
                    color=INK, fontsize=8.5, weight="bold")

    # Reference lines are labelled at the right margin, clear of the curves.
    ax.axhline(60, color=MUTED, linewidth=1.3, linestyle="--")
    ax.text(24.6, 60, "  60%: the answer\n  word alone", color=INK2, fontsize=7.5,
            va="center", ha="left")
    ax.axhline(50, color=MUTED, linewidth=1, linestyle=":")
    ax.text(24.6, 50, "  50%: a coin toss", color=INK2, fontsize=7.5,
            va="center", ha="left")

    ax.set_xlabel("Layer in the language model")
    ax.set_ylabel("Probe accuracy")
    ax.set_ylim(35, 112)
    ax.set_xlim(-1, 32)
    ax.set_xticks([0, 5, 10, 15, 20, 24])
    ax.set_yticks([40, 60, 80, 100]); ax.set_yticklabels(["40%", "60%", "80%", "100%"])
    ax.grid(axis="y", color=MUTED, linewidth=0.5, alpha=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_title("Figure 5  The middle layers show which instruction the model follows",
                 color=INK, fontsize=10, weight="bold", loc="left", pad=8)
    save(fig, "fig5_probe.png")


# ---------------------------------------------------------------- fig 6
def fig_sweep():
    one = json.loads((ROOT / "results" / "defense" / "sweep_train_2b.json").read_text())
    many = json.loads((ROOT / "results" / "defense" / "sweep_train_2b_multi.json").read_text())

    labels, values, colours = ["No defense"], [one[0]["attack_rate"] * 100], [MUTED]
    for r in one[1:]:
        labels.append(f"1 layer\ns={r['strength']:g}")
        values.append(r["attack_rate"] * 100); colours.append(BLUE)
    for r in many[1:]:
        labels.append(f"5 layers\ns={r['strength']:g}")
        values.append(r["attack_rate"] * 100); colours.append(AQUA)

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    # Band first, bars on top: a translucent span over the bars reads as if the
    # bars were segmented.
    base = values[0]
    ax.axhspan(base - 25, base + 25, color=MUTED, alpha=0.22, zorder=0)
    ax.axhline(base, color=INK2, linewidth=1.2, linestyle="--", zorder=1)
    bars = ax.bar(range(len(values)), values, 0.62, color=colours, zorder=2)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{bar.get_height():.0f}%", ha="center", color=INK, fontsize=8.5, weight="bold")

    ax.text(len(values) - 0.4, base + 26, "range of chance (95% limits, 15 scenes)",
            ha="right", color=INK2, fontsize=8)

    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Attack success rate")
    ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=MUTED, linewidth=0.5, alpha=0.5); ax.set_axisbelow(True)
    ax.set_title("Figure 6  The defense does not change the result",
                 color=INK, fontsize=10, weight="bold", loc="left", pad=8)
    save(fig, "fig6_sweep.png")


if __name__ == "__main__":
    print("writing figures:")
    fig_threat_model()
    fig_chai_pipeline()
    fig_our_pipeline()
    fig_asr()
    fig_probe()
    fig_sweep()
    print(f"\nall figures in {OUT}")
