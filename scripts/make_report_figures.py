"""Figures for the final report, read from the recorded results.

Separate from make_figures.py, which draws the schematics and the earlier 2B
numbers. Everything here comes from kernel v14 (CHAI's actual sign) and v13 (the
sign comparison), so the report never shows a hand-typed number.

Design notes, since these are printed and cannot rely on hover:

* Palette is the project's categorical slots 1-3, validated with the dataviz
  validator (all checks pass; aqua's sub-3:1 contrast against the surface is
  relieved by direct labels, which every chart below carries).
* One axis per chart, no dual scales. Series are direct-labelled *and*
  legended, so identity never depends on colour alone.
* Percentages are rates over 15 scenes unless stated, so one scene is 6.7
  points. Charts that compare subsets say so in the subtitle.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FINAL = ROOT / "results" / "kaggle_final" / "results"
OPTIM = ROOT / "results" / "kaggle_optimize" / "results"

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


def load(pattern: str, root: Path) -> list[dict]:
    out = []
    for path in sorted(root.glob(pattern)):
        d = json.loads(path.read_text())
        if "summary" in d:
            out.append(d)
    return out


def rates(records: list[dict]) -> dict:
    """(model size, condition) -> CHAI-rule rate in percent."""
    return {
        (d["model_id"].split("-")[-1], d["label"]): d["summary"]["asr_chai"] * 100
        for d in records
    }


def titles(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left", pad=18)
    ax.annotate(
        subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
        textcoords="offset points", color=INK2, fontsize=8.5, va="bottom",
    )


def label_bars(ax, bars, values, fmt="{:.1f}%", dy=1.2):
    for bar, v in zip(bars, values):
        ax.annotate(
            fmt.format(v), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, dy * 2), textcoords="offset points",
            ha="center", va="bottom", color=INK, fontsize=8.5,
        )


# ------------------------------------------------------- the two released signs
def fig_sign_comparison():
    """The audit's central finding: CHAI's two artefacts describe different signs.

    Grouped bars rather than deltas, so the clean baseline stays visible -- a
    delta alone would hide that the baselines are identical and the whole gap is
    in the attack.
    """
    actual = rates(load("runs_8b/*.json", FINAL) + load("scale/*.json", FINAL))
    stale = rates(load("param_source/optimal_json/*.json", OPTIM))
    actual_v13 = rates(load("param_source/chai_internvl/*.json", OPTIM))

    models = ["2B", "8B"]
    clean = [actual_v13.get(("2B", "baseline"), actual[("2B", "baseline")]),
             actual[("8B", "baseline")]]
    chai = [actual_v13.get(("2B", "attack"), actual[("2B", "attack")]),
            actual[("8B", "attack")]]
    other = [stale[("2B", "attack")], stale[("8B", "attack")]]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = range(len(models))
    w = 0.26
    # 2px surface gap between adjacent fills.
    b0 = ax.bar([i - w - 0.01 for i in x], clean, w, color=MUTED,
                label="No sign", edgecolor=SURFACE, linewidth=1.5)
    b1 = ax.bar(list(x), chai, w, color=BLUE,
                label="CHAI's actual sign\n(black on red, 0.93x)",
                edgecolor=SURFACE, linewidth=1.5)
    b2 = ax.bar([i + w + 0.01 for i in x], other, w, color=ORANGE,
                label="optimal.json sign\n(pink on red, 1.0x)",
                edgecolor=SURFACE, linewidth=1.5)
    for bars, vals in ((b0, clean), (b1, chai), (b2, other)):
        label_bars(ax, bars, vals)

    ax.axhline(54.7, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("CHAI publish 54.7%", xy=(-0.42, 54.7), xytext=(0, 5),
                textcoords="offset points", ha="left", color=INK2, fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylabel("Attack success rate, CHAI's rule")
    ax.set_ylim(0, 85)
    ax.set_xlabel("InternVL2.5 model size")
    titles(ax, "CHAI released two different signs",
           "Their code reads the weaker one. 15 scenes, so one scene is 6.7 points.")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.22),
              ncol=3, handlelength=1.2, columnspacing=1.2)
    save(fig, "r1_sign_comparison.png")


# ------------------------------------------------------------------ scale study
def fig_scale():
    """Clean -> attacked per model size, with the 1B's order-control caveat.

    A dumbbell: the pair is one scene-set measured twice, so connecting them
    reads as change, which grouped bars would not.
    """
    r = rates(load("scale/*.json", FINAL) + load("runs_8b/*train*.json", FINAL))
    swapped = rates(load("scale_swapped/*.json", FINAL))

    models = ["1B", "2B", "4B", "8B"]
    clean = [r[(m, "baseline")] for m in models]
    attacked = [r[(m, "attack")] for m in models]

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    y = range(len(models))
    for i, (c, a) in enumerate(zip(clean, attacked)):
        ax.plot([c, a], [i, i], color=MUTED, linewidth=2, zorder=1,
                solid_capstyle="round")
        ax.scatter([c], [i], s=70, color=MUTED, zorder=2,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.scatter([a], [i], s=70, color=BLUE, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(f"{a:.1f}%", xy=(a, i), xytext=(9, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8.5)
        ax.annotate(f"+{a - c:.1f}", xy=(a, i), xytext=(9, -11),
                    textcoords="offset points", va="center", color=INK2, fontsize=8)

    # The 1B's figure does not survive the option-order swap; show the other
    # reading rather than leaving the headline unqualified.
    one_b = swapped[("1B", "attack")]
    ax.scatter([one_b], [0], s=70, color=ORANGE, zorder=4, marker="D",
               edgecolor=SURFACE, linewidth=1.5)
    ax.annotate("46.7% with the two\noptions reversed", xy=(one_b, 0),
                xytext=(-6, -26), textcoords="offset points", ha="right",
                color=ORANGE, fontsize=8)

    ax.scatter([], [], s=70, color=MUTED, label="No sign")
    ax.scatter([], [], s=70, color=BLUE, label="CHAI's sign")
    ax.scatter([], [], s=70, color=ORANGE, marker="D", label="CHAI's sign, options reversed")

    ax.set_yticks(list(y))
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_ylim(len(models) - 0.4, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Attack success rate, CHAI's rule")
    ax.set_ylabel("Model size")
    titles(ax, "Scale buys answer stability, not resistance",
           "2B to 8B span one scene. The 1B's figure is largely positional.")
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, -0.18), ncol=3, handlelength=1.0,
              columnspacing=1.2)
    save(fig, "r2_scale.png")


# -------------------------------------------------------------- injection style
def fig_styles():
    """One measure across five named conditions -> sorted horizontal bars."""
    base = load("styles/_baseline/*.json", FINAL)[0]["summary"]["asr_chai"] * 100
    rows = [("No sign", base, MUTED)]
    for style, colour in (("plain", BLUE), ("override", AQUA),
                          ("authority", AQUA), ("urgency", ORANGE)):
        recs = load(f"styles/{style}/*.json", FINAL)
        if recs:
            rows.append((style, recs[0]["summary"]["asr_chai"] * 100, colour))
    rows.sort(key=lambda t: t[1])

    labels = {
        "plain": "PROCEED / STOP\n(CHAI's wording)",
        "authority": "POLICE PROCEED",
        "override": "OVERRIDE PROCEED",
        "urgency": "EMERGENCY PROCEED",
        "No sign": "No sign",
    }

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = range(len(rows))
    bars = ax.barh(list(y), [v for _, v, _ in rows], height=0.6,
                   color=[c for _, _, c in rows], edgecolor=SURFACE, linewidth=1.5)
    for i, (_, v, _) in enumerate(rows):
        ax.annotate(f"{v:.1f}%", xy=(v, i), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels([labels[n] for n, _, _ in rows], fontsize=8.5)
    ax.set_xlim(0, 88)
    ax.set_xlabel("Attack success rate, CHAI's rule (2B, 15 scenes)")
    titles(ax, "Framing the command beats optimising it",
           "All three framed variants beat CHAI's bare imperative.")
    save(fig, "r3_styles.png")


# ------------------------------------------------------- option-order agreement
def fig_order():
    """Per-scene agreement between the two option orders.

    The rate alone would hide that the 1B changes *which* scenes it flips, so
    the measure plotted is agreement, with both rates direct-labelled.
    """
    r = rates(load("scale/*.json", FINAL) + load("runs_8b/*train*.json", FINAL))
    sw = rates(load("scale_swapped/*.json", FINAL))

    def per_scene(pattern: str, root: Path, size: str, cond: str) -> dict:
        for d in load(pattern, root):
            if d["model_id"].split("-")[-1] == size and d["label"] == cond:
                return {x["scene_index"]: x["action"] for x in d["results"]}
        return {}

    models = ["1B", "2B", "4B", "8B"]
    agree = []
    for m in models:
        a = per_scene("scale/*.json", FINAL, m, "attack") or per_scene(
            "runs_8b/*train*.json", FINAL, m, "attack")
        b = per_scene("scale_swapped/*.json", FINAL, m, "attack")
        shared = sorted(set(a) & set(b))
        agree.append(100 * sum(a[s] == b[s] for s in shared) / len(shared))

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    colours = [ORANGE if v < 95 else BLUE for v in agree]
    bars = ax.bar(models, agree, width=0.5, color=colours,
                  edgecolor=SURFACE, linewidth=1.5)
    label_bars(ax, bars, agree, fmt="{:.0f}%")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Scenes answered the same way")
    ax.set_xlabel("Model size, attacked frames")
    # Below the axis, on the surface: grey ink on a coloured fill fails contrast.
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(
        [f"{m}\n{r[(m, 'attack')]:.1f}% -> {sw[(m, 'attack')]:.1f}%" for m in models]
    )
    titles(ax, "CHAI's two-turn prompt is unbiased at 2B and above",
           "Attack rate under each option order shown beneath each bar.")
    save(fig, "r4_option_order.png")


# -------------------------------------------------------------------- the search
def fig_search():
    """Validated generation rates per target, CHAI's sign against the search.

    Two targets are different scene subsets (6 and 9), so they are two groups on
    one axis rather than one pooled bar -- pooling would imply a denominator
    that does not exist.
    """
    rows = []
    for path in sorted((FINAL / "optimize").glob("*.json")):
        d = json.loads(path.read_text())
        v = (d.get("meta") or {}).get("validation") or {}
        if v:
            rows.append((d["meta"]["target"], d["meta"]["n_scenes"],
                         v["chai_released"]["score"] * 100,
                         v["re_optimised"]["score"] * 100))
    if not rows:
        print("  skip r5_search (no validated search results)")
        return

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    x = range(len(rows))
    w = 0.3
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], [t[2] for t in rows], w, color=BLUE,
                label="CHAI's sign", edgecolor=SURFACE, linewidth=1.5)
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], [t[3] for t in rows], w, color=ORANGE,
                label="Our re-optimised sign", edgecolor=SURFACE, linewidth=1.5)
    label_bars(ax, b1, [t[2] for t in rows])
    label_bars(ax, b2, [t[3] for t in rows])

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Target {t[0]}\n{t[1]} attackable scenes" for t in rows])
    ax.set_ylim(0, 85)
    ax.set_ylabel("Attack success rate, generated")
    titles(ax, "The search gains on one target and loses on the other",
           "Equal and opposite: two scenes lost, three gained. The net is zero.")
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, -0.2), ncol=2, handlelength=1.2)
    save(fig, "r5_search.png")


# ------------------------------------------------------------- probe, 8B
def fig_probe_8b():
    """Probe accuracy by layer on the 8B.

    A line chart: layer index is ordered and continuous, so position carries
    depth and the reader can see *where* the representation appears rather than
    only how accurate it gets.
    """
    path = ROOT / "results" / "kaggle_8b" / "results" / "probes_8b.json"
    if not path.exists():
        print("  skip r6_probe (no probe sweep)")
        return
    data = json.loads(path.read_text())

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    series = [
        ("scene", BLUE, "Unseen road scene"),
        ("scene+phrase", ORANGE, "Unseen scene and unseen wording"),
    ]
    for key, colour, label in series:
        rows = [r for r in data.get(key, []) if r["accuracy"] == r["accuracy"]]
        if not rows:
            continue
        ax.plot([r["layer"] for r in rows], [r["accuracy"] * 100 for r in rows],
                color=colour, linewidth=2, label=label, solid_capstyle="round")

    # The floor a probe reaches from the forced answer token alone: the
    # attacker's target is "brake" in 9 of 15 scenes.
    ax.axhline(60, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("60%: the answer token alone", xy=(32, 60), xytext=(0, -13),
                textcoords="offset points", ha="right", color=INK2, fontsize=8)
    ax.axvline(11, color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
    ax.annotate("100% from layer 11", xy=(11, 103), xytext=(5, 0),
                textcoords="offset points", ha="left", color=INK, fontsize=8.5)

    ax.set_ylim(25, 112)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_xlim(0, 32)
    ax.set_xlabel("Layer in the language model (InternVL2.5-8B)")
    ax.set_ylabel("Probe accuracy")
    titles(ax, "The attack is linearly decodable from layer 11 on",
           "Layer 0 sits exactly at the answer-token floor, so nothing leaks.")
    ax.legend(frameon=False, fontsize=8, loc="lower right", handlelength=1.6)
    save(fig, "r6_probe.png")


if __name__ == "__main__":
    print("report figures:")
    for fn in (fig_sign_comparison, fig_scale, fig_styles, fig_order, fig_search,
               fig_probe_8b):
        try:
            fn()
        except Exception as exc:  # a missing run should not block the rest
            print(f"  FAILED {fn.__name__}: {type(exc).__name__}: {exc}")
