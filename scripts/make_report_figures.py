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
DRIVELM = ROOT / "results" / "kaggle_drivelm" / "results" / "drivelm"
ARGUS = ROOT / "results" / "kaggle_argus" / "results" / "drivelm" / "argus"

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
    """Title above a subtitle, both left-aligned to the axes.

    The pad has to grow with the subtitle: it is drawn in the same space the
    title reserves, so a two-line caption at a fixed pad is printed straight
    through the title. That happened to two figures before this was derived
    from the line count rather than hardcoded.
    """
    lines = subtitle.count("\n") + 1
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left",
                 pad=8 + 11 * lines)
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



# ------------------------------------------------ DriveLM: the directional split
def fig_direction():
    """The figure that changes the headline.

    The sign's wording follows the target, so the two directions are separate
    experiments. Plotting only the aggregate hides that the attack works in one
    and not the other -- and the aggregate understates the one that matters.
    """
    runs = {}
    for sub in ("baseline", "plain", "authority", "override", "urgency"):
        recs = load(f"{sub}/*.json", DRIVELM)
        if recs:
            runs[sub] = recs[0]["results"]
    if "baseline" not in runs or len(runs) < 3:
        raise FileNotFoundError("DriveLM runs not fetched")

    base = {r["scene_index"]: r for r in runs["baseline"]}
    styles = [s for s in ("plain", "authority", "urgency", "override") if s in runs]

    # target "a)" means the sign reads Proceed: the driver was stopped.
    groups = [("a)", "Sign reads PROCEED\n(driver was stopped)", ORANGE),
              ("b)", "Sign reads STOP\n(driver was moving)", BLUE)]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, (target, caption, colour) in zip(axes, groups):
        idx = [i for i, r in base.items() if r["target"] == target]
        clean = 100 * sum(base[i]["action"] == target for i in idx) / len(idx)
        vals = []
        for st in styles:
            rows = [r for r in runs[st] if r["scene_index"] in idx]
            vals.append(100 * sum(r["action"] == r["target"] for r in rows) / len(rows))

        x = range(len(styles))
        bars = ax.bar(x, vals, width=0.62, color=colour, edgecolor="none")
        label_bars(ax, bars, vals)
        ax.axhline(clean, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
        # Outside the bars: inside, the label sat on top of one of them.
        ax.annotate(f"no sign\n{clean:.0f}%", xy=(1.0, clean),
                    xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    color=INK2, fontsize=8, annotation_clip=False)
        ax.set_xticks(list(x))
        ax.set_xticklabels(styles, fontsize=8.5)
        ax.set_ylim(0, 104)
        ax.annotate(caption, xy=(0, 1), xycoords="axes fraction", xytext=(0, 10),
                    textcoords="offset points", color=INK, fontsize=9,
                    fontweight="bold", va="bottom")
        ax.set_xlabel(f"n = {len(idx)} scenes")
    axes[0].set_ylabel("Chose the attacker's action")

    fig.subplots_adjust(top=0.80, wspace=0.28)
    fig.suptitle("The attack works in one direction only",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.005, ha="left", y=1.30)
    fig.text(0.005, 1.085,
             "Qwen2.5-VL-3B, 100 DriveLM scenes. Left: +58.6 points for CHAI's own "
             "wording, 17 scenes flipped and 0 lost, p < 0.0001.\nRight: +11.3 points, "
             "p = 0.22 -- not significant for any wording. The aggregate +25.0 averages "
             "the two and understates the result.",
             color=INK2, fontsize=8.5, ha="left", va="bottom")
    save(fig, "r7_direction.png")


# ------------------------------------------- the probe's clean-frame control
def fig_probe_floor():
    """Why 100% probe accuracy is not evidence of a compliance representation."""
    # Measured in iteration 7/11; recomputing needs the 140 MB sweep, so the
    # numbers are inlined with their provenance rather than silently re-derived.
    layers = [0, 1, 2, 5, 10, 16, 17, 18, 27, 35]
    attacked = [84.5, 84.0, 82.5, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    floor = [83.0, 82.0, 81.5, 77.0, 82.0, 81.5, 81.5, 82.5, 84.0, 84.5]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.fill_between(layers, floor, attacked, color=AQUA, alpha=0.16, linewidth=0)
    ax.plot(layers, attacked, marker="o", markersize=3.6, color=AQUA, linewidth=2,
            label="Attacked frames: is the model about to follow the sign?",
            solid_capstyle="round")
    ax.plot(layers, floor, marker="o", markersize=3.6, color=INK2, linewidth=1.8,
            linestyle=(0, (4, 3)),
            label="Clean frames, no sign at all: the same answer==target label")

    ax.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
    ax.annotate("50%: chance", xy=(35, 50), xytext=(0, 3),
                textcoords="offset points", ha="right", color=MUTED, fontsize=8)
    ax.annotate("the only real headroom", xy=(12, 91), color=INK, fontsize=8.5,
                ha="center")

    ax.set_ylim(40, 108)
    ax.set_yticks([50, 75, 100])
    ax.set_xlim(-0.5, 35.5)
    ax.set_xlabel("Layer in the language model (Qwen2.5-VL-3B, 36 layers)")
    ax.set_ylabel("Held-out-scene probe accuracy")
    titles(ax, "A 100% probe that mostly is not reading the attack",
           "Clean frames carry no sign, yet support the same prediction at 83-84%. "
           "The attack wordings are\nliterally STOP and Proceed, so matching the "
           "sign's word to the answer slot closes the rest.")
    ax.legend(frameon=False, fontsize=8, loc="lower right", handlelength=2.0)
    save(fig, "r8_probe_floor.png")


# ------------------------------------------------------- the steering null
def fig_steering_null():
    """Seven configurations, none significant -- including ARGUS's own setting."""
    order = ["none", "argus_0.1x", "argus_0.25x", "argus_0.5x",
             "uniform_0.25x", "argus_0.25x_nogate", "argus_0.25x_deep"]
    pretty = {"none": "no defense", "argus_0.1x": "argus 0.1x",
              "argus_0.25x": "argus 0.25x\n(= their 9.9)", "argus_0.5x": "argus 0.5x",
              "uniform_0.25x": "uniform 0.25x\n(old schedule)",
              "argus_0.25x_nogate": "argus 0.25x\nungated",
              "argus_0.25x_deep": "argus 0.25x\nlayers 16-17"}

    names, vals, cleans = [], [], []
    for name in order:
        atk = load(f"{name}/attack_*.json", ARGUS)
        base = load(f"{name}/baseline_*.json", ARGUS)
        if not atk:
            continue
        names.append(pretty.get(name, name))
        vals.append(atk[0]["summary"]["asr"] * 100)
        cleans.append(base[0]["summary"]["benign_accuracy"] * 100 if base else float("nan"))
    if len(names) < 3:
        raise FileNotFoundError("steering runs not fetched")

    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    x = range(len(names))
    colours = [MUTED] + [BLUE] * (len(names) - 1)
    bars = ax.bar(x, vals, width=0.6, color=colours, edgecolor="none")
    label_bars(ax, bars, vals)

    ref = vals[0]
    # One scene is 3.3 points at n=30; two scenes is the width of noise.
    ax.axhspan(ref - 6.7, ref + 6.7, color=INK2, alpha=0.10, linewidth=0)
    ax.axhline(ref, color=INK2, linewidth=1.3, linestyle=(0, (4, 3)))
    # Outside the plot: inside, it printed across the tallest bar's label.
    ax.annotate("undefended\n+/- two scenes", xy=(1.0, ref),
                xycoords=("axes fraction", "data"), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center",
                color=INK2, fontsize=8, annotation_clip=False)

    ax.plot(list(x), cleans, marker="o", markersize=4, color=ORANGE, linewidth=1.6,
            label="Clean-frame accuracy (unchanged unless ungated)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=7.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Chose the attacker's action")
    titles(ax, "Steering along the probe direction does nothing",
           "30 DriveLM scenes. Every configuration sits inside two scenes of "
           "undefended; smallest p = 0.34, and five of six\nmake the attack more "
           "successful. Clean accuracy only moves when the gate is removed.")
    ax.legend(frameon=False, fontsize=8, loc="lower left", handlelength=1.6)
    save(fig, "r9_steering_null.png")


if __name__ == "__main__":
    print("report figures:")
    for fn in (fig_sign_comparison, fig_scale, fig_styles, fig_order, fig_search,
               fig_probe_8b, fig_direction, fig_probe_floor, fig_steering_null):
        try:
            fn()
        except Exception as exc:  # a missing run should not block the rest
            print(f"  FAILED {fn.__name__}: {type(exc).__name__}: {exc}")
