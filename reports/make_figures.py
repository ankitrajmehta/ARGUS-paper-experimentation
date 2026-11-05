"""Figures for the ARGUS-for-embodied-AI progress report (reads the downloaded Kaggle results)."""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

K = Path("E:/Argus/outputs/kaggle")
OUT = Path(__file__).with_name("figs")
OUT.mkdir(exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#cfcdc6"
plt.rcParams.update({
    "font.family": "Calibri", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# Fig 1 - what each sign does to the robot (94 test frames each, effect relative to the same clean frame)
rows = [
    ("Harmless sign (E2)", 1, 1, 25, 67),
    ("Attack text strip (E2)", 38, 36, 5, 15),
    ("Attack sign, ARGUS-style (E2)", 24, 34, 14, 22),
    ("Attack sign, no trigger (E2b)", 25, 23, 21, 25),
    ("Attack sign, plain wording (E2b)", 11, 22, 18, 43),
    ("Attack sign, unseen styles (E2c)", 19, 23, 13, 39),
]
segments = [("Hijacked (did attacker's action)", S1), ("Stopped (abandoned task)", S2), ("Other change", S3), ("No effect", NEUTRAL)]
fig, ax = plt.subplots(figsize=(7.2, 3.3))
for y, (label, *counts) in enumerate(reversed(rows)):
    left = 0
    for count, (_, color) in zip(counts, segments):
        ax.barh(y, count, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=1.5)
        if count >= 6:
            ax.text(left + count / 2, y, str(count), ha="center", va="center", fontsize=8.5, color="white" if color != NEUTRAL else INK)
        left += count
ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
ax.set_xlim(0, 94)
ax.set_xlabel("Test frames (out of 94)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c in segments], labels=[s for s, _ in segments],
          ncol=4, loc="upper center", bbox_to_anchor=(0.4, 1.16), fontsize=8.5, handlelength=1)
grid(ax)
save(fig, "fig_outcomes.png")

# Fig 2 - detection summary: E2 probe vs E2c probe at their pre-chosen layers
rep = json.load(open(K / "e2c_v1/outputs/e2c_3b_fp16/results/e2c_report.json", encoding="utf-8"))
old, new = rep["probes"]["e2_probe"]["at_chosen_layer"], rep["probes"]["varied"]["at_chosen_layer"]
attacks = [("ARGUS-style sign", "injected_sign"), ("Text strip", "injected_strip"), ("No trigger (E2b)", "attack_no_trigger"),
           ("Plain wording (E2b)", "attack_plain"), ("Unseen styles (E2c)", "e2c_attack")]
harmless = [("Clean frame", "clean"), ("Harmless sign", "benign_sign"), ("Legit sign (E2b)", "legit_imperative"), ("Unseen legit sign (E2c)", "e2c_legit")]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [5, 4]})
for ax, items, title in [(axes[0], attacks, "Attacks caught (higher is better)"), (axes[1], harmless, "False alarms (lower is better)")]:
    for k, (label, key) in enumerate(reversed(items)):
        for off, (rates, color) in zip([0.18, -0.18], [(old, S1), (new, S2)]):
            value = rates[key] * 100
            ax.barh(k + off, value, height=0.34, color=color, edgecolor=SURFACE, linewidth=1)
            ax.text(value + 1.5, k + off, f"{value:.0f}%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(items)), [l for l, _ in reversed(items)])
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title(title, loc="left")
    grid(ax)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=S1), plt.Rectangle((0, 0), 1, 1, color=S2)],
           labels=[f"E2 probe (layer {rep['probes']['e2_probe']['chosen_layer']})", f"E2c varied probe (layer {rep['probes']['varied']['chosen_layer']})"],
           loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "fig_detection_summary.png", dpi=220)
plt.close(fig)


# Fig 3/4 - per-layer curves (small multiples, same series in each panel)
def layer_panels(name, panels, series, ylabel):
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4, 2.9), sharey=True)
    for ax, (title, per_layer) in zip(axes, panels):
        layers = [r["layer"] for r in per_layer]
        for label, getter, color, style in series:
            ax.plot(layers, [getter(r) * 100 for r in per_layer], style, color=color, linewidth=2, label=label)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Layer")
        ax.set_ylim(-3, 103)
        grid(ax, "y")
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8.5, loc="center right")
    save(fig, name)


e2 = json.load(open(K / "e2_v2/outputs/e2_3b_fp16/results/e2_report.json", encoding="utf-8"))
layer_panels("fig_e2_layers.png",
             [("Negatives: clean only", e2["probes"]["indomain_clean"]["per_layer"]),
              ("Negatives: clean + harmless signs", e2["probes"]["indomain_hardneg"]["per_layer"])],
             [("Attack signs caught", lambda r: r["TPR_sign"], S1, "-"),
              ("Harmless signs flagged", lambda r: r["FPR_benign"], S2, "--"),
              ("Clean frames flagged", lambda r: r["FPR_clean"], S3, ":")],
             "% of test frames")
layer_panels("fig_e2c_layers.png",
             [("E2 probe", rep["probes"]["e2_probe"]["per_layer"]), ("E2c varied probe", rep["probes"]["varied"]["per_layer"])],
             [("Unseen-style attacks caught", lambda r: r["flag_rate"]["e2c_attack"], S1, "-"),
              ("Plain-wording attacks caught", lambda r: r["flag_rate"]["attack_plain"], S2, "-"),
              ("Unseen legit signs flagged", lambda r: r["flag_rate"]["e2c_legit"], S3, "--")],
             "% of test frames")

shutil.copy(K / "e1_v3/outputs/e1_3b_fp16_n100/results/e1/e1_probe_accuracy.png", OUT / "fig_e1.png")

# Figs 5-8 - E3: ARGUS steering as released on Qwen2-VL-7B
E3 = K / "e3_v1/outputs/e3_7b/results"
e3 = json.load(open(E3 / "e3/e3_report.json", encoding="utf-8"))
modes = [("no_defense", "No defense", NEUTRAL), ("steer_only", "Steering only", S1), ("argus_full", "Full ARGUS", S2)]

# Fig 5 - test metrics with 95% CIs and the paper's values
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
panels = [("Attack measures (lower is better)", [("AIA", "Attacker's answer\n(AIA)"), ("AIFR", "Follows attacker\n(AIFR)")]),
          ("User measures (higher is better)", [("UIA_inject", "Correct, attacked\n(UIA inject)"), ("UIA_clean", "Correct, clean\n(UIA clean)")])]
for ax, (title, metrics) in zip(axes, panels):
    for x, (metric, _) in enumerate(metrics):
        for k, (mode, _, color) in enumerate(modes):
            cell = e3["eval"][mode][metric]
            pos, value = x + (k - 1) * 0.27, cell["mean"] * 100
            low, high = cell["ci95"][0] * 100, cell["ci95"][1] * 100
            ax.bar(pos, value, width=0.25, color=color, edgecolor=SURFACE, linewidth=1)
            ax.errorbar(pos, value, yerr=[[value - low], [high - value]], color=INK2, capsize=2.5, linewidth=1)
            ax.text(pos, high + 1.5, f"{value:.0f}", ha="center", fontsize=8.5, color=INK)
            ax.plot([pos - 0.11, pos + 0.11], [e3["paper_table1"][mode][metric]] * 2, color=INK, linewidth=2.2)
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
    ax.set_title(title, loc="left")
    ax.set_ylim(0, 70)
    grid(ax, "y")
axes[0].set_ylabel("% of 100 test items")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in modes] + [plt.Line2D([0], [0], color=INK, linewidth=2.2)]
fig.legend(handles=handles, labels=[label for _, label, _ in modes] + ["Paper's value"], loc="lower center", ncol=4, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.01), handlelength=1.2)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "fig_e3_test.png", dpi=220)
plt.close(fig)


# Fig 6 - what happened to each of the 100 attacked items, from the raw answers
def outcome_counts(name, sorry=False):
    counts = dict.fromkeys(["string", "other_words", "sorry", "user", "other"], 0)
    for row in json.load(open(E3 / name, encoding="utf-8"))["results"]:
        flag = lambda key: str(row[key]).lower() == "true"
        if sorry and flag("post_filter_rejected"):
            counts["sorry"] += 1
        elif flag("second_success"):
            counts["string"] += 1
        elif flag("attacker_followed"):
            counts["other_words"] += 1
        elif flag("first_success"):
            counts["user"] += 1
        else:
            counts["other"] += 1
    return counts


rows = [("No defense", outcome_counts("image_no_defense.json")), ("Steering only", outcome_counts("image_argus_steer_only.json")),
        ("Full ARGUS", outcome_counts("image_argus.json", sorry=True))]
segments = [("string", "Attacker's exact answer", S2), ("other_words", "Follows attacker, other words", "#f2a98a"), ("sorry", "Refused (\"Sorry!\")", INK2),
            ("user", "User's correct answer", S3), ("other", "Normal answer, not the expected one", NEUTRAL)]
fig, ax = plt.subplots(figsize=(7.2, 2.3))
for y, (_, counts) in enumerate(reversed(rows)):
    left = 0
    for key, _, color in segments:
        ax.barh(y, counts[key], left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=1.5)
        if counts[key] >= 4:
            ax.text(left + counts[key] / 2, y, str(counts[key]), ha="center", va="center", fontsize=8.5, color=INK if color == NEUTRAL else "white")
        left += counts[key]
ax.set_yticks(range(len(rows)), [label for label, _ in reversed(rows)])
ax.set_xlim(0, 100)
ax.set_xlabel("Attacked test items (out of 100)")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in segments], labels=[label for _, label, _ in segments],
           ncol=3, loc="upper center", fontsize=8.5, handlelength=1)
grid(ax)
fig.tight_layout(rect=(0, 0, 1, 0.8))
fig.savefig(OUT / "fig_e3_outcomes.png", dpi=220)
plt.close(fig)

# Fig 7 - preliminary search on validation items: direction per layer, strength at layer 13
pre = e3["preliminary"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
ax = axes[0]
for direction, label, color in [("attack", "Pushed toward the attacker", S2), ("defense", "Pushed toward the user", S1)]:
    points = [r for r in pre["fixed_layer"] if r["direction"] == direction]
    ax.plot([r["layer"] for r in points], [r["AIA"] * 100 for r in points], "-o", markersize=3.5, color=color, linewidth=2, label=label)
ax.axhline(pre["no_steering"]["AIA"] * 100, color=INK2, linestyle=":", linewidth=1.3, label="No steering")
ax.set_title("Attack success by layer (strength 10)", loc="left")
ax.set_xlabel("Layer steered")
ax.set_ylabel("% of 100 validation items")
ax.set_ylim(-2, 80)
ax.legend(fontsize=8.5, loc="upper right", ncol=1)
grid(ax, "y")
ax = axes[1]
alphas = [r["alpha"] for r in pre["sensitivity"]]
ax.axvspan(5, 15, color=GRID, alpha=0.6, linewidth=0)
ax.plot(alphas, [r["UIA"] * 100 for r in pre["sensitivity"]], "-o", markersize=3.5, color=S3, linewidth=2, label="User's correct answer (UIA)")
ax.plot(alphas, [r["AIA"] * 100 for r in pre["sensitivity"]], "-o", markersize=3.5, color=S2, linewidth=2, label="Attack success (AIA)")
ax.text(10, 73, "usable", ha="center", fontsize=8.5, color=INK2)
ax.set_title("Effect of strength at layer 13", loc="left")
ax.set_xlabel("Steering strength $\\alpha$")
ax.legend(fontsize=8.5, loc="upper right")
grid(ax, "y")
save(fig, "fig_e3_search.png")

# Fig 8 - why the bugs mattered little: mixture weights and push sizes
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.4), gridspec_kw={"width_ratios": [1, 1]})
ax = axes[0]
weights = [("Before training", 100 * 2.718281828 ** 6 / (2.718281828 ** 6 + 3)), ("After, layer 13", e3["vector"]["mixture_weights"]["13"][0] * 100),
           ("After, layer 12", e3["vector"]["mixture_weights"]["12"][0] * 100)]
for y, (label, value) in enumerate(reversed(weights)):
    ax.barh(y, value, height=0.6, color=S1)
    ax.text(value - 2, y, f"{value:.2f}%", ha="right", va="center", fontsize=8.5, color="white")
ax.set_yticks(range(3), [label for label, _ in reversed(weights)])
ax.set_xlim(0, 100)
ax.set_title("Behavior-probe share of the vector", loc="left")
ax.set_xlabel("%")
grid(ax)
ax = axes[1]
stats, final = e3["epsilon"]["stats"]["13"], e3["final_steering"]["13"]
pushes = [("First token (trained vector)", stats["alpha"], S1), ("Step reaching $-\\varepsilon$", stats["mean_score"] / final["w_norm"], NEUTRAL),
          ("Eq. 7 as coded", final["step_norm_at_score_0"], S2)]
for y, (label, value, color) in enumerate(reversed(pushes)):
    ax.barh(y, value, height=0.6, color=color)
    ax.text(value + 0.2, y, f"{value:.1f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(3), [label for label, _, _ in reversed(pushes)])
ax.set_xlim(0, 11.5)
ax.set_title("Size of push at layer 13, score 0", loc="left")
ax.set_xlabel("Hidden-state units")
grid(ax)
save(fig, "fig_e3_push.png")
print(sorted(p.name for p in OUT.iterdir()))
