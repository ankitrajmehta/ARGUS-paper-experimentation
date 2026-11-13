"""Is the steering probe reading "about to comply", or just surface features?

A probe that scores 100% at layer 5 of 36 has almost certainly not found an
instruction-following representation -- the network has barely started. Two
surface features in the *input* can together determine the label with no such
representation at all:

1. **The forced answer token.** It is appended to the prompt, so it is present
   from the embedding layer on.
2. **The sign's wording.** ``midas-activations`` draws phrases from
   ``dictionary[target]``, so the attacker's phrase is chosen *by target*. With
   one phrase per target the rendered text identifies the target outright.

``follows_injection`` is exactly ``answer == target``, so a probe that reads
both features computes the label by matching, without encoding compliance.

The control is clean frames. They carry no sign, but the run still records
``is_target_answer``, the same ``answer == target`` quantity. So:

* if a probe predicts ``is_target_answer`` on **clean** frames about as well as
  it predicts ``follows_injection`` on attacked ones, then the sign contributes
  nothing and the "steering direction" is an answer-identity direction;
* if clean accuracy is near the 71% that guessing the majority target gives,
  while attacked accuracy is far higher, the sign is carrying real signal.

    python scripts/probe_leak_check.py <answer-sweep.npz>
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from midas import probes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activations", type=Path)
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 1, 2, 5, 10, 18, 27, 35])
    args = parser.parse_args(argv)

    acts = probes.load_activations(args.activations)
    cond = acts.column("condition")
    print(f"{len(acts.rows)} rows: {dict(Counter(cond))}")

    attacked = acts.subset(cond == "attack")
    clean = acts.subset(cond == "clean")

    # How often does the majority target alone get it right? That is the floor
    # a probe has to beat before it is reading anything at all.
    tgt = attacked.column("is_target_answer").astype(int)
    print(f"\nattacked rows: {tgt.mean():.1%} are the target answer "
          f"(balanced by construction -- one row per answer)")
    phrases = np.unique(attacked.column("phrase"))
    print(f"attack wordings: {len(phrases)} -> {list(phrases)}")

    # The leak this checks: is the wording chosen by target?
    by_phrase = {}
    for p in phrases:
        sel = attacked.column("phrase") == p
        targets = {r["answer"] for r, s in zip(attacked.rows, sel) if s}
        by_phrase[p] = targets
    print("\nwording -> answers present:")
    for p, t in by_phrase.items():
        print(f"  {p!r:28s} {sorted(t)}")

    print("\n" + "=" * 72)
    print(f"{'layer':>6s} {'attacked: follows_injection':>28s} "
          f"{'clean: is_target_answer':>26s}")
    print("=" * 72)
    for layer in args.layers:
        a_acc, _ = probes.probe_layer(
            attacked.vectors[:, layer, :],
            attacked.column("follows_injection").astype(int),
            attacked.column("scene"),
        )
        c_acc, _ = probes.probe_layer(
            clean.vectors[:, layer, :],
            clean.column("is_target_answer").astype(int),
            clean.column("scene"),
        )
        print(f"{layer:6d} {a_acc:28.1%} {c_acc:26.1%}")

    print("""
Read it this way. The clean column has no sign in the image at all, so any
accuracy there is the model's internal state distinguishing "I was handed the
action this scene calls for" from "I was handed the other one" -- a quantity
that exists with or without an attack. Whatever the clean column reaches, the
attacked column must *exceed* before the probe can be said to have found
anything about injections.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
