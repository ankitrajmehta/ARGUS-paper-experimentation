"""Is the 100% probe fully explained by two surface features?

The steering probe learns ``follows_injection``, which on attacked rows is
exactly ``answer == target``. Two facts make that decomposable:

* the sign's wording is drawn from ``dictionary[target]``, so **the word on the
  sign identifies the target**;
* the forced answer is appended to the prompt, so **the word in the answer slot
  is in the input**.

``follows_injection`` is therefore the conjunction "the answer word matches the
sign word". If both components are individually decodable from the same
activation, the 100% needs no representation of instruction-following at all —
a linear probe over a space where both are present can compute the match.

This asks that directly. For each layer:

* **sign word** — can a probe tell `STOP` from `Proceed` on the sign?
* **answer word** — can it tell which action was forced into the answer slot?
* **follows_injection** — the quantity the defense is built on.

Held-out scenes throughout. If the first two are at ceiling wherever the third
is, the lexical explanation is sufficient and the direction cannot be called a
compliance direction on this evidence.

    python scripts/probe_decomposition.py <answer-sweep.npz>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from midas import probes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activations", type=Path)
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 1, 2, 3, 5, 8, 16, 24, 35])
    args = parser.parse_args(argv)

    acts = probes.load_activations(args.activations)
    attacked = acts.subset(acts.column("condition") == "attack")

    phrase = attacked.column("phrase")
    answer = attacked.column("answer")
    scenes = attacked.column("scene")
    follows = attacked.column("follows_injection").astype(int)

    sign_word = (phrase == np.unique(phrase)[0]).astype(int)
    answer_word = (answer == np.unique(answer)[0]).astype(int)

    print(f"{len(attacked.rows)} attacked rows")
    print(f"  sign wordings : {sorted(set(phrase))}")
    print(f"  answer slots  : {sorted(set(answer))}")
    # If this holds, follows_injection really is the conjunction.
    match = (sign_word == answer_word)
    agree = max((match == follows).mean(), (match != follows).mean())
    print(f"  'answer word matches sign word' reproduces follows_injection "
          f"for {agree:.1%} of rows")

    head = (f"\n{'layer':>6s} {'sign word':>11s} {'answer word':>13s} "
            f"{'follows_injection':>19s}")
    print(head)
    print("-" * len(head))
    for layer in args.layers:
        X = attacked.vectors[:, layer, :]
        a, _ = probes.probe_layer(X, sign_word, scenes)
        b, _ = probes.probe_layer(X, answer_word, scenes)
        c, _ = probes.probe_layer(X, follows, scenes)
        print(f"{layer:6d} {a:11.1%} {b:13.1%} {c:19.1%}")

    print("""
Read the first two columns as the ingredients and the third as the dish. Where
both ingredients are at ceiling, a linear probe can compute the third without
representing anything about instructions -- so a high score in the third column
is not evidence that the model encodes "about to obey". It is only evidence
once the first two are *not* both available.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
