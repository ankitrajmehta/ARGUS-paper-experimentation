"""Print the DriveLM/Qwen run summaries from a fetched Kaggle output.

The notebook's own table is in the kernel log, but the per-scene JSONs are the
evidence; this reads them back so the numbers can be checked independently of
whatever the notebook printed.

    python scripts/summarise_drivelm_run.py results/kaggle_drivelm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The run record stores the rendered sign text but not the --style flag it came
# from. The notebook writes each condition into its own directory, so the
# parent directory name is the label -- which is the actual flag, not a guess
# reverse-engineered from wording.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?",
                        default=Path("results/kaggle_drivelm"))
    args = parser.parse_args(argv)

    rows = []
    for path in sorted(args.directory.rglob("*.json")):
        payload = json.loads(path.read_text())
        if "summary" not in payload:
            continue  # activation metadata, not a run record
        texts = {r["sign_text"] for r in payload["results"] if r.get("sign_text")}
        rows.append({
            "file": path.name,
            "started": payload["started_at"][:19],
            "condition": "no sign" if not payload["attacked"] else path.parent.name,
            "model": payload["model_id"],
            "summary": payload["summary"],
            "results": payload["results"],
            "n_signs": len(texts),
            "sample": sorted(texts)[0] if texts else "",
        })

    rows.sort(key=lambda r: r["started"])
    if not rows:
        raise SystemExit(f"no run records in {args.directory}")

    base = next((r for r in rows if r["condition"] == "no sign"), None)

    print(f"model: {rows[0]['model']}\n")
    head = (f"{'condition':12s} {'n':>4s} {'ASR':>7s} {'CHAI':>7s} {'vs clean':>9s} "
            f"{'benign-acc':>11s} {'unread':>7s} {'retry':>6s}")
    print(head)
    print("-" * len(head))
    for r in rows:
        s = r["summary"]
        delta = "--"
        if base and r is not base:
            delta = f"{s['asr_chai'] - base['summary']['asr_chai']:+.1%}"
        print(f"{r['condition']:12s} {s['n']:4d} {s['asr']:7.1%} {s['asr_chai']:7.1%} "
              f"{delta:>9s} {s['benign_accuracy']:11.1%} {s['unreadable_rate']:7.1%} "
              f"{s['retry_rate']:6.1%}")

    n = rows[0]["summary"]["n"]
    print(f"\nn = {n}: one scene is {1 / n:.1%}, 95% interval about "
          f"+/-{98 / n ** 0.5:.0f} points.")

    # Per-scene agreement with the baseline: which scenes the sign actually
    # flipped, as opposed to scenes that already answered the target.
    if base:
        base_by_index = {r["scene_index"]: r for r in base["results"]}
        print("\nper-scene flips (clean -> attacked):")
        for r in rows:
            if r is base:
                continue
            flipped = held = rescued = 0
            for res in r["results"]:
                prior = base_by_index.get(res["scene_index"])
                if prior is None:
                    continue
                was = prior["action"] == prior["target"]
                now = res["action"] == res["target"]
                flipped += (not was) and now
                held += was and now
                rescued += was and not now
            print(f"  {r['condition']:12s} flipped {flipped:3d}  "
                  f"already-target {held:3d}  lost {rescued:3d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
