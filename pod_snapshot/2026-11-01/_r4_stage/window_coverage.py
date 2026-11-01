"""Which training windows did a run actually visit, and were they new ones?

`r4_patch.py` writes the sampler order to `FW_WINDOW_LOG` as `<pass> <index>`
lines, so this is a measurement, not an inference from the step count. The
permutation itself is regenerated here from the same seed the sampler uses,
which is what makes "was this window already trained in r2/r3" answerable.

Run inside the pod venv:

    python3 window_coverage.py /workspace/a3_train_runs/a3_lora_r4/windows.log \
        --steps 4000 --batch 6 --size 39481 --prior 18000

`--prior` is how many windows r2/r3 consumed from the head of the same
permutation (both ran 3,000 steps at batch 6 from offset 0), so everything from
`permutation[prior:]` is what r4 was supposed to reach for the first time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def permutation(seed: int, size: int, offset: int) -> list[int]:
    """The sampler's pass-0 order: randperm(seed) with the head skipped."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randperm(size, generator=gen).tolist()[offset:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--size", type=int, default=39481)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offset", type=int, default=18000)
    ap.add_argument("--prior", type=int, default=18000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    for line in Path(args.log).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        part, idx = line.split()
        rows.append((int(part), int(idx)))

    trained_n = args.steps * args.batch
    trained = [idx for _, idx in rows[:trained_n]]
    prefetch_tail = rows[trained_n:]

    expected_pass0 = permutation(args.seed, args.size, args.offset)
    prior_seen = set(expected_pass0[:0])  # placeholder, filled below
    full_pass = permutation(args.seed, args.size, 0)
    prior_seen = set(full_pass[:args.prior])
    unvisited = [w for w in full_pass[args.prior:]]
    unvisited_set = set(unvisited)

    # The log must be the permutation, not something that merely overlaps it.
    head_matches = sum(1 for a, b in zip(trained, expected_pass0) if a == b)
    distinct = set(trained)
    repeated = len(trained) - len(distinct)
    new_windows = sorted(distinct - prior_seen)

    out = {
        "log": str(args.log),
        "lines_logged": len(rows),
        "steps": args.steps,
        "batch": args.batch,
        "trained_indices_expected": trained_n,
        "trained_indices_logged": len(trained),
        "prefetch_tail_lines": len(prefetch_tail),
        "pass_values_seen": sorted({p for p, _ in rows}),
        "matches_permutation_head": head_matches,
        "distinct_windows_trained": len(distinct),
        "repeated_indices": repeated,
        "windows_already_seen_in_prior_runs": len(distinct & prior_seen),
        "windows_never_seen_before": len(new_windows),
        "unvisited_pool_size": len(unvisited_set),
        "unvisited_pool_coverage": round(len(new_windows) / max(len(unvisited_set), 1), 6),
        "prior_pool_size": args.prior,
        "first_8_trained": trained[:8],
        "last_8_trained": trained[-8:],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    if len(trained) < trained_n:
        print(f"WARN: only {len(trained)} of {trained_n} trained indices were logged "
              f"-- the run stopped before {args.steps} steps or the log was truncated.")
    if head_matches != min(len(trained), len(expected_pass0)):
        print("NOTE: the log leaves pass 0 (offset exhausted) or a resume skipped "
              "ahead; check pass_values_seen.")


if __name__ == "__main__":
    main()
