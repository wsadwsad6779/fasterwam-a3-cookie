"""Sampler pass-boundary test (CPU only, no model, no training data).

r4 opens at `sample_offset=18000`, which leaves 39,481 - 18,000 = 21,481 unvisited
windows. At batch 6 with `drop_last=False` (the stock `trainer.py:174` DataLoader
sets no drop_last) that is 3,580 full steps plus a final 1-sample step, so a
4,000-step run crosses the end of the index list at step 3,581 and spends 419
steps on the next pass. Whether those 419 steps are new data or a replay of the
head is exactly what `r4_patch.py` decides, so the patched sampler is checked
directly:

1. pass 0 is the same r2/r3 permutation with its first 18,000 windows dropped;
2. pass 1 is a *different* permutation, keyed on the trainer's epoch counter
   (the stock sampler ignores it and would rebuild the identical list);
3. a resume in the middle of pass 1 skips only the batches already consumed and
   does not re-apply `sample_offset`;
4. the 1-sample step is measured, and the offset that removes it is reported;
5. pass 0 alone takes cumulative coverage of the 39,481 windows from 45.6%
   (what r1-r3 together ever saw) to 100%.

Run on the A3 pod:
    /opt/venv/bin/python3 test_sampler_passes.py
"""
import math
import sys
import types
from pathlib import Path

import torch

BRIDGE = Path(__file__).resolve().parent
for p in (str(BRIDGE), "/workspace/FasterWAM/src", "/workspace/FasterWAM"):
    if p not in sys.path:
        sys.path.insert(0, p)

import fasterwam.r4_patch as r4_patch  # noqa: E402
from fasterwam.utils.samplers import ResumableEpochSampler  # noqa: E402

N = 39481
BATCH = 6
SEED = 42
OFFSET = 18000
STEPS = 4000
R2R3_SEEN = 18000  # r2/r3 each consumed the first 18,000 indices of pass 0


class FakeSet:
    def __len__(self):
        return N


def perm(pass_id):
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED + pass_id)
    return torch.randperm(N, generator=g).tolist()


def build(passes_done, resume_batch_offset=0, sample_offset=OFFSET):
    s = ResumableEpochSampler(FakeSet(), seed=SEED, batch_size=BATCH, num_processes=1)
    s.set_sample_offset(sample_offset)
    s.resume_batch_offset = resume_batch_offset
    s._trainer = types.SimpleNamespace(epoch=passes_done)
    return list(iter(s))


def main():
    r4_patch.install()
    p0, p1 = perm(0), perm(1)

    stream0 = build(0)
    stream1 = build(1)
    steps_pass0 = math.ceil(len(stream0) / BATCH)
    steps_after = STEPS - steps_pass0

    tail_len = steps_after * BATCH
    seq = stream0 + stream1[:max(0, tail_len)]
    seen_before = set(p0[:R2R3_SEEN])
    new_in_r4 = sum(1 for w in seq if w not in seen_before)
    union = len(seen_before | set(seq))

    clean_offsets = [o for o in (OFFSET, OFFSET + 1, OFFSET - 1)
                     if (N - o) % BATCH == 0]

    checks = {
        "pass0_is_r2r3_permutation_minus_prefix": stream0 == p0[OFFSET:],
        "pass0_has_no_repeats": len(stream0) == len(set(stream0)),
        "pass1_has_no_repeats": len(stream1) == N,
        "offset_moves_head": stream0[:BATCH] != p0[:BATCH],
        "pass1_is_new_permutation": stream1 == p1 and stream1[:BATCH] != p0[:BATCH],
        "pass1_not_rebuild_of_pass0": stream1 != stream0,
        "crosses_epoch_boundary": steps_after > 0,
        "resume_midpass_skips_consumed_only":
            build(1, resume_batch_offset=10) == p1[10 * BATCH:],
        "sample_offset_not_reapplied_after_pass0":
            build(1, resume_batch_offset=10)[0] != p1[OFFSET + 10 * BATCH],
        "repeats_only_across_passes":
            len(seq) - len(set(seq)) == len(set(stream0) & set(stream1[:tail_len])),
        "pass0_alone_closes_coverage": len(seen_before | set(stream0)) == N,
        "windows_sampled_matches_steps":
            len(seq) == STEPS * BATCH - (BATCH - len(stream0) % BATCH),
    }
    for k, v in checks.items():
        print("%s %s" % (k, v))
    print("steps_pass0 %d (last batch size %d)" % (steps_pass0, len(stream0) % BATCH or BATCH))
    print("steps_after_boundary %d" % steps_after)
    print("windows_distinct_in_r4 %d" % len(set(seq)))
    print("windows_sampled_by_r4 %d" % len(seq))
    print("windows_distinct_never_seen_before %d" % len(set(seq) - seen_before))
    print("occurrences_not_seen_by_r1r3 %d" % new_in_r4)
    print("cumulative_unique_coverage_pct %.1f" % (100.0 * union / N))
    print("partial_batch_free_offset %s" % clean_offsets)

    ok = all(v is True for v in checks.values())
    print("RESULT passed=%s" % ok)
    if not ok:
        raise SystemExit("sampler pass-boundary test FAILED")


if __name__ == "__main__":
    main()
