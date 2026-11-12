"""Pre-flight for the boundary checker: does the corpus lookup work on the live run?

S3 asserts that one dataset length explains the sampler counters of every node. Before
node 1 exists there is nothing to explain, so this only checks that the lookup finds the
config, the dataset directory it points at, and a row count, and that the arithmetic
predicts where the first epoch wrap should appear.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/_v1_stage/fw/scripts")
sys.path.insert(0, "/workspace/_v1_stage/fw/src")

import segment_check as sc  # noqa: E402
import drive40k as plan  # noqa: E402

ds = sc.dataset_batches(plan.RUN)
print("dataset_batches ->", ds)
if ds:
    bpe = ds["batches_per_epoch"]
    print("first wrap at batch %d => global_step %d (node 1 is step %d: %s)"
          % (bpe, bpe, plan.NODE, "before the wrap" if plan.NODE < bpe else "after the wrap"))
    print("epoch/batch_in_epoch predicted at each node:")
    for i in range(1, plan.SEGMENTS + 1):
        g = plan.NODE * i
        print("   step %6d -> epoch %d, batch_in_epoch %d" % (g, g // bpe, g % bpe))
else:
    print("LOOKUP FAILED: S3 would report dataset=null at the boundary")
