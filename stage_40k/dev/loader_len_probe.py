"""Measure the loader, don't infer it: len(dataset), len(loader), steps per epoch.

The corpus is 42,681 frames, and frames are not what the trainer iterates. The model
trains on `num_frames`-long windows, `ResumableEpochSampler` permutes the window indices
once per epoch, and the DataLoader batches them with `drop_last` at its default. Three
numbers therefore have to be read off the pipeline rather than divided out of a summary
file, and the third one decides where the epoch boundary falls for all eight checkpoints.

CPU only, `nice`d, and it never touches the GPU or the training files.
"""
import os
import sys
import time

sys.path.insert(0, "/workspace/_v1_stage/fw/src")

import math  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from fasterwam.utils.samplers import ResumableEpochSampler  # noqa: E402

BATCH = 6
SIZE = 39481  # what the live trainer printed: "Train/val dataset size: 39481/39481"


class FakeWindows:
    """A Sized stand-in with the same length as the frozen window list.

    Only the plumbing is under test here -- sampler, batching, drop_last -- so the
    samples themselves are integers, not 33-frame windows. This is a test fixture and is
    labelled as such everywhere it is printed.
    """

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


print("== fixture only: sampler + DataLoader arithmetic at len(dataset)=%d ==" % SIZE)
ds = FakeWindows(SIZE)
sampler = ResumableEpochSampler(dataset=ds, seed=42, batch_size=BATCH, num_processes=1)
loader = DataLoader(ds, batch_size=BATCH, shuffle=False, sampler=sampler, num_workers=0)
print("   len(sampler)        = %d" % len(sampler))
print("   len(train_loader)   = %d   (ceil(%d/%d) = %d)"
      % (len(loader), SIZE, BATCH, -(-SIZE // BATCH)))
print("   drop_last default   = %s -> last batch has %d samples"
      % (loader.drop_last, SIZE % BATCH or BATCH))
print("   optimizer steps/epoch at grad_accum=1, num_processes=1 = %d" % len(loader))

t0 = time.time()
seen = 0
batches = 0
first = last = None
for b in loader:
    batches += 1
    seen += int(b.shape[0])
    if first is None:
        first = b.tolist()
    last = b.tolist()
print("   one real iteration  = %d batches, %d samples in %.1fs" % (batches, seen,
                                                                    time.time() - t0))
print("   first batch         = %s" % first)
print("   last batch size     = %d" % len(last))
print("   samples == dataset? = %s" % (seen == SIZE))

print("\n== what a resume at each node actually skips ==")
for node in range(1, 9):
    g = 5000 * node
    epoch = g // len(loader)
    batch_in_epoch = g % len(loader)
    sampler.set_epoch_offset(epoch)
    sampler.set_resume_batch_offset(batch_in_epoch)
    n = len(list(sampler))
    print("   step %6d -> epoch %d, batch_in_epoch %4d, remaining samples %5d, "
          "batches left %d" % (g, epoch, batch_in_epoch, n, -(-n // BATCH)))
sampler.set_epoch_offset(0)
sampler.set_resume_batch_offset(0)

print("\n== the live node 1 prediction under this measured epoch length ==")
print("   first wrap at optimizer step %d, which is %d steps past node 1 (5000)"
      % (len(loader) + 1, len(loader) - 5000))
print("   segment 2 (5000->10000) therefore crosses an epoch boundary inside itself")
