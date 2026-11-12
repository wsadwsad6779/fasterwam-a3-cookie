# Video / Action / Force share one noise stream at inference — analysis only (2026-11-12)

Asked for as item 5 of the post-40k directive and re-listed as part of fix 8: the per-chunk seed
table is validated and stays, the correlated-noise issue is to be **analysed**, and `fw/` is not to be
modified. This document changes nothing; every claim below is a file:line in the frozen copy of the
training source under `_v1_stage/fw/src`, which is also the copy the 40k run trained on and the copy
the eval harness anchors `fasterwam` to.

## The claim

For one `infer_action` call, the three modalities' initial latents are drawn from **three generators
seeded with the same integer**, so they are not three independent draws — they are the same random
sequence read three times from the same starting position. Concretely, with

* action latent shape `(1, action_horizon=32, action_dim=7)` → 224 elements,
* force latent shape `(1, horizon=32, force_dim=6)` → 192 elements,
* video latent shape `(1, z_dim, latent_t, latent_h, latent_w)` → thousands of elements,

the flattened force vector equals the first 192 numbers of the action vector, and both equal the
first 224 / 192 numbers of the video vector.

## Evidence

`jointwam.py` — the entry point the eval harness calls (`ForcePolicy.chunk` →
`model.infer_action`):

```python
jointwam.py:183   video_generator  = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
jointwam.py:184   action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
jointwam.py:185   latents_video  = torch.randn((1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
                     generator=video_generator, device=rand_device, dtype=torch.float32)
jointwam.py:191   latents_action = torch.randn((1, action_horizon, self.action_expert.action_dim),
                     generator=action_generator, device=rand_device, dtype=torch.float32)
```

and the force side, reached through the same `seed` argument:

```python
fastwam.py:935    force_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
fastwam.py:936    latents_force   = torch.randn((1, cfg["horizon"], cfg["force_dim"]),
                     generator=force_generator, device=rand_device, dtype=torch.float32)
```

One `seed` in, three `manual_seed(seed)` out. `torch.Generator` is a fresh object each time, so no
state is shared — worse, each one *restarts* the same sequence. The same pattern is in the second
entry point (`jointwam.py:353-361`, i.e. `infer_action_one_pass_future_cache`, the mode the released
worker actually uses) and in `fastwam.py:1314-1322`. `fastwam.py:1476` is the action-only path, which
has one generator and therefore no cross-stream problem — that is the released model the bridge was
written against, and it is why nobody noticed.

The device matters for how exactly the sequences coincide: `EVALUATION.rand_device` is `cpu` in the
deployment config, so these are CPU generators (MT19937): same seed ⇒ same sequence, element for
element, deterministically. A CUDA generator is Philox-counter based and the per-element mapping is
kernel-block dependent, so the prefix equality should be asserted, not assumed — the probe below
prints it rather than reasoning about it.

## Why this is a train/inference mismatch, not merely a coupling

Training never reused a sequence. `fastwam.py:723,735,754`:

```python
noise_video  = torch.randn_like(input_latents)     # global default generator
... timestep_video = self.train_video_scheduler.sample_training_t(...)
noise_action = torch.randn_like(action)            # same stream, advanced past the video draw
... timestep_action = self.train_action_scheduler.sample_training_t(...)
noise_force  = torch.randn_like(force_future)      # same stream, advanced past both
```

Three sequential draws from **one** generator, with `sample_training_t` calls interleaved, so the
three noise tensors are disjoint segments of a single stream. `fastwam.py:750-751` even states the
intent in a comment — "one shared probability path parameter `t`, two independent noise
realizations" — and independence in training came from the stream *advancing*, which inference
undoes by re-seeding. The model therefore learned on independent triples and is being handed
identical-prefix triples at eval time.

Two consequences worth naming:

1. **Lost diversity.** A chunk's exploration is one draw, not three. The action and force heads
   cannot disagree about where they started, which is exactly the axis a joint sampler uses to
   decorrelate its outputs.
2. **Correlated failure.** One bad realisation hits all three streams at once, so the ~60 replans of
   an episode under-represent the spread the per-chunk seed table was built to produce. This does not
   bias any single seed — it makes the seed *less* effective as an independent sample, which inflates
   the variance of a 20-seed success rate and is a legitimate line in the final report's
   "unverified assumptions" section either way it is fixed.

The 2026-11-11 per-chunk fix (`policy40k.noise_seed(seed, call)`) is orthogonal and stays: it made
the seed vary across replans. It did not, and was not asked to, decorrelate the three streams inside
one call.

## Options

**A. Eval-side only — hand the model `seed=None` and own the global RNG per chunk.**
`torch.randn(..., generator=None)` falls back to the process default generator, so the three draws
become three *sequential, non-overlapping* segments of one stream — which is exactly what training
did, and is arguably a closer match to it than any independent-seed scheme. The harness would do,
immediately before each `infer_action` call:

```python
torch.manual_seed(noise_seed(episode_seed, chunk_index))     # or set_default_generator
pred = model.infer_action(..., seed=None)
```

* Cheapest and touches only `_v1_stage/dev/` — inside the authorisation.
* Determinism is preserved per chunk **only if nothing else in the call consumes the default
  generator**. Candidates to check before adopting: VAE encoding of the input image, `encode_prompt`,
  any `dropout` (the module is in `eval()` after fix 3, so dropout is off), and the scheduler's own
  sampling. This is a measurable question: run the same seed twice in one process and require
  bit-identical actions; if that fails, some other consumer exists and option A is rejected.
* Risk: the default generator is global state shared with anything else in the process, so two
  policies in one process would interfere. The single-model-per-process rule already holds.

**B. Per-stream seeds inside the model** — `video_generator.manual_seed(seed)`,
`action_generator.manual_seed(hash((seed, 1)))`, `force_generator.manual_seed(hash((seed, 2)))`.
Three lines, and the honest fix, but it is `fw/` and therefore **not authorised**; it also silently
changes the numbers of every comparison already planned at 40k, so it should only be done as a
declared new experiment, not as an eval repair.

**C. Do nothing for the 160-episode run, measure and report.** Keep `infer_action(seed=ns)` exactly as
the frozen protocol says, and state the correlated-stream effect as a known property of the measured
configuration. This is what the current authorisation implies (fix 8: 「保持已验证的逐 chunk 噪声种子
机制」 and 「不要为了修复评估错误改变模型结构」), and it is the default here.

Recommendation: **C for the 160-episode run, then A as a separate ablation** — one node (40k), the
same 20 seeds, both noise handlings, so the cost of the correlation is a number instead of an
argument. Only if A reproduces bit-identically does it become the deployment path.

## The measurement (probe design, to be added only after the sync — it needs the pod)

Capture the three draws without modifying `fw/`: wrap `torch.randn` for the duration of one
`infer_action` call in the smoke's existing noise probe, record `(shape, first 8 values, tensor)`,
then restore.

```python
captured = []
real_randn = torch.randn
def spy(*shape, **kw):
    t = real_randn(*shape, **kw)
    captured.append((tuple(shape), t.detach().float().flatten()))
    return t
torch.randn = spy
try:
    model.infer_action(**kwargs)          # one chunk, seed = noise_seed(s, c)
finally:
    torch.randn = real_randn
```

Assertions to record, not to assume:

1. exactly three captured draws in the expected shape order (video, action, force);
2. `torch.equal(action_flat[:192], force_flat)` — the prefix equality;
3. `torch.equal(video_flat[:224], action_flat)`;
4. across two chunks of the same episode (`c=1`, `c=2`), the captured tensors differ — the per-chunk
   table still works after any change to the seeding path;
5. for option A only: the same `(seed, chunk)` replayed twice in one process is bit-identical, which
   is what licenses touching the global generator at all.

Then the quantity that actually matters for the report: with the correlated streams, how much of the
20-seed spread of success is explained by the seed at all. Same seeds, both handlings, on one node —
that is the ablation in the recommendation, and its result is the only thing that should change the
protocol.

## What must not be concluded from this document

Nothing here has been measured on the model. Items 2 and 3 are a consequence of reading
`manual_seed(seed)` three times with the same integer, which is a property of the code, and the
prefix-equality numbers are stated as predictions for the probe to confirm — the CUDA-vs-CPU
generator caveat above is exactly why they belong in a probe and not in a report table. No success
rate, noise number, or force metric in the final report may cite this file as evidence; it is
evidence about the code only.

One verification left open by the blocked channel: the line numbers above are from the local mirror
`_v1_stage/fw/src/fasterwam/models/wan22/{jointwam,fastwam}.py`
(`jointwam.py` sha256 `3212e692dd823302d5a32be8f8e56acb0b399934592c94a16acf7f7de6867175`,
`fastwam.py` `df5894614ddad54b83c0f363d52f22fffe90daad57840d4d06979303dff5931b`). At sync time, hash
the pod's two files against those; if they differ, re-read
`sed -n '180,200p' jointwam.py` and `sed -n '930,945p' fastwam.py` on the pod before this analysis is
used for anything.
