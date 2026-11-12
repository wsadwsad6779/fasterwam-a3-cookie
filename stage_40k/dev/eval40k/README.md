# eval40k — how the eight 5k checkpoints get evaluated, and what was measured on CPU

Everything here runs without the training GPU except the real rollouts themselves, and no rollout
has been started yet: the batch eval is deliberately scheduled after step 40,000 so that no node
competes with the main training for the card (`AGENTS` rule from the 40k brief: 先做评估流程测试,
不在当前训练 GPU 上启动大规模评估).

## The protocol, as of 2026-11-11

Per checkpoint: **the 20 screened seeds, exactly one episode each** → 20 episodes/node, 160 for the
eight nodes. This replaces the earlier "20 seeds × 3 repeats" (60/node, 480 total), which the user
dropped to shorten the post-40k window. Dropping the repeats costs no information here, because an
episode is reproducible from its own seed alone -- `policy40k.noise_seed(episode seed, chunk index)`
fixes the noise of every replan -- so a second episode at the same seed re-plays the same rollouts
rather than sampling new noise: three repeats would have been three near-identical episodes. What the
change does buy is ~3× less wall clock on the card.

One thing had to be fixed before that argument worked. `infer_action` keeps no sampler state between
calls: it builds `torch.Generator(...).manual_seed(seed)` afresh (`jointwam.py:183-184` for the video
and action latents, `fastwam.py:935` for the force latents) and the latent shapes are constant, so an
episode that passed its seed unchanged drew **the identical initial noise on all ~60 replans** -- a
single unlucky realisation repeated, and a distribution training never used (`torch.randn_like` per
window). `ForcePolicy` now derives one seed per chunk from `(episode seed, chunk index)`, records it
in that chunk's `meta["noise_seed"]`, and `flow_test.py` asserts the sequences are distinct within an
episode, reproducible from the pair, and not shared between screened episodes. Note the stub policy
could never have caught this: its rng is built once per episode and advances by itself.

Two related properties are left alone, because they live in `fw/` and the seed contract is frozen for
this run: one `seed` value still initialises the video, action *and* force latents, so their noise
draws come from the same stream and overlap element by element (verified on CPU: the first 224 values
of the video latents equal the action latents); and nothing in the harness can split that without
touching the model. Worth a change after 40k, not during it.

### The seed table, verified from the loop rather than from the function

`check_noise_seeds` proves the table is good; it does not prove the harness looks at it. So
`check_rollout_seeds` drives the real `rollout40k.run_episode` with a stand-in model whose only
behaviour is recording the `seed=` keyword it receives, and every assertion is on that recording: 60
chunks for a 600-step episode at replan 10, `replan_from_frame` advancing 0, 10, 20, ..., and each
call's seed equal to `noise_seed(episode seed, chunk index)`, with `meta["noise_seed"]` holding the
same value. The regression is replayed as well -- `policy40k.noise_seed` temporarily replaced by a
constant, the same episode rolled again, and the check requiring the recording to collapse to one
seed -- so the test is armed rather than decorative.

The full 20 seed x 60 chunk range is hashed rather than sampled: 1,200 draws, 1,200 distinct, zero
collisions. The table itself is dumped into the result JSON under `noise.table`, with its digest in
`noise.table_sha256` (`0801c61fc8830d1e3ac35f644fd01ac8ffa0e3ee7eade65a99c2a1bfedeaed9b`), because it is
the reference the eight nodes get compared against afterwards and a rerun or a restarted node has to
reproduce it entry for entry. `noise_seed` takes no checkpoint-node argument, so all eight nodes replay
one identical table -- which is what the paired per-seed comparison in `report40k.py` depends on.

Two boundaries on that, stated so the post-40k smoke does not get surprised. The proof is on the harness
side: `infer_action`'s own consumption of `seed=` is read from the frozen `fw/` source
(`jointwam.py:183-184`, `fastwam.py:935`), not executed here, and the first confirmation on real weights
is the inference smoke after step 40,000. And `eval40k.py` does not persist chunk `meta` -- the per-seed
artifacts are `measured_seed<N>.npz` and `episode_seed<N>.json` -- so the seeds a real node actually
consumed can only be audited by re-deriving them from this table, never by reading them back off a
finished rollout.

The list is not hard-coded any more. `eval40k.py` and `batch_eval.py` read `valid_seeds` out of
`../../dev/out40k/seed_screen.json` and refuse to start unless it holds 20 distinct, screened seeds;
`--seeds` still overrides, but a repeated seed is refused because `episode_seed<N>.json` and
`measured_seed<N>.npz` are keyed by seed, so a repeat would overwrite one file while counting two
episodes. The 20 come from `seed_screen.py --want 20 --scan 40 --start 11` (seeds 11–30, all legal
and reproducible); the previous 12-seed default mixed in 31/32/33/41/42/51/52, which were never
screened at all.

Checked on CPU the day the protocol changed (`batch_eval.py --fake-merged --steps 5000 10000`, fixture
backend + stub policy, no GPU): `plan.json` holds seeds 11..30, `episodes_per_node` is 20, and every
node wrote exactly 20 `episode_seed<N>.json` + 20 `measured_seed<N>.npz` with no seed used twice.
`--seeds 11 11 21` is refused before a checkpoint is opened. Wall clock on the real eight nodes drops
from 480 rollouts to 160, i.e. the post-40k eval window is about a third of what was planned.

## The pieces

| file | what it is |
| --- | --- |
| `eval40k.py` | one node, the 20 screened seeds, one episode each: merged checkpoint → policy → closed-loop episodes → `summary.json` + per-seed npz |
| `rollout40k.py` | the closed loop itself (B0, bare policy, no safety layer) |
| `force_stream.py` | the frozen force contract: channels `[0,1,2,12,13,16]`, hist 16, horizon 32, `force-calib-v1` |
| `force_metrics.py` | prediction-vs-measurement accounting with the parked-chunk rule (a frame can be predicted by more than one chunk) |
| `policy40k.py` | real model policy + the deterministic `StubPolicy` used by the flow test |
| `report40k.py` | cross-node table, per-channel MAE, paired per-seed comparison, warnings |
| `flow_test.py` | 33 hand-computed checks of the whole pipeline against a fixture world |
| `seed_screen.py` | CPU-only prescreen of evaluation seeds: validity, determinism, diversity |
| `knob_probe.py` | CPU-only sweep of every scene-randomisation knob the env exposes: what each one is worth as a diversity split, before and after the physics settles |
| `batch_eval.py` | the post-40k loop: per node merge → evaluate → verify → delete the 11 GiB artifact, with a ledger and the cross-node report at the end |

```sh
cd /workspace/_v1_stage/dev/eval40k
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python flow_test.py --json ../../dev/out40k/flow_test.json
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python eval40k.py --backend fixture --policy stub \
    --fixture-force-mode const --steps 5000
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python report40k.py --out ../../dev/out40k/evalflow
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python seed_screen.py --want 20 --scan 60
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python knob_probe.py --count 5 --seeds 11 --settle-steps 50
```

## What has actually been measured (pod, CPU only; 2026-11-10, seed checks redone 2026-11-11 UTC)

* `flow_test.py` — **33/33 checks pass**, run twice on the pod CPU with `CUDA_VISIBLE_DEVICES=''` and
  `nice -n 19` so nothing could reach the training card. The two result files are byte-identical:
  `../../dev/out40k/flow_test_run1.json` and `../../dev/out40k/flow_test_run2.json`, 47,536 B each,
  SHA256 starting `4d8f6023b7938c6e`. Of those 33, 24 are the original pipeline checks; the 9 that came
  in with the seed fix are 6 about `noise_seed` as a table and 3 about the rollout probe above.
* Driver-level fixture flow (`--fixture-force-mode const`, stub offset 0.5 N): force MAE is
  **0.500 N on all six channels**, coverage 0.4945, `vs zero` 0.3488, success 3/6 with Wilson
  95% CI [0.1876, 0.8124], npz shapes `measured (34,6)`, `chunk_frames (4,)`, `chunk_pred
  (4,32,6)`. `report40k.py` grades this itself: `expected MAE 0.5000 N, worst channel 0.5000 N --
  PASS`. A constant-force world makes the persistence baseline exact, so the only error left is
  the offset the stub adds; that is what makes the number checkable rather than plausible.
* Shadow wiring (`--shadow`): 17 decisions logged, `verdict_counts {hold: 17}`,
  `actions_mutated 0`, `feedback_applied_decisions 0`. Every `hold` on the fixture is explained by
  the fixture's non-physical force trace, so the honest false-positive rate was measured
  separately on real force data — see `../p4_shadow/README.md`.
* `seed_screen.py --want 20 --scan 40 --start 11 --recheck 3` — **20/20 seeds valid** (11–30),
  `scene_sanity.ok` true, `max_penetration_mm 0.0`, `cookies_in_target 0`, force finite, and the
  recheck pass reproduced all three re-checked seeds (`unreproducible: []`).
* `knob_probe.py --count 5 --seeds 11 --settle-steps 50` — the ten-cell grid below, 50 env builds
  and 250 bare `mj_step` calls, no model loaded and no GPU touched.
* `report40k.py`'s scene warning: exercised on a **synthetic** env-backend `summary.json` (written
  by hand with `distinct_scene_keys=1, episodes=6`, not a model rollout), which produced
  `W: step 5000: all 6 episodes start from the same scene (position_noise_m=0.0,
  yaw_noise_rad=0.0, randomize_boxes=False), so this success rate is repeatability on one layout,
  not generalisation across layouts`; the genuine fixture run correctly stays silent, because
  `eval40k.py` only writes the `scene` block for the env backend. This proves the reporting path,
  not any policy.

## The seed/diversity finding, stated plainly

A seed is not a scene. `env_factory.reset(env, seed, randomize_cookies=True)` draws nothing unless
`make_env` was given a non-zero `position_noise_m`/`yaw_noise_rad` or `reset` was given
`randomize_boxes=True`/`batch_offset_m`, and `eval40k.py`'s defaults for all four are zero/False.
Measured, hashing the `cookie_*_free` joint qpos: a 12- or 20-episode run at the defaults measures
**repeatability of one scene**, where the episode-to-episode spread comes from the policy's own
diffusion seed (`policy40k` derives every chunk's `seed=` from it), not from layout variation.
`eval40k.py` records this per node (`summary["scene"]`, with the distinct hash count) and
`report40k.py` warns when a node reports more episodes than distinct scenes. The line was exercised
on a synthetic env-node summary (see above), and reads:
`W: step 5000: all 6 episodes start from the same scene (position_noise_m=0.0, yaw_noise_rad=0.0,
randomize_boxes=False), so this success rate is repeatability on one layout, not generalisation
across layouts`.

## Every randomisation knob, measured (`knob_probe.py`, seeds 11–15, CPU)

Rather than argue about which knob *should* work, all ten settings the env exposes were run and
judged twice: at frame 0 (the state `reset` hands back) and after 50 more bare `mj_step` calls with
no action applied. 50 steps at `model.opt.timestep` = 0.001 s is **0.050 s of simulated time,
exactly one 20 Hz control frame** (`physics_hz: 1000`, `control_hz: 20`); `reset` already ends with
its own 50 steps + `mj_forward`, so the second column is the *second* frame of the episode, which is
what a policy chunk actually sees. "valid" = `scene_sanity.ok` **and** no cookie outside the source
bin; the gate `scene_sanity` uses is `max_penetration_mm <= 0.5`.

| cell | knobs | f0 valid | f0 distinct | f0 worst pen | f0 median shift | +0.050 s valid | +0.050 s distinct | of those, E0 again | median drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `E0_eval_defaults` | shipped defaults, all zero | 5/5 | **1** | 0.000 mm | 0.000 mm | 5/5 | **1** | **5** | 0.000 mm |
| `boxes_jitter` | `randomize_boxes=True` | 5/5 | 5 | 0.000 mm | 0.743 mm | 5/5 | 5 | 0 | 0.000 mm |
| `cookie_noise_0.3mm` | `position_noise_m=0.0003` | 5/5 | 5 | 0.006 mm | 0.393 mm | 5/5 | 5 | 0 | 0.001 mm |
| `cookie_noise_1mm` | `position_noise_m=0.001` | **1/5** | 5 | 0.626 mm | 1.272 mm | — | — | — | — |
| `cookie_noise_25mm` | `position_noise_m=0.025` | **1/5** | 5 | 5.901 mm | 33.909 mm | — | — | — | — |
| `batch_offset_8mm` | `batch_offset_m=0.008` | **5/5** | 5 | 0.000 mm | 7.997 mm | **5/5** | 5 | 0 | 0.000 mm |
| `batch_offset_10mm` | `batch_offset_m=0.010` | 4/5 | 5 | 0.000 mm | 9.996 mm | 5/5 | 5 | 0 | 0.000 mm |
| `batch_offset_20mm` | `batch_offset_m=0.020` | 1/5 | 5 | 0.000 mm | 19.992 mm | **2/5** | 5 | 0 | 29.879 mm |
| `batch_offset_50mm` | `batch_offset_m=0.050` | 0/5 | 5 | 0.000 mm | 49.979 mm | 0/5 | 5 | 0 | 49.188 mm |
| `boxes_and_offset_20mm` | both of the last two | 1/5 | 5 | 0.000 mm | 22.849 mm | — | — | — | — |

`distinct` counts different qpos hashes over **all** five seeds, including the invalid ones, which is
why a destroyed scene still shows 5 — read it together with `valid`, never alone. Two further
columns that were measured because they matter for the pixels: the target-bin pose is distinct in
5/5 seeds only for `boxes_jitter` (it is ±2 mm + ±0.015 rad there, fixed at `fd82b044` everywhere
else), and the non-geometry `randomization_metadata` — textures, lighting, camera — is **1 distinct
in every cell**. No knob touches the appearance; a scene differs only where the cookies are.

Five things that were not predictable from the source code:

* **The defaults are not "a little random": they are exactly zero.** All five E0 seeds share one
  geometry hash, one target-bin hash *and* one appearance hash. A seed changes neither the task nor
  the pixels.
* **Per-cookie noise has a hard ceiling of about 0.3–0.5 mm, and it is arithmetic, not luck.** The
  source bin is a 4 x 20 grid of 80 slots (`cookie_source_positions_m`) and the cookie is 50 mm x
  6.3 mm x 25 mm (`cookie_half_size_m = (0.025, 0.0031667, 0.0125)`):

  | axis | slot pitch | cookie extent | clearance left | half-clearance | 25 mm noise is |
  | --- | --- | --- | --- | --- | --- |
  | x | 50.40 mm | 50.0 mm | 0.40 mm | 0.20 mm | 125x too big |
  | y | 8.833 mm | 6.33 mm | 2.50 mm | 1.25 mm | 20x too big |

  The bin is packed edge-to-edge, so anything larger than half the x clearance overlaps a neighbour:
  0.3 mm passes at 0.006 mm penetration, 1 mm already reaches 0.626 mm and fails the 0.5 gate in
  4/5 seeds, 25 mm reaches 5.901 mm with cookies displaced a median of 33.9 mm and produces
  `Nan, Inf or huge value in QACC at DOF 501. The simulation is unstable`.
* **The solver undoes large per-cookie noise instead of diversifying it.** At 25 mm exactly one of
  the five seeds survives `scene_sanity`, and that seed's layout is E0 again — hash `bf4f9b7f`, all
  80 cookies back on their nominal slots. The neighbour contact forces push an overlapped cookie out
  of the pile before the episode starts. So big noise does not buy diversity: it buys either an
  invalid scene or the same scene as before.
* **`batch_offset` is the only knob that gives a real layout change, and its cliff is the bin wall,
  not the cookies.** Penetration stays 0.000 mm at every magnitude because all 80 cookies are
  translated by one shared vector — they move as a rigid group and never interpenetrate. What breaks
  is the boundary: the bin has roughly 15 mm of slack per side in x and 10 mm in y, so 8 mm keeps
  every cookie inside (5/5 valid, shifts 3.6–8.6 mm), 10 mm puts a few outside at frame 0 (4/5),
  20 mm leaves 1/5, and 50 mm leaves none with cookies flung ~126 mm. Above the cliff the scene is
  not merely flagged, it is moving: median drift 29.9 mm in the first control frame at 20 mm.
* **`eval40k.py`'s gate runs at frame 0 and therefore over-rejects.** At 10 mm, four of five seeds
  fail at frame 0 but all five are valid and all five distinct one control frame later — seed 12's
  four escaped cookies come back on their own. That is an argument for 8 mm rather than 10 mm: 8 mm
  passes the gate as it is written, so no eval run needs the escape hatch
  `--allow-invalid-scene`, which would also let a genuinely broken scene through.

One more fact worth keeping, because "the config default" and "what the eval gets" are different
numbers: `SimConfig`'s own `object_position_noise_m` is **0.025 m**, but the bridge's
`make_env(position_noise_m=0.0, ...)` overrides it to zero — and the corpus was recorded through
`make_env`, so the model has only ever seen the packed zero-noise layout.

## What that means for the final report

The eight-node table has two defensible splits and they must not be averaged into each other:

* **Fixed layout (the default).** `--batch-offset 0.0`, seeds 11–30 as prescreened. One scene, N
  episodes. This is in-distribution and it measures the policy's own sampling behaviour, which is
  what a success rate over N seeds is meant to summarise; `report40k.py` labels it, and it is
  honestly described as repeatability rather than generalisation.
* **Diversity split: `--batch-offset 0.008`.** Measured above — 5/5 seeds valid at frame 0 and after
  one control frame, 5 distinct layouts out of 5 seeds, zero penetration, zero drift, shifts of
  3.6–8.6 mm that keep the whole grid inside the bin. It is the only setting in the grid that is
  both legal and a different task. It is **out-of-distribution**: the source bin is 8 mm off-centre
  relative to everything the corpus contains, so it answers "does the policy transfer", not "how
  good is it on the trained scene".

`boxes_jitter` and `cookie_noise_0.3mm` produce distinct hashes for every seed and pass every gate,
but a sub-millimetre shift is a different hash, not a different task; they are the right choice only
if the question is "does the eval pipeline notice layout changes at all", not "does the policy
generalise".

## The batch driver, and what its flow test caught

`eval40k.py` resolves and scores a node and deliberately never deletes: a merged artifact is
11.1 GiB (measured, task #64), the volume had 43.19 GiB free during training, and a hand-run loop of
eight merge/eval/delete cycles is where an 11 GiB mistake gets made. `batch_eval.py` writes that loop
down, with the rules as code:

* **an interlock, not a reminder**: it refuses to start while the trainer looks alive — the segment
  log written within `--stale-log-seconds` (default 300) or a live `scripts/train_seg.py` process —
  because an 11 GiB load on the same card would slow the 40k run and change no result;
* **one artifact at a time**: if a *different* node's merged file is on disk, the loop aborts rather
  than merging on top of it;
* **an existing artifact is reused, not re-merged**, when its publish record is still there, so a
  failed eval costs a re-eval and not another 11 GiB export;
* **a merged file is deleted only after every split it was meant to score has a complete
  `summary.json`** (episodes = seeds, a success rate, a force block), and the freed bytes are written
  to `batch_eval_ledger.jsonl` together with the artifact's SHA256;
* **the merge is not trusted on exit code alone**: `published_step` must be the step being merged,
  `published_force_calibration` must equal the fingerprint in that node's own `trainer_state.json`,
  and `output_bytes` must equal the file that actually landed.

What has been measured (pod, CPU only, no GPU touched, training uninterrupted):

* `--dry-run` prints one JSON document with the pre-flight, the exact merge and eval commands per
  node, and the blockers. Its blocker text right now is the interlock working:
  `trainer looks alive (.../run40k/logs/seg2.log last written 35 s ago, pids ['50963', '63152', ...
  9 pids'])`, and it exits 1. Pre-flight reported `free_gib_before 43.19` against
  `merge_needs_gib 15.53` (the 1.5x margin).
* `--fake-merged --diversity --steps 5000 10000` runs the whole cycle against a 2 MiB placeholder and
  the stub policy in the fixture world. The ledger line: `merged_sha256 5647f05e…`, both splits
  complete (`success 5/12`, per-channel MAE `[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]` N, `mae_vs_zero 0.3488`,
  `force_coverage 0.5473`, `nan_aborts 0`), then `deleted_merged.bytes 2097152` and
  `stray_merged_after: []`. Step 10000 is skipped as `no_light_node`, which is the honest state while
  training is at 6,800.

Four bugs came out of that flow test, all four in the driver and none in the scoring path:

1. the fake publish record carried the version *string* `force-calib-v1` where the node carries the
   fingerprint *dict*, so `check_merge` was comparing unlike shapes — the fix makes the placeholder
   copy the node's own dict, which is what a real record is checked against;
2. a leftover artifact from a failed node blocked the next run *and* had no reuse path, so the run
   could only be restarted by hand;
3. the child's log file was opened before its split output directory existed;
4. `summary_complete` accepted `status="ok"`, and `"ok"` is a status `eval40k.py` never writes — the
   real values are `ran` and `no_light_node`.

What the flow test does **not** prove: no weight was loaded, no real environment episode was run,
and `distinct_scenes` is `null` in these lines because the fixture backend does not write the `scene`
block at all. The 5/12 is the stub's, on the fixture world, and `report40k.py` labels it `(test run)`
wherever it appears.

## The interface defects that stood between this harness and a real checkpoint

Found by running it, in the order they appeared. None of them is visible by reading one file, which
is why each one now has an assertion attached to it rather than a comment. Items 1-5 were found by the
first attempt to load a real checkpoint; items 6-10 were found only by the first attempt to *finish*
two rollouts with one, which is a different class of bug: the model answered, and the harness
described the answer wrongly.


1. **Wrong `fasterwam` gets imported.** Three copies exist on the pod. `bridge/_next/env_factory.py`
   front-loads `/workspace/fasterwam_a3_bridge/FasterWAM/src`, a released copy whose
   `create_fasterwam` has **no `force` parameter**; `infer_a3_worker` imports `fasterwam.datasets...`
   at module level, so after the bridge is imported the package is bound there and a later
   `sys.path` edit cannot undo it. The fix is to `import` the `fw/src` copy *first*
   (`eval40k.use_stage_fw()`), and the check is object-level, not path-level: the package
   `__file__`, `runtime.__file__` and `create_fasterwam.__module__`'s file must all sit under
   `fw/src`, and the factory must really accept `force`.
2. **`cfg.EVALUATION` does not exist in the training config.** `run40k/config.yaml` has no
   `EVALUATION:` block at all; the block lives in the released `sim_libero.yaml`, and two of its
   values are Hydra interpolations (`${eval_num_inference_steps}`,
   `${default_action_infer_mode}`). `attach_eval_settings()` copies only the keys that file
   actually defines and resolves those two against this run's own config. It refuses to invent a
   default, because an invented `num_inference_steps` is a different model call than the one being
   measured.
3. **The model stayed on the CPU in train mode.** `load_checkpoint` does not move or freeze
   parameters; the worker does `model.to(device).eval()` after loading and so must this harness.
   `smoke40k.py` asserts the parameter device and the `training` flag.
4. **The processor was built with a call that does not exist.** `FastWAMProcessor(stats=...)` is not
   a signature in either copy; the worker does `instantiate(cfg.data.train.processor).eval()` and
   then `processor.set_normalizer_from_stats(stats)`.
5. **8-wide proprio against a 14-wide model.** `infer_a3_worker.build_proprio` returns eight columns
   (left EEF pos 3 + axis-angle 3 + 2 finger joints), and it was right for the model it was written
   against. These checkpoints' `shape_meta` declares `raw_shape: 14`, and
   `build_v1_dataset.py:116,130` assembled columns 8-13 as *gathered raw newtons of channels
   `[0, 1, 2, 12, 13, 16]` minus `zero_load_ref_median`* -- a zero-load tare, in float64, cast to
   float32 once. Without the tare the left-wrist `fy` column is off by 10.7 N. `eval40k.build_proprio14`
   does exactly that and calls the worker's own function for the first eight columns;
   `policy40k.ForcePolicy(proprio_fn=...)` is the one call site that uses it, and
   `eval40k.build_policy` -- the only place a real policy is constructed -- always passes it, so the
   batch and the smoke cannot diverge. `proprio14_test.py` proves the arithmetic and the routing on
   any machine with numpy alone.
6. **Device equality was compared as text.** `str(tensor.device) == str(policy.device)` is False for
   a model and a policy on the same card, because `cuda` and `cuda:0` are two spellings of one device.
   The smoke now resolves both through `torch.device` (`same_device()`) and the flow test pins the
   whole case table -- `cpu` vs `cuda:N` false, `cuda:N` vs `cuda:N+1` false, `cuda` vs `cuda:N` true
   when CUDA is available -- and forbids the `str()` spelling from coming back.
7. **The output shape was assumed, not read.** The checks demanded `(1, 32, 7)` for `action` and
   `(1, 32, 6)` for `force_future_newton`. `jointwam.py` builds that dict with
   `"action": latents_action[0]` and `fastwam.py` with `"force_future_newton": raw[0]`: the batch
   dimension is removed *inside* the model, so the contract is `[T, D]`. The corrected check is not
   merely looser -- it drives the real recorders with a 2-D stub (which must report `[32, 7]` and
   `[32, 6]`) and a 3-D stub (which must report `['[1, 32, 7]']`), and it proves the expectation
   against the two model source files, so a genuine shape change still fails.
8. **The scene fingerprint was taken after the robot moved.** Two rollouts of the frozen,
   un-randomised layout produced two different hashes, which reads exactly like the randomisation knobs
   being alive. The hash is now computed at reset (`scene_at_reset[seed]`), and the end-of-episode pose
   is kept under its own field name; the flow test asserts the source order of the two calls rather
   than trusting a value.
9. **An aggregate `min`/`max` over a 14-wide proprio cannot answer "is this the normaliser or the
   physics?"** The first real rollout reported `proprio_min_all -5.0 / proprio_max_all 5.0`, i.e. the
   clamp was reached somewhere. That number says nothing about *which* column, how often, or how far
   outside the recorded band the raw value went. It is now a per-column table: min, max, the frame each
   happened at, the raw newtons or metres behind it, which columns were ever clamped and how many calls
   were clamped. Whether those excursions are physical is then a reading of the table, not a boolean.
10. **Wall time was totalled, not split, so one starved episode set the projection.** Seed 11 took
    195.91 s and seed 12 took 1673.41 s for the same 601 frames and the same mean inference cost
    (1.435 s vs 1.383 s); their mean produced a 160-episode projection of 41.58 h, which is a CPU
    quota artefact presented as a GPU estimate. The smoke and the batch now meter the simulator inside
    `env.step`, split each episode into simulation / inference / a named remainder, read the container's
    `cpu.stat` usage and throttling counters as deltas (never `/proc/loadavg`, which is the host's and
    means nothing inside a cgroup), and project from the median as well as the mean.


Adjacent, same class: **the default `--stats` pointed at the released LIBERO stats file**, which
describes an 8-wide state the model never saw. Both drivers now default to
`/workspace/a3_data/a3_fw_v1g/dataset_stats.json` -- the file `run40k/config.yaml` names as
`pretrained_norm_stats` -- and `check_stats_width()` fails loudly if that file's state columns are
not 14 wide.

## Normalisation semantics that look like a bug and are not

The `min/max` mode of `SingleFieldLinearNormalizer` maps to **[-1, 1]** (`output_min=-1`,
`output_max=1`), not [0, 1], and clamps the result at **±5**. For the six force columns
`build_v1_dataset.py:144-145` records the band as `+/- residual_std` *on purpose*, so min/max mode
computes `residual / std` for them with offsets of exactly 0.0 and scale of exactly `1/std`. The
consequence, measured on the corpus: the force columns legitimately leave [-1, 1] (min
`[-1.919, -6.0245, -7.475, 0, 0, -0.873]`, max `[4.8549, 2.1614, 2.6685, 2.2598, 2.2146, 2.1265]`)
and **620 of 42,681 rows (1.45 %) saturate the ±5 clamp** on columns 1 and 2.

Do not "fix" that. A first pass of this harness asserted the proprio band was [0, 1] and flagged a
double normalisation that was not happening; the assertion was wrong, not the data. What is asserted
now: the eight base columns inside [-1, 1] and the six force columns equal to the analytic
`residual / std`, which is what catches a genuine double application (it would put base columns well
outside ±1). `force_history` is a different input and stays **raw newtons** -- the mean/std
calibration is applied inside the model (`force_condition.normalize`), so taring or scaling it on the
eval side would be the double application.

## How to run the CPU checks anywhere, and how not to be fooled by them

```sh
python proprio14_test.py     # 17 checks, numpy only, no torch needed
python flow_test.py          # 35 checks, needs torch (import only), pod or any CPU box
python flow_test.py --json out/flow_test.json
```

`flow_test_local_shim.py` is for a box with **no** torch: it installs a numpy stand-in for the three
names the fixtures touch (`from_numpy`, `float32`, `no_grad`) and runs the same 35 checks. It is a
*wiring* pre-check -- it caught `new_policy()` building a `ForcePolicy` with `object.__new__` and
therefore missing the new `proprio_fn` attribute, which would only have surfaced on the pod. It
cannot load a weight, a corpus or a simulator, so its banner says so and its output is never the
acceptance record.

The recording half has its own CPU gates, and `RUN_RECORD_SEEDS.sh` runs all four before it is allowed to
touch the GPU:

```sh
python sensor_identity40k.py            # 39 checks; 40 with --require-right-press. exit 3 = legs A+B only
python sensor_identity_tamper40k.py     # proves the gate above bites: 7 bad XMLs refused, 4 press files graded
python panelvideo_selftest40k.py        # 52 checks on the composed layout and its per-frame sync
python panelsink_selftest40k.py         # 30 checks on PanelSink.add_frame, the production entry point
```

What each one claims, and which leg of evidence it is, is in `ROOT_CAUSE_40K_20261112.md` section 1.5 and
section 7.1; the layout sample those gates describe is in `../out40k/panel_layout_sample/`.

## The gate that still blocks the real batch

`eval40k.py --steps N` on the real backend stops with `no_merged_artifact`/`merged artifact ...
does not exist`: the deployment artifact is `base + v3 adapter` merged by
`fw/scripts/export_merged_v3.py`, and a merged file is ~11.1 GB while `/workspace` had 44 GB free
during training. The plan after step 40,000 is therefore per node: merge → evaluate → delete the
merged file (the light node and the adapter stay on disk and are already backed up to Hugging
Face), never all eight at once. That plan is now a command, and the interlock above is what keeps it
from being run early:

```sh
cd /workspace/_v1_stage/dev/eval40k
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python batch_eval.py --dry-run --diversity
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python batch_eval.py --diversity      # after step 40,000
```

Two facts from the pod on 2026-11-12 that change how that plan is executed, both of them about the
same 11 GB: `--dry-run` reports `blockers: []` with **27.44 GiB** free, which clears the 1.5x merge
margin, and `dev/eval40k/merged/` already holds `step_040000_merged.pt`. The loop treats a second
node's merged artifact on disk as a stop condition, not a warning, so a run that starts at step 5,000
aborts on its first iteration over the 40,000 file -- reordering `--steps` does not help, because the
check is "any other node's file", not "the next node's file". The order that neither deletes nor
re-merges anything is four moves on one volume: rename `step_040000_merged.pt` out of `merged/`
(same filesystem, instant, reversible, nothing lost), run the seven nodes 5k-35k, where each merged
file is removed only after every split of that node has a complete `summary.json`, rename the 40,000
file back, and finish with `--steps 40000 --keep-merged` so the artifact the smoke was run on is the
artifact the batch is scored on and it stays on disk (its `publish_record_step040000.json` is already
in the work directory, which is what makes it reuse rather than re-merge). In no variant does the
driver touch a light checkpoint, an adapter, a training artifact or an HF backup.


## Status at 2026-11-12, after round 8, so the next reader does not have to rediscover it

Training is finished: 40,000 steps, eight light nodes kept, `step_040000` uploaded to Hugging Face,
re-downloaded and reloaded, and the segment-8 boundary report checked. `dev/eval40k/merged/
step_040000_merged.pt` (11,126,493,579 B, first-MiB sha16 `f9fe3c9407c89169`) exists on the pod and
was produced before the interface fix, so it is still valid: the fix changed only what the harness
*feeds* the model, never the weights.

The file bridge that the previous revision of this section said was blocked is working again. Eight
rounds of writes have gone through it, each one gated by `dev/_probe_1112/install_gate.py` (staged
hash must equal the recorded local hash, pre-image read back from `pre/`, one `os.replace`, hash and
`py_compile` on the pod, unified diff archived, one `post_manifest_round<N>.json` per round and none
of them rewritten). The live identities on the pod now are:

| file | bytes | sha256 | written by |
|---|---|---|---|
| `eval40k.py` | 61,094 | `fb723254` | round 7 |
| `policy40k.py` | 10,865 | `f7d8ec3c` | not named in any round manifest |
| `smoke40k.py` | 92,302 | `d69a038d` | round 8 |
| `flow_test.py` | 100,599 | `215458cc` | round 8 |
| `proprio14_test.py` | 18,628 | `6b0fa29c` | not named in any round manifest |
| `batch_eval.py` | 21,698 | `3d6292ab` | not named in any round manifest |
| `report40k.py` | 14,145 | `5b78b5d4` | not named in any round manifest |

Every hash in that table was read back off the pod -- `sha256sum` over all seven files plus
`_probe_1112/r3_plan_compare.py` at 2026-11-12T06:05:45Z, output kept in
`dev/out40k/podrun/r8_hashes.txt` -- and matches the local file byte for byte. For the four files
with no manifest row, the honest statement is "the pod holds exactly this, and this directory cannot
say which write put it there" -- not "unchanged".

**Corrected 2026-11-12 late (panel-recording round).** That table is a snapshot of 06:05:45Z and three of
its rows are now behind the pod: round 9 installed `report40k.py` and round 10 installed `batch_eval.py`
and `flow_test.py`, so the pod's live bytes for those three are `b48b7a33`, `7dc2a0c5` and `ee45c272` --
the same values `RUN_BATCH_STEP6.sh` pins and the same ones its step-0 gate accepted before the
160-episode batch started. Read back off the pod? No: the channel answers 403, so those three are
"inferred from the round specs plus the fact that the batch ran", not "verified now".

And the recording half of this toolchain has **never been on the pod at all**. None of the ten install
specs names `record40k.py`, `panelvideo40k.py`, `recorder_selftest40k.py`, `panelvideo_selftest40k.py`,
`panelsink_selftest40k.py`, `sensor_identity40k.py`, `sensor_identity_tamper40k.py`, `panel_sample40k.py`
or `RUN_RECORD_SEEDS.sh`; their local identities, with the ever-pushed flag, are recorded in
`dev/eval40k/_panel_hashes_1112.json`. Until those nine go through the bridge, `RUN_RECORD_SEEDS.sh`
cannot run there and no composed panel video of a real rollout exists -- the only watchable artifact is
`dev/out40k/panel_layout_sample/`, which is synthetic by construction and says so on the panel.
`force_stream.py` and `rollout40k.py` are a third case: they may already be on the pod, but with the
channel answering 403 their remote sha cannot be read back, so the manifest marks both `unknown` and the
first thing to do when the bridge returns is to read them before deciding whether to push.

What **has** happened since: six live smoke attempts against the real `step_040000`, and the fifth
(`out/smoke_20261112_r7`) ran to completion — two full 601-frame MuJoCo rollouts on the fixed cookie
scene, `checks_passed` 45 of 47, `RC=1`. All four interface defects that section B of
`dev/_diffrecord/20261112_iface_fix/DEBUG_LOG.md` lists are confirmed fixed on real rollouts: one
reset scene key for both seeds, the `[32, 7]` / `[32, 6]` / `[1, 16, 6]` / width-14 contract read off
the model's own tensors, and the per-column proprio exceedance table with **0 unexplained**
excursions in both episodes. The two failures were one check printed twice, and round 8 re-scoped it
(B5 in that log): staying inside the training band for 600 steps is a property of the policy, not of
the harness, so the interface check now asserts only what an interface can decide and a new
per-episode check asserts the attribution instead -- and reports the out-of-band fact rather than
burying it in a pass/fail.

That fact is worth reading on its own, because it is a model result and not a tooling one. On the
fixed scene the 40k checkpoint starts at raw EEF z 0.920 -- exactly the recorded band edge -- dips to
0.782 at frames 40-50 where the first contact shows up, and then rises without turning back to 1.80 m
by frame 590, 5.65 band widths past the recorded EEF-z maximum, with one force channel pinned near
+20.5 N from frame 400 on and 56 of the episode's 60 replans hitting the ±5 clamp. Seed 12 leaves the
band by colliding instead: raw contact forces of −321.9 N and +373.0 N at frame 580 and a
denormalised action absmax of 8.67. Both episodes end `terminated: false,
truncated: false` with 0 cookies in the target and no NaN: **0 of 2**. And the cost is now measured
rather than guessed -- seed 11 201.41 s wall = 111.5 s simulation + 89.8 s inference, seed 12
1676.89 s wall = 1593.0 s simulation + 83.8 s inference, one simulator step alone taking 10.9 s --
giving `projected_160_episodes_h` 41.77 by mean, 41.74 by median. The 160-episode batch is a ~42 h
GPU commitment, and it is simulator-bound, not inference-bound.

What still has **not** happened: the 8-node × 20-seed × 160-episode formal evaluation and the fixed
scene sampling report. The gate stands exactly as written: only after a real-model smoke with zero
failed checks may step 6 start; round 8's smoke is what has to clear it. Before it can run at all,
the merged-artifact/disk interlock in the section above has to be honoured -- 27.44 GiB free, one
11.13 GB merged file present, seven nodes still unmerged, and nothing in the plan deletes a light
checkpoint, an adapter, a training artifact or an HF backup.

Two environment gotchas that cost time once each: `MUJOCO_GL=egl` must be exported before the A3 env
is imported on this headless box, and `grep` on the pod's shell needs `-E` -- `a\|b` in a BRE
silently matches nothing here, which is a way worse failure than an error message. A third, for the
next person driving the pod from a browser: the JupyterLab API is served under the instance prefix,
so a root-relative `/api/contents` returns an nginx 404 that looks like a dead bridge -- derive the
prefix from the page URL. And the kernel this session runs in has polluted globals (`print` is
`None`), so write results to a file and read them back rather than trusting a stdout capture.
