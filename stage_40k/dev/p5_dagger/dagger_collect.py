"""Drive the P5 corrective-data collector against the real A3 env and the real scripted expert.

`dagger5.py` holds the recording rules; this file is the wiring, and it keeps three gates that
matter more than the wiring itself:

* it refuses to start a collection run that has no takeover rule. An episode where the model
  never leaves the wheel has zero corrective labels in it, and a collector that can produce an
  empty "dataset" without saying so is how fabricated supervision gets made;
* it declares the expert's peek-ability from the provider itself (`stateless_peek`), so a
  stateful `A3SameColumnBatchExpert` can never be quietly advanced by queries whose actions are
  thrown away;
* it will not occupy the training GPU while a 40k segment is still writing. The evaluation and
  collection work is scheduled after the eight nodes, per the standing decision, and `--dry-run`
  plus `--backend fixture` are what can be executed today.

    cd /workspace/_v1_stage/dev/p5_dagger
    PYTHONDONTWRITEBYTECODE=1 python dagger_collect.py --dry-run
    PYTHONDONTWRITEBYTECODE=1 python dagger_collect.py --backend fixture --policy stub \
        --takeover manual:8 --episodes 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval40k"))
sys.path.insert(0, str(HERE.parent / "p4_shadow"))
sys.path.insert(0, str(STAGE / "fw" / "scripts"))
sys.path.insert(0, str(STAGE / "fw" / "src"))

from dagger5 import (CONTRACT_VERSION, Controller, DaggerIntegrityError,  # noqa: E402
                    ForceZAbove, ManualFrames, NoProgress, NonfiniteState, ShadowVerdict,
                    labels_for_training, run_dagger_episode)
from force_stream import (CHANNEL_NAMES, HIST_LEN, check_calib,  # noqa: E402
                          select_channels)

RUN = STAGE / "run40k"
LIGHT = RUN / "checkpoints" / "light"
CALIB_PATH = Path("/workspace/a3_data/a3_fw_force/force_calib_v1.json")
BRIDGE_DIRS = (str(Path("/workspace/fasterwam_a3_bridge/bridge")),
               str(Path("/workspace/fasterwam_a3_bridge/bridge/_next")),
               str(Path("/workspace/a3_dual_arm_sim/src")))
ACTION_DIM = 14
DEFAULT_SEEDS = [11, 21, 22, 23, 24, 31, 32, 33]
STATS_DEFAULT = Path("/workspace/FasterWAM/checkpoints/fasterwam_release/"
                     "libero/dataset_stats.json")
INSTRUCTION_DEFAULT = "Pick up the cookie and place it in the box."


def read_json(path):
    return json.loads(Path(path).read_text())


def parse_takeover(specs, calib, seed):
    """`--takeover` strings -> rule objects, with the thresholds kept in the report.

    Grammar: `manual:FRAME[,FRAME...]`, `force_z:Z`, `no_progress:FRAMES`, `nonfinite[:FIELD]`,
    `shadow:VERDICT[,VERDICT...]`. Anything else is an error, not a silently ignored token.
    """
    rules = []
    for spec in specs or []:
        parts = str(spec).split(":")
        kind = parts[0]
        arg = parts[1] if len(parts) > 1 else None
        if kind == "manual":
            frames = [int(x) for x in arg.split(",")] if arg else []
            rules.append(ManualFrames([(int(seed), f) for f in frames]))
        elif kind == "force_z":
            rules.append(ForceZAbove(np.asarray(calib["mean"], dtype=np.float64),
                                     np.asarray(calib["std"], dtype=np.float64),
                                     z=float(arg) if arg else 6.0))
        elif kind == "no_progress":
            rules.append(NoProgress(patience=int(arg) if arg else 120))
        elif kind == "nonfinite":
            rules.append(NonfiniteState(arg or "observation.eef_pose"))
        elif kind == "shadow":
            rules.append(ShadowVerdict(verdicts=tuple(
                v.strip() for v in (arg or "stop").split(","))))
        else:
            raise SystemExit(f"unknown --takeover rule {spec!r}; expected one of "
                             "manual, force_z, no_progress, nonfinite, shadow")
    return rules


def training_is_live():
    """True when a segment log has been written in the last two minutes.

    Used only to keep a GPU collection run from starting on top of the 40k training that owns the
    card. It is a heuristic about file mtime, not a claim about the process table, so it errs
    toward "live" and the operator has to override it explicitly.
    """
    logs = sorted(RUN.glob("logs/seg*.log"))
    if not logs:
        return False
    return (time.time() - logs[-1].stat().st_mtime) < 120.0


def build_expert(args, env):
    for p in BRIDGE_DIRS:
        if p not in sys.path:
            sys.path.insert(0, p)
    from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert
    expert = A3SameColumnBatchExpert(env)
    if not hasattr(expert, "stateless_peek"):
        expert.stateless_peek = False          # phase machine: peeking would corrupt it
    return expert, "a3_same_column_batch"


def build_stub_expert(args):
    class FixtureExpert:
        """Deterministic scripted expert for the CPU path, with the same 14-wide action."""

        stateless_peek = False

        def __init__(self):
            self.calls = 0

        def reset(self):
            self.calls = 0

        def act(self, obs):
            self.calls += 1
            f = select_channels(np.asarray(obs["observation.force"], dtype=np.float64))
            return np.concatenate([np.zeros(7), (f[:3] * 0.001), np.zeros(3), [0.0]])

    return FixtureExpert(), "fixture"


def plan(args, calib, rules):
    return {
        "contract": CONTRACT_VERSION,
        "backend": args.backend,
        "policy": args.policy,
        "expert": args.expert,
        "seeds": args.seeds[:8] + (["..."] if len(args.seeds) > 8 else []),
        "episodes_per_seed": args.episodes,
        "replan": args.replan,
        "max_steps": args.max_steps,
        "controller": {"sticky": args.sticky, "release_after": args.release_after,
                       "rules": [r.describe() for r in rules]},
        "peek_expert": args.peek_expert,
        "shadow": args.shadow,
        "out": str(args.out),
        "calib": {"version": calib["version"], "channels": calib["channels"],
                  "mean": calib["mean"], "std": calib["std"]},
        "bytes_per_episode_estimate": int(args.max_steps * (ACTION_DIM * 5 + 18 + 6 + 7) * 8
                                         + 4096),
        "gpu_gate": {"training_log_is_recent": training_is_live(),
                     "override": args.allow_during_training},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["fixture", "env"], default="fixture")
    ap.add_argument("--policy", choices=["stub", "node"], default="stub")
    ap.add_argument("--expert", choices=["same_column_batch", "fixture"],
                    default="fixture")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--replan", type=int, default=10)
    ap.add_argument("--takeover", action="append", default=[],
                    help="repeatable; manual:8 force_z:6.0 no_progress:120 nonfinite "
                         "shadow:stop")
    ap.add_argument("--sticky", choices=["end", "rule"], default="end")
    ap.add_argument("--release-after", type=int, default=None)
    ap.add_argument("--peek-expert", action="store_true",
                    help="ask a stateless expert at every visited frame for diagnostics")
    ap.add_argument("--shadow", action="store_true", help="attach a P4 ShadowMonitor")
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--calib", default=str(CALIB_PATH))
    ap.add_argument("--out", default=str(STAGE / "dagger40k"))
    ap.add_argument("--step", type=int, default=5000, help="node used with --policy node")
    ap.add_argument("--merged", default=None, help="explicit merged artifact for --policy node")
    ap.add_argument("--merged-dir", default=str(STAGE / "dev" / "eval40k" / "merged"),
                    help="directory holding step_%06d_merged.pt, same path eval40k uses")
    ap.add_argument("--config", default=None, help="training config for --policy node; "
                    "defaults to eval40k's frozen 40k config")
    ap.add_argument("--stats", default=None, help="dataset stats json, same file eval40k uses")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--gripper-mode", default="pm1", choices=["pm1", "libero01"])
    ap.add_argument("--cpu", action="store_true",
                    help="load the node on CPU: a plumbing smoke, never a result")
    ap.add_argument("--position-noise", type=float, default=0.0)
    ap.add_argument("--yaw-noise", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-during-training", action="store_true",
                    help="required to use the GPU while a 40k segment is still writing")
    args = ap.parse_args()

    calib = check_calib(read_json(args.calib), where="P5 calibration")
    if args.backend == "env" and args.expert != "same_column_batch":
        raise SystemExit("--backend env should pair with the scripted "
                         "same_column_batch expert; a fixture expert against the real env "
                         "produces actions nobody designed")

    # one rule set per seed, because manual points are seed-scoped and the report must show it
    rules_probe = parse_takeover(args.takeover, calib, args.seeds[0])
    if args.dry_run:
        print(json.dumps(plan(args, calib, rules_probe), indent=2))
        return 0
    if not rules_probe:
        raise SystemExit(
            "refusing to collect: no --takeover rule given, so every frame would be "
            "policy-controlled and the run would export zero corrective labels. Name the rule "
            "that hands the wheel to the expert (manual:FRAME, force_z:Z, no_progress:N, "
            "shadow:stop) or run --dry-run.")
    if args.backend == "env" and args.policy == "node" and training_is_live() \
            and not args.allow_during_training:
        raise SystemExit(
            "a 40k segment log was written within the last two minutes; the training owns the "
            "GPU. Finish the eight nodes, or pass --allow-during-training deliberately.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"

    env_factory = None
    policy = None
    if args.backend == "env":
        if args.policy != "node":
            raise SystemExit("--backend env needs --policy node: a stub policy in the real env "
                             "would record takeover states nobody visited")
        # the attribute names match eval40k's CLI on purpose, so the same merged artifact,
        # calibration and stats feed the collector and the evaluator
        import eval40k as E
        args.config = args.config or str(E.DEFAULT_CFG)
        args.stats = args.stats or str(STATS_DEFAULT)
        args.instruction = args.instruction or INSTRUCTION_DEFAULT
        env_factory, W = E.build_env(args)
        policy = E.build_policy(args, None, (env_factory, W))

    n_written = 0
    total_labels = 0
    for seed in args.seeds:
        rules = parse_takeover(args.takeover, calib, seed)
        controller = Controller(rules=rules, sticky=args.sticky,
                                release_after=args.release_after)
        for ep in range(args.episodes):
            if args.backend == "fixture":
                from flow_test import FixtureEnv
                env = FixtureEnv(seed=seed, frames=args.max_steps + 1)
                reset = lambda e, s: e.reset(seed=s)  # noqa: E731
                expert, expert_name = build_stub_expert(args)
                from policy40k import StubPolicy
                pol = StubPolicy(action_horizon=args.horizon, pred_mode="offset")
            else:
                env = env_factory.make_env(position_noise_m=args.position_noise,
                                           yaw_noise_rad=args.yaw_noise, render_cameras=True)
                expert, expert_name = build_expert(args, env)
                pol = policy

                def reset(e, s, _f=env_factory):
                    obs, _info = _f.reset(e, s, randomize_cookies=True, randomize_boxes=False,
                                          batch_offset_m=0.0)
                    sanity = _f.scene_sanity(e)
                    if not sanity["ok"]:
                        raise RuntimeError(f"scene invalid at seed={s}: {sanity}")
                    return obs

            shadow = None
            if args.shadow:
                from shadow import ShadowMonitor
                shadow = ShadowMonitor(calib=calib, horizon=args.horizon, hist_len=HIST_LEN,
                                       channel_names=list(CHANNEL_NAMES))

            report, table, _measured = run_dagger_episode(
                env, pol, expert, reset, seed, out_dir=out_dir, max_steps=args.max_steps,
                replan=args.replan, controller=controller, shadow=shadow,
                peek_expert=args.peek_expert, episode_id=ep, action_dim=ACTION_DIM,
                extra_meta={"expert_kind": expert_name, "backend": args.backend})
            if not report["integrity_ok"]:
                raise DaggerIntegrityError(
                    f"seed={seed} ep={ep}: {report['integrity_problems'][:3]}")
            try:
                labels = labels_for_training(table)
                report["exported_label_rows"] = len(labels)
            except DaggerIntegrityError as exc:
                report["exported_label_rows"] = 0
                report["export_refused"] = str(exc)
            total_labels += report["exported_label_rows"]
            if shadow is not None:
                report["shadow"] = {k: v for k, v in shadow.finalize().items()
                                    if k in ("verdict_counts", "anomaly_counts",
                                             "actions_mutated", "alignment")}
            with summary_path.open("a") as fh:
                fh.write(json.dumps(report) + "\n")
            n_written += 1
            print(f"[p5] seed={seed} ep={ep} rows={report['rows']} "
                  f"interventions={report['intervention_rows']} "
                  f"labels={report['exported_label_rows']} "
                  f"discarded={report['discarded_policy_actions']}", flush=True)
            if hasattr(env, "close"):
                env.close()

    print(json.dumps({"episodes_written": n_written, "total_label_rows": total_labels,
                      "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
