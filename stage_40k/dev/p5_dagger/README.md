# p5_dagger — the corrective-data collector, and what "no fabricated supervision" costs in code

P5 is the one part of the force plan that cannot be trained yet: the model's own failures have no
labels. This directory therefore builds only the *recorder*, and the recorder's job is to make the
difference between "the policy acted" and "the expert acted" impossible to lose later. Nothing here
has been run against a real checkpoint; every number below is from the CPU self-test on the fixture
world.

```sh
cd /workspace/_v1_stage/dev/p5_dagger
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python dagger_selftest.py --json out/dagger_selftest.json
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python dagger_collect.py --backend fixture --policy stub --dry-run
```

| file | what it is |
| --- | --- |
| `dagger5.py` | the contract (`p5-dagger-v1`), the row table, the rule controller, the integrity audit, the npz writer |
| `dagger_collect.py` | the closed loop: one row per *executed* frame, takeover rules, optional P4 `ShadowMonitor` |
| `dagger_selftest.py` | 96 hand-computed assertions about what the table must contain |

## Measured (pod, 2026-11-10 UTC, CPU only — no model, no simulator, no GPU)

`dagger_selftest.py --json out/dagger_selftest.json` → `n_checks 96, n_passed 96, failed []`.

The checks that are worth naming, because each one is a way a recorder can lie:

* **One row per executed step, not per chunk.** 24 steps ⇒ 24 action rows and 25 state rows
  (`steps + 1`, the terminal observation included); the policy is asked for a chunk exactly at the
  replan boundaries `[0, 10, 20]`, and every frame is still observed 1 + 24 pushes.
* **`policy_action` is the array the policy says it emitted, `executed_action` is what the env
  reports back.** Both are compared against the source that produced them, per frame. The r3
  column name `action14` is kept, but it is pointed at the *executed* action — never at the
  expert's intent — so an old converter cannot accidentally read a wish as a record.
* **`force6` is exactly the frozen channels of `force18`** (`[0,1,2,12,13,16]`), and `force18` at
  row *f* is the observation that the action at row *f* was taken on. Two clocks are stored and must
  not be mixed: `ts_mono` is seconds since episode start, `ts_wall` is absolute epoch time,
  non-decreasing; a wall-clock value in the monotonic column is caught as a problem.
* **A stateful expert is never peeked.** If a frame was policy-controlled, the collector refuses to
  advance the expert to "borrow" an action for it; the row says `label_kind=none` and the refusal is
  *counted in the report*, not only written in the rows. `--peek-expert` exists so that
  counterfactual labels can be recorded honestly, and then the rows are `label_kind=counterfactual`
  — which `labels_for_training()` deliberately does **not** return. A takeover is still required.
* **Exporting labels from an expert-free episode raises** rather than falling back to the policy's
  own actions, which would be self-supervision with a label-shaped name. With a real takeover
  section, `labels_for_training()` returns exactly the 16 takeover frames, each with a finite expert
  action equal to what was executed, and no policy-controlled frame leaks in.
* **The integrity audit is a positive test.** A truthful table verifies with no problems; and each of
  these is caught: a policy row without a policy action, an expert label claimed on a
  policy-controlled frame, an expert row whose executed action is not the expert's, an
  expert-controlled frame that forgot to set `intervention`, a clock that goes backwards.
* **Shadow integration keeps observation and action separate.** A `proceed` verdict ⇒ 24 policy rows;
  a `stop` verdict hands control over from the *next* frame and the reason column names the verdict;
  under all of that the P4 monitor's own counters stay at `actions_mutated 0`, and the takeover is
  accounted by the collector, not by the monitor.

## What is not claimed

No corrective episode has been collected yet: there is no trained-policy failure corpus to label, and
the expert here is the scripted one. The rule set, thresholds and the `intervention` column are the
placeholders that a real collection run will fill; the part that is fixed and tested is the schema
and the refusal behaviour, so that when the labels do arrive they cannot be quietly fabricated.
