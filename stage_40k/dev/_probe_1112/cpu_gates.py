"""The pod's CPU verification round after the evaluation-interface fix (2026-11-12).

Three things run here, in order, and none of them opens a checkpoint, holds a CUDA context or
writes anywhere except this directory:

  * `proprio14_test.py` -- the unit test for the 14-wide proprio the eval toolchain builds;
  * `flow_test.py` -- the whole CPU flow test, now with the text-conditioning block that needs the
    pod's real `torch` (the bit-exactness checks cannot register on the laptop);
  * `smoke40k.corpus_proprio14_proof()` -- the 42,681-row equivalence proof, called through the
    driver's own function so this pre-check and the smoke run exercise one implementation.

The equivalence and normalisation assertions below are copied from `smoke40k.main()` at the lines
that gate the real run, not re-derived: running them here first is an early look at the same
conditions, and the smoke run repeats them before a model is loaded.
"""
import json
import subprocess
import sys
from pathlib import Path

D = Path("/workspace/_v1_stage/dev/eval40k")
out = {"python": sys.version.split()[0], "scripts": {}}

for script in ["proprio14_test.py", "flow_test.py"]:
    r = subprocess.run([sys.executable, script], cwd=str(D),
                       capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    out["scripts"][script] = {
        "rc": r.returncode,
        "n_pass_lines": sum(1 for ln in lines if "PASS" in ln),
        "n_fail_lines": sum(1 for ln in lines if "FAIL" in ln),
        "tail": lines[-16:],
        "err_tail": [ln for ln in r.stderr.splitlines() if ln.strip()][-8:]}

sys.path.insert(0, str(D))
import eval40k as E                # noqa: E402
import smoke40k as SM              # noqa: E402

proof = SM.corpus_proprio14_proof()
gate = {
    "rows": proof["rows"] == 42681,
    "gathered_channels_equal_force6": proof["gathered_channels_equal_force6"],
    "gathered_channels_bit_equal_frac_is_one": proof["gathered_channels_bit_equal_frac"] == 1.0,
    "select_channels_is_a_single_vector_function":
        proof["select_channels_is_a_single_vector_function"],
    "select_channels_rowwise_equals_vectorised_gather":
        proof["select_channels_rowwise_equals_vectorised_gather"],
    "rows_probed_rowwise_at_least_512": proof["rows_probed_rowwise"] >= 512,
    "state_force_columns_equal_tared_channels": proof["state_force_columns_equal_tared_channels"],
    "state_force_columns_bit_equal_frac_is_one":
        proof["state_force_columns_bit_equal_frac"] == 1.0,
    "state_base_columns_equal_stored_8_wide_proprio":
        proof["state_base_columns_equal_stored_8_wide_proprio"],
    "v1g_proprio_width_is_PROPRIO_DIM": proof["v1g_proprio_shape"][1] == E.PROPRIO_DIM,
    "base8_width_matches": proof["base8_width"] == E.PROPRIO_BASE_DIM,
    "calib_version_matches": proof["calib_version"] == E.CALIB_VERSION,
    "stats_state_width_is_PROPRIO_DIM": proof["stats_state_width"] == E.PROPRIO_DIM,
    "stats_force_band_is_symmetric": proof["stats_force_band_is_symmetric"],
    "force_halfwidth_equals_calib_std": proof["force_halfwidth_equals_calib_std"],
}
out["corpus_proof"] = {
    "gate_passed": all(gate.values()),
    "gate": gate,
    "values": {k: proof[k] for k in ["rows", "v1g_proprio_shape", "force_arrays_shape",
                                    "gathered_channels_bit_equal_frac", "rows_probed_rowwise",
                                    "state_force_columns_max_abs_diff",
                                    "state_force_columns_bit_equal_frac",
                                    "state_base_columns_max_abs_diff", "base8_width",
                                    "calib_version", "stats_state_width",
                                    "force_halfwidth_minus_calib_std_max", "stats_source"]},
    "force6_zeroed_is_not_the_tare": {
        "max_abs_diff_vs_tared": proof["force6_zeroed_vs_tared_max_abs_diff"],
        "per_channel_max": proof["force6_zeroed_vs_tared_per_col_max_abs_diff"],
        "rows_differing": proof["force6_zeroed_vs_tared_rows_differing"],
        "reported_not_asserted": True},
    "stats_path_named_by_this_run": str(E.V1G_STATS),
    "calib_path": str(E.CALIB_PATH),
    "v1g_arrays": str(E.V1G_ARRAYS),
    "force_arrays": str(E.FORCE_ARRAYS)}
out["scripts_failed"] = [k for k, v in out["scripts"].items() if v["rc"] != 0]
out["all_passed"] = (not out["scripts_failed"]) and out["corpus_proof"]["gate_passed"]
print(json.dumps(out, indent=1, sort_keys=True, default=str))
