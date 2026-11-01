cat >> /workspace/_r5_stage/single_lowforce_probe.py <<'PY'


def main():
    import shutil

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[6001])
    ap.add_argument("--max-steps", type=int, default=3200)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--out", default="/workspace/a3_eval/pilot_single_lowforce")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.glob("*")):
        raise SystemExit("refusing to write into a non-empty directory: " + str(out))
    free = shutil.disk_usage(str(out.parent if out.parent.exists() else "/workspace")).free
    if free < 1_000_000_000:
        raise SystemExit("only %d bytes free, refusing to start a recording run" % free)
    out.mkdir(parents=True, exist_ok=True)
    print("out=%s free_bytes=%d" % (out, free), flush=True)

    rows = []
    for seed in args.seeds:
        row = run_seed(seed, args.max_steps, args.noise, out)
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
        (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
PY
python3 -m py_compile /workspace/_r5_stage/single_lowforce_probe.py && echo COMPILE_OK
wc -l /workspace/_r5_stage/single_lowforce_probe.py
sha256sum /workspace/_r5_stage/single_lowforce_probe.py
echo DONE