p = "/workspace/fasterwam_a3_bridge/bridge/_next/dagger_to_dataset.py"
s = open(p).read()
a = '''    ap.add_argument("--root", default="/workspace/fasterwam_a3_bridge/bridge/outputs/dagger")'''
b = '''    ap.add_argument("--root", nargs="+", default=["/workspace/fasterwam_a3_bridge/bridge/outputs/dagger"])'''
c = '''        take_dir = Path(args.root) / f"seed{seed}_s{take}"'''
d = '''        td = [Path(r) / f"seed{seed}_s{take}" for r in args.root]
        take_dir = next((x for x in td if x.exists()), None)
        if take_dir is None:
            raise SystemExit(f"seed{seed}_s{take}: not found under {args.root}")'''
assert a in s and c in s
s = s.replace(a, b).replace(c, d)
open(p, "w").write(s)
print("PATCHED")
