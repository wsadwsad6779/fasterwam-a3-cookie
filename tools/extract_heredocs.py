import json, os

src = r"C:\Users\ASUS\.qoder-cn\projects\C--Users-ASUS-Documents-Qoder-2026-10-24-c3f102f0\c3f102f0-fb56-47da-90ae-6877ad58126e.jsonl"
outdir = r"C:\Users\ASUS\Documents\Qoder\2026-10-24\c3f102f0\heredoc_extracts"
os.makedirs(outdir, exist_ok=True)

targets = {
    "infer_a3_worker.py": "worker",
    "replay_diagnostic.py": "replay",
}

def walk(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)

count = 0
with open(src, encoding="utf-8") as f:
    for ln, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        for s in walk(obj):
            for fname, tag in targets.items():
                marker = "cat > /workspace/fasterwam_a3_bridge/" + fname
                if marker in s:
                    count += 1
                    outp = os.path.join(outdir, f"{tag}_{ln}_{count}.txt")
                    with open(outp, "w", encoding="utf-8") as o:
                        o.write(s)
print("saved", count, "extracts to", outdir)
