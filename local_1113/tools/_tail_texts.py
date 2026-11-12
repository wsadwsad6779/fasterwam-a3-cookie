import json, sys

path = r"C:\Users\ASUS\.qoder-cn\projects\C--Users-ASUS-Documents-Qoder-2026-10-31-b462d38d\b462d38d-2bab-475d-a8b3-0dbd231385f9.jsonl"
texts = []
user_texts = []
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get("type")
        m = o.get("message") or {}
        c = m.get("content")
        if isinstance(c, str):
            if t == "user":
                user_texts.append((i, o.get("timestamp"), c))
            continue
        if not isinstance(c, list):
            continue
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip():
                if t == "assistant":
                    texts.append((i, o.get("timestamp"), b["text"]))
                elif t == "user":
                    user_texts.append((i, o.get("timestamp"), b["text"]))

n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
which = sys.argv[2] if len(sys.argv) > 2 else "a"
src = texts if which == "a" else user_texts
for rec in src[-n:]:
    print("=" * 90)
    print(rec[0], rec[1])
    print(rec[2][:2500])
