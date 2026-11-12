import json, sys

path = r"C:\Users\ASUS\.qoder-cn\projects\C--Users-ASUS-Documents-Qoder-2026-10-31-b462d38d\b462d38d-2bab-475d-a8b3-0dbd231385f9.jsonl"
keep = []
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
        if t not in ("user", "assistant"):
            continue
        m = o.get("message") or {}
        c = m.get("content")
        parts = []
        if isinstance(c, str):
            parts.append("TEXT: " + c[:900])
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    parts.append("TEXT: " + b.get("text", "")[:900])
                elif bt == "tool_use":
                    parts.append("TOOL_USE " + str(b.get("name")) + " " + json.dumps(b.get("input"), ensure_ascii=False)[:400])
                elif bt == "tool_result":
                    r = b.get("content")
                    s = r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
                    parts.append("TOOL_RES " + s[:300])
        if not parts:
            continue
        keep.append((i, o.get("timestamp"), t, "\n".join(parts)))

print("total kept", len(keep))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
for rec in keep[-n:]:
    print("=" * 90)
    print(rec[0], rec[1], rec[2])
    print(rec[3][:1600])
