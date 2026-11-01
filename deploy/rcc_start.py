"""Create the Radeon Cloud notebook once credits allow it, then publish deploy/pod.env.

Runs forever, one attempt a minute. Create is submitted with a stable
idempotency key, so retrying after a lost response cannot open a second paid
instance.  Once the instance reports ssh_ready the SSH line is written to
deploy/pod.env and the script exits; deploy/wait_and_launch.sh notices the file
within a minute and starts the retry/bootstrap chain.

    py -3.13 deploy/rcc_start.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "outputs", "rcc_start.log")
POD_ENV = os.path.join(ROOT, "deploy", "pod.env")

ALLOW_CREATE = os.environ.get("ALLOW_CREATE") == "1"
INTERVAL = int(os.environ.get("INTERVAL", "60"))
ENV_ID = os.environ.get("ENV_ID", "production-1")
POD_TYPE = os.environ.get("POD_TYPE", "one-click")
TEMPLATE = os.environ.get("TEMPLATE", "resource-16c55g1u")
IDEM = os.environ.get("IDEM", "a3-cookie-r27-20261030")

RCC = shutil.which("rcc") or r"D:\APP\Python313\Scripts\rcc.EXE"


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def rcc(*args: str) -> tuple[dict | None, str]:
    cmd = [RCC, "--output", "json", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=180)
    doc = None
    text = p.stdout.strip()
    if text:
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            pass
    if doc is None:
        # Event lines can precede the result document; take whichever brace
        # position spans the most characters, not the innermost object.
        decoder = json.JSONDecoder()
        best = -1
        for i, ch in enumerate(p.stdout):
            if ch != "{":
                continue
            try:
                obj, end = decoder.raw_decode(p.stdout[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and end > best:
                best, doc = end, obj
    return doc, (p.stderr or "").strip()


def publish(ssh_line: str) -> None:
    tmp = POD_ENV + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write('POD_SSH="%s"\n' % ssh_line)
    os.replace(tmp, POD_ENV)
    log(f"wrote {POD_ENV}: {ssh_line}")


def main() -> int:
    if os.path.exists(POD_ENV) and "1.2.3.4" not in open(POD_ENV, encoding="utf-8").read():
        log("deploy/pod.env already names an endpoint; nothing to do")
        return 0

    log(f"rcc_start: env={ENV_ID} pod={POD_TYPE} template={TEMPLATE} every {INTERVAL}s")
    attempt = 0
    while True:
        attempt += 1
        doc, err = rcc("instance", "status")
        data = (doc or {}).get("data") or {}
        status = data.get("status")

        if status == "not_found":
            if not ALLOW_CREATE:
                log(f"attempt {attempt}: no instance; creation disabled "
                    f"(set ALLOW_CREATE=1 to submit the paid create request)")
                return 2
            doc, err = rcc("instance", "create", "--env", ENV_ID,
                           "--pod-type", POD_TYPE, "--resource-template", TEMPLATE,
                           "--idempotency-key", IDEM, "--no-wait")
            error = (doc or {}).get("error") or {}
            if error:
                log(f"attempt {attempt}: create rejected [{error.get('code')}] {error.get('message')}")
            elif doc and doc.get("status") == "error":
                log(f"attempt {attempt}: create error {doc.get('message')}")
            else:
                log(f"attempt {attempt}: create accepted, idempotency={IDEM}")
        elif data.get("ssh_ready") and data.get("ssh_host"):
            host = data["ssh_host"]
            port = data.get("ssh_port")
            user = data.get("ssh_username") or "root"
            publish(f"ssh -p {port} {user}@{host}" if port else f"ssh {user}@{host}")
            return 0
        else:
            log(f"attempt {attempt}: status={status} ssh_stage={data.get('ssh_stage')} "
                f"detail={data.get('detail') or data.get('message')}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
