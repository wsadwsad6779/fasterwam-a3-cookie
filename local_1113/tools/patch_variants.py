import hashlib
import pathlib
import sys

P = pathlib.Path("/workspace/_r5_stage/replay_forces.py")
src = P.read_text(encoding="utf-8")
before = hashlib.sha256(src.encode()).hexdigest()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    if n != count:
        raise SystemExit("anchor seen %d times, expected %d: %r" % (n, count, old[:60]))
    src = src.replace(old, new, count)


sub(
    "def run_episode(seed, mode, max_steps, noise, out_dir, clamp_limit=0):\n",
    "def run_episode(seed, mode, max_steps, noise, out_dir, clamp_limit=0, variant=0):\n",
)

sub(
    "    expert.reset()\n",
    "    expert.variant_index = int(variant)\n    expert.reset()\n",
)

sub(
    "class SingleCookieLowForce(A3SameColumnBatchExpert):\n"
    "    batch_size = 1\n"
    "    num_batches = 1\n",
    "class SingleCookieLowForce(A3SameColumnBatchExpert):\n"
    "    batch_size = 1\n"
    "    num_batches = 1\n"
    "    variant_index = 0\n"
    "\n"
    "    def _candidate_batches(self):\n"
    "        # Spread the collection over source columns and rows: with zero noise every\n"
    "        # seed would replay the identical column-0 row-0 cookie.\n"
    "        source = np.asarray(self.env.SOURCE_POSITIONS)\n"
    "        columns = sorted(set(source[:, 0].tolist()))\n"
    "        x = columns[self.variant_index % len(columns)]\n"
    "        order = sorted(\n"
    "            np.flatnonzero(np.isclose(source[:, 0], x)).tolist(),\n"
    "            key=lambda i: source[i, 1],\n"
    "        )\n"
    "        start = (self.variant_index // len(columns)) % len(order)\n"
    "        for k in range(start, len(order)):\n"
    "            batch = [order[k]]\n"
    "            if all(self.env.privileged_cookie_in_source(i) for i in batch):\n"
    "                yield batch\n",
)

sub(
    "        self.target_grip_force = 0.45\n"
    "        self.stable_grip_force = 0.35\n",
    "        band = 0.45 + 0.05 * (self.variant_index % 3)\n"
    "        self.target_grip_force = band\n"
    "        self.stable_grip_force = band - 0.10\n",
)

sub(
    "            column_index = order[0] if self.batch_index < 5 else order[1]\n",
    "            column_index = order[self.variant_index % len(order)]\n",
)

sub(
    "        offset = rows[self.batch_index % len(rows)] - rows.mean()\n",
    "        offset = rows[(self.variant_index // 2) % len(rows)] - rows.mean()\n",
)

sub(
    "    for seed in args.seeds:\n"
    "        row = run_episode(seed, args.mode, args.max_steps, args.noise, out, args.clamps)\n",
    "    for index, seed in enumerate(args.seeds):\n"
    "        row = run_episode(seed, args.mode, args.max_steps, args.noise, out,\n"
    "                          args.clamps, variant=index)\n",
)

P.write_text(src, encoding="utf-8")
print("sha_before", before)
print("sha_after", hashlib.sha256(src.encode()).hexdigest())
print("lines", src.count(chr(10)))
