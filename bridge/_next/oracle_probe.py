"""Proxy that counts simulator-truth reads made by a controller.

Anything reached through `env.cookie_positions` or `env.data` is simulator truth
that a real robot does not have. `_sensor` is counted separately: force is
legitimately observable, it just happens to be read from the simulator here.

Only external accesses through this proxy are counted. Reads made *inside*
`env.step`/`env.reset` stay on the real env object, so the counts measure the
controller's dependency, not the simulator's own bookkeeping.
"""
import numpy as np

WATCHED_STATE = ("cookie_positions", "data")
WATCHED_SENSOR = ("_sensor",)


class OracleProbe:
    def __init__(self, env):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "counts", {k: 0 for k in WATCHED_STATE + WATCHED_SENSOR})

    def __getattr__(self, name):
        if name in self.counts:
            self.counts[name] += 1
        return getattr(object.__getattribute__(self, "_env"), name)

    def reset_counts(self):
        for k in self.counts:
            self.counts[k] = 0

    def truth_cookie_positions(self):
        """Explicit, counted access for diagnostics only."""
        self.counts["cookie_positions"] += 1
        return np.asarray(object.__getattribute__(self, "_env").cookie_positions)
