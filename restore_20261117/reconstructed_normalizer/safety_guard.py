"""Multi-channel safety guard for diagnostic closed-loop runs (pure function + offline self-test).

Monitored
  wrist force 0:2        norm > WRIST_ABORT              -> abort
  wrist torque 3:5       norm > TORQUE_ABORT             -> abort
  pad_inner/pad_outer    single frame > PAD_ABORT        -> abort
  pad_inner/pad_outer    > PAD_SUSTAIN for >=2 frames    -> suspend 10 steps
  gripper actuator 16    > GRIP_SUSTAIN for >=2 frames   -> suspend 10 steps
  opening step jump      > OPEN_JUMP                     -> wedge flag, suspend gating 5
  blocked-open streak    >= BLOCK_CAP                    -> cool 10 (caller counts)

Actions: 'ok' | 'suspend' | 'abort'. Abort means stop the diagnostic run and save state.
"""
import numpy as np


class Guard:
    def __init__(self, wrist_abort=150.0, torque_abort=None, pad_abort=500.0,
                 pad_sustain=100.0, grip_sustain=200.0, open_jump=0.10,
                 cool_steps=10, wedge_cool=5, block_cap=40):
        self.wrist_abort = wrist_abort
        self.torque_abort = torque_abort
        self.pad_abort = pad_abort
        self.pad_sustain = pad_sustain
        self.grip_sustain = grip_sustain
        self.open_jump = open_jump
        self.cool_steps = cool_steps
        self.wedge_cool = wedge_cool
        self.block_cap = block_cap
        self.pad_hi = 0
        self.grip_hi = 0
        self.cool = 0
        self.prev_open = None
        self.blocks = 0
        self.events = []

    def update(self, force18, opening, blocked=False):
        f = np.asarray(force18, np.float64)
        wrist = float(np.linalg.norm(f[0:3]))
        torque = float(np.linalg.norm(f[3:6]))
        pad = float(max(abs(f[12]), abs(f[13])))
        grip = float(abs(f[16]))
        self.blocks = self.blocks + 1 if blocked else 0
        ev = []
        act = 'ok'
        if pad > self.pad_abort:
            ev.append('pad_abort'); act = 'abort'
        if wrist > self.wrist_abort:
            ev.append('wrist_abort'); act = 'abort'
        if self.torque_abort is not None and torque > self.torque_abort:
            ev.append('torque_abort'); act = 'abort'
        self.pad_hi = self.pad_hi + 1 if pad > self.pad_sustain else 0
        self.grip_hi = self.grip_hi + 1 if grip > self.grip_sustain else 0
        if act != 'abort' and (self.pad_hi >= 2 or self.grip_hi >= 2):
            ev.append('sustain'); act = 'suspend'
            self.cool = max(self.cool, self.cool_steps)
            self.pad_hi = self.grip_hi = 0
        if self.prev_open is not None and abs(float(opening) - self.prev_open) > self.open_jump:
            ev.append('wedge'); act = 'suspend' if act == 'ok' else act
            self.cool = max(self.cool, self.wedge_cool)
        if self.blocks >= self.block_cap:
            ev.append('block_cap'); act = 'suspend' if act == 'ok' else act
            self.cool = max(self.cool, self.cool_steps)
            self.blocks = 0
        self.prev_open = float(opening)
        if self.cool > 0:
            self.cool -= 1
            if act == 'ok':
                act = 'suspend'
        if ev:
            self.events.append({'frame_events': ev, 'pad': round(pad, 1), 'grip': round(grip, 1),
                                'wrist': round(wrist, 1), 'opening': round(float(opening), 3)})
        return act, ev


def _selftest():
    g = Guard()
    base = np.zeros(18)
    base[12] = base[13] = 27.0
    base[16] = 58.0
    base[0:3] = [10, 5, 2]
    ok = g.update(base, 0.55)
    assert ok[0] == 'ok', ok
    a = base.copy(); a[13] = 600.0
    assert g.update(a, 0.55)[0] == 'abort'
    g2 = Guard()
    b = base.copy(); b[13] = 120.0
    assert g2.update(b, 0.55)[0] == 'ok'
    assert g2.update(b, 0.55)[0] == 'suspend'
    g3 = Guard()
    c = base.copy(); c[16] = 250.0
    assert g3.update(c, 0.55)[0] == 'ok'
    assert g3.update(c, 0.55)[0] == 'suspend'
    g4 = Guard()
    g4.update(base, 0.55)
    assert g4.update(base, 0.70)[0] == 'suspend'
    g5 = Guard()
    d = base.copy(); d[0:3] = [200, 0, 0]
    assert g5.update(d, 0.55)[0] == 'abort'
    g6 = Guard(torque_abort=5.0)
    e = base.copy(); e[3:6] = [0, 6, 0]
    assert g6.update(e, 0.55)[0] == 'abort'
    g7 = Guard(block_cap=3)
    for _ in range(3):
        r = g7.update(base, 0.55, blocked=True)
    assert r[0] == 'suspend' and 'block_cap' in r[1], r
    print('SELFTEST_OK 7/7', len(g.events), len(g2.events))


if __name__ == '__main__':
    _selftest()