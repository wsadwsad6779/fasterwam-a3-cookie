"""Causal fingertip force processor: channels 12-15 only, 8-step history,
train-split normalization, inactive-channel mask, invalid-frame handling.
Channels 16/17 (actuator force) and 0-11 (wrist) are never used."""
import numpy as np

CHANNELS = (12, 13, 14, 15)
CHANNEL_NAMES = ('L_inner', 'L_outer', 'R_inner', 'R_outer')


class FingertipForceProcessor:
    def __init__(self, mean, std, hist_len=8, contact_thr=0.05, clip=5.0,
                 std_floor=1e-3, max_abs=500.0):
        self.mean = np.asarray(mean, dtype=np.float64).reshape(4)
        self.std = np.asarray(std, dtype=np.float64).reshape(4)
        self.hist_len = int(hist_len)
        self.contact_thr = float(contact_thr)
        self.clip = float(clip)
        self.std_floor = float(std_floor)
        self.max_abs = float(max_abs)
        self.active = (self.std > 1e-6).astype(np.float64)
        self.eff_std = np.maximum(self.std, self.std_floor)
        self.reset()

    def reset(self):
        self.hist = np.zeros((self.hist_len, 4), dtype=np.float32)
        self.valid = True
        self.abnormal = False
        self.contact = np.zeros(4, dtype=bool)
        self.contact_any = False

    def update(self, force18):
        f = np.asarray(force18, dtype=np.float64).reshape(-1)
        frame_valid = bool(np.all(np.isfinite(f))) and f.size >= 16
        pads = np.zeros(4, dtype=np.float64)
        if frame_valid:
            pads = f[list(CHANNELS)]
            frame_valid = bool(np.all(np.abs(pads) <= self.max_abs))
        self.abnormal = not frame_valid
        if frame_valid:
            norm = np.clip((pads - self.mean) / self.eff_std * self.active, -self.clip, self.clip)
            self.contact = np.abs(pads) > self.contact_thr
        else:
            norm = np.zeros(4, dtype=np.float64)
            self.contact = np.zeros(4, dtype=bool)
        self.valid = frame_valid
        self.contact_any = bool(self.contact.any())
        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = norm.astype(np.float32)
        return self.output()

    def output(self):
        return {
            'hist': self.hist.copy(),
            'mask': self.active.astype(np.float32).copy(),
            'contact': self.contact.copy(),
            'contact_any': self.contact_any,
            'valid': self.valid,
            'abnormal': self.abnormal,
        }