"""Run `flow_test.py` on a machine that has no torch -- a pre-check, not a result.

`flow_test.py` is the CPU flow test for the 40k evaluation loop, and the real one runs on the pod
with the real torch. It is still worth executing here, because it builds its policies the way the
drivers do (`new_policy` assigns the attributes `ForcePolicy.__init__` would have set), which is
how a constructor change -- the `proprio_fn` parameter, for instance -- breaks the fixture if the
fixture is not updated with it. That class of failure is a missing attribute, not a wrong number,
so a stub is enough to catch it.

The stub replaces the handful of torch names the fixtures touch: `from_numpy`, `tensor`, `device`,
the dtypes, `no_grad` and a `cuda` namespace that reports no GPU, which is the branch the device
checks assert on. A stubbed tensor is an `ndarray` subclass that answers `.to(...)` and the in-place
`mul_`/`add_` the mutation fixture uses, so every shape and value assertion in `flow_test.py` runs
against the same numbers it would on the pod. What this cannot test is anything that needs the model, the simulator or the corpus --
so the record of a real acceptance run is the pod's own `python flow_test.py`, and this prints a
banner saying which one it is.

    py -3.13 flow_test_local_shim.py        (local, no torch)
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class _Tensor(np.ndarray):
    """An ndarray that answers the few tensor calls the CPU fixtures make."""

    def to(self, *a, **kw):
        return self

    def unsqueeze(self, dim):
        return np.expand_dims(np.asarray(self.view(np.ndarray)), dim)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self if self.dtype == np.float32 else self.astype(np.float32).view(_Tensor)

    def numpy(self):
        # A real float32 CPU tensor's `.numpy()` shares memory with it, and that is the whole
        # point: it is why the recorder has to `.copy()` its before-snapshot. Returning a view
        # here keeps that assertion honest instead of making it pass by accident.
        return self.view(np.ndarray)

    def mul_(self, k):
        np.multiply(self.view(np.ndarray), k, out=self.view(np.ndarray))
        return self

    def add_(self, k):
        np.add(self.view(np.ndarray), k, out=self.view(np.ndarray))
        return self


class _Device:
    """The two attributes of `torch.device` that `same_device` reads, parsed the same way."""

    def __init__(self, spec):
        if isinstance(spec, _Device):
            self.type, self.index = spec.type, spec.index
            return
        type_, sep, index = str(spec).partition(":")
        if not sep and type_ not in ("cpu", "cuda"):
            raise ValueError("stub torch cannot parse a device from %r" % (spec,))
        self.type = type_
        self.index = None if not sep else int(index)

    def __str__(self):
        return self.type if self.index is None else "%s:%d" % (self.type, self.index)

    def __repr__(self):
        return str(self)


def install_stub_torch():
    mod = types.ModuleType("torch")
    mod.float32 = np.float32
    mod.bfloat16 = np.float32
    mod.float64 = np.float64
    mod.int64 = np.int64
    mod.device = _Device

    def from_numpy(a):
        return np.asarray(a).view(_Tensor)

    def tensor(a, dtype=None):
        return from_numpy(np.asarray(a, dtype=dtype)) if dtype else from_numpy(a)

    mod.from_numpy = from_numpy
    mod.tensor = tensor
    # A CPU-only box: `cuda.is_available()` is False, which is the branch the fixtures assert on.
    mod.cuda = types.SimpleNamespace(is_available=lambda: False, current_device=lambda: 0)

    @contextmanager
    def no_grad():
        yield

    mod.no_grad = no_grad
    sys.modules["torch"] = mod
    return mod


if __name__ == "__main__":
    install_stub_torch()
    print("BANNER: flow_test under a stubbed torch (%s) -- a wiring pre-check on the machine "
          "that has no GPU, NOT the pod acceptance run" % np.__version__, flush=True)
    import flow_test
    results = flow_test.run()
    raise SystemExit(0 if results["all_passed"] else 1)
