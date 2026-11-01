"""Tier 2: real-model coverage, parameter parity and action parity for v2.

Imports the live harness (test_light_state.py) so the model, the env, the
processor and the reseeded predict path are built exactly the way the passing
v1 test builds them. Nothing here writes to a training run; scratch goes to
--out, which must not exist yet.

Checks, on the real FasterWAM + LoRA model:
  A. every requires_grad=True parameter is enumerated and named, and the
     proprio_encoder rows are reported individually;
  B. v1 adapter -> v2 loader refuses it and says which tensors are missing;
  C. v2 adapter -> independently built model: every trainable tensor equal and
     the predicted action chunk bit-for-bit equal;
  D. the size of the hole: load the same file the v1 way (LoRA only), leaving
     proprio_encoder at the base checkpoint, and measure the action difference;
  E. what r4 and r5 actually left on disk, and whether their proprio_encoder
     can be recovered at all.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), '/workspace/fasterwam_a3_bridge/bridge/_next',
           '/workspace/FasterWAM/src', '/workspace/a3_dual_arm_sim/src',
           '/workspace/FasterWAM'):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import test_light_state as H  # noqa: E402  (installs r4_patch, reseeds predict)
import infer_a3_worker as W  # noqa: E402
import light_state_v2 as V2  # noqa: E402
from fasterwam import light_state as L  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

R = {}
EXPECTED_PROPRIO = 4096 * 8 + 4096


def note(msg):
    print('[v2test] ' + msg, flush=True)


def perturbed(model, seed, proprio_too):
    torch.manual_seed(seed)
    moved = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora_B' in name or (proprio_too and 'proprio_encoder' in name):
                param.normal_(0.0, 1e-3)
                moved.append(name)
    return moved


def chunk(model, obs, processor, inf_cfg, seed, device):
    return W.predict_action_chunk(obs, model, processor, inf_cfg, seed, device)[1]


def trainable_equal(a, b, names):
    na, nb = dict(a.named_parameters()), dict(b.named_parameters())
    bad = [n for n in names if not torch.equal(na[n].detach().float(),
                                               nb[n].detach().float())]
    return bad


def main():
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit('refusing to write into an existing directory: ' + str(out))
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    runs = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('/workspace/a3_train_runs')
    out.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    W.INSTRUCTION = H.INSTRUCTION
    inf_cfg = W.load_cfg()
    lut_cfg = H.train_cfg()
    processor = instantiate(inf_cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(W.load_dataset_stats_from_json(H.STATS))
    env, obs = H.build_obs(seed)

    note('building model A')
    model_a = H.build_model(inf_cfg, lut_cfg.lora, device)
    cov = V2.coverage(model_a)
    R['coverage'] = cov
    R['expected_proprio_elements'] = EXPECTED_PROPRIO
    note('trainable tensors=%d elements=%d kept_by_v1=%d dropped_by_v1=%d elems=%s'
         % (cov['trainable_tensors'], cov['trainable_elements'], cov['kept_by_v1'],
            len(cov['dropped_by_v1']), cov['dropped_by_v1']))
    dropped = {d['name'] for d in cov['dropped_by_v1']}
    R['proprio_named_in_dropped'] = sorted(n for n in dropped if 'proprio' in n)
    R['dropped_elements_total'] = cov['dropped_by_v1_elements']

    names_a = [n for n, _ in V2.trainable_named_parameters(model_a)]
    # the optimizer the trainer builds must be the same set, in the same order
    try:
        import fasterwam.trainer as T
        src = Path(T.__file__).read_text()
        R['trainer_file'] = T.__file__
        i = src.find('proprio_encoder')
        R['trainer_proprio_context'] = (src[max(0, i - 320):i + 200]
                                        if i >= 0 else None)
    except Exception as exc:
        R['trainer_proprio_context'] = 'import failed: ' + repr(exc)

    moved = perturbed(model_a, 1234, proprio_too=True)
    R['perturbed_tensors'] = len(moved)
    R['perturbed_proprio'] = [n for n in moved if 'proprio' in n]
    ca = chunk(model_a, obs, processor, inf_cfg, seed, device)
    V2.save_adapter_v2(model_a, out / 'v2_adapter.pt',
                       meta={'test': 'tier2', 'seed': seed})
    L.save_adapter(model_a, out / 'v1_adapter.pt', meta={'test': 'tier2'})
    R['v2_adapter_bytes'] = (out / 'v2_adapter.pt').stat().st_size
    R['v1_adapter_bytes'] = (out / 'v1_adapter.pt').stat().st_size
    note('adapters: v2=%d B v1=%d B' % (R['v2_adapter_bytes'], R['v1_adapter_bytes']))

    note('building model B (independent, base checkpoint)')
    model_b = H.build_model(inf_cfg, lut_cfg.lora, device)
    perturbed(model_b, 99, proprio_too=True)
    cb0 = chunk(model_b, obs, processor, inf_cfg, seed, device)
    R['control_b0_differs_from_a'] = bool(not torch.equal(cb0, ca))

    # B. v1 file must be refused by the v2 loader, naming the gap.
    try:
        V2.load_adapter_v2(model_b, out / 'v1_adapter.pt')
        R['v1_refused'] = False
    except RuntimeError as exc:
        R['v1_refused'] = str(exc)
    note('v1 refusal: %s' % (R['v1_refused'],))

    # C. v2 file: parameter parity then action parity.
    rep = V2.load_adapter_v2(model_b, out / 'v2_adapter.pt')
    bad = trainable_equal(model_a, model_b, names_a)
    cb = chunk(model_b, obs, processor, inf_cfg, seed, device)
    R['v2_report'] = rep
    R['v2_param_mismatch'] = bad
    R['v2_action_bit_exact'] = bool(torch.equal(ca, cb))
    R['v2_action_maxabs'] = float((ca.float() - cb.float()).abs().max())
    note('v2: mismatch=%s bit_exact=%s maxabs=%.3e'
         % (bad, R['v2_action_bit_exact'], R['v2_action_maxabs']))

    # D. how much does the missing proprio_encoder move the actions?
    del model_b
    torch.cuda.empty_cache()
    model_c = H.build_model(inf_cfg, lut_cfg.lora, device)
    perturbed(model_c, 99, proprio_too=True)
    rep_g = V2.load_adapter_v2(model_c, out / 'v1_adapter.pt', allow_legacy=True)
    cc = chunk(model_c, obs, processor, inf_cfg, seed, device)
    R['legacy_report'] = rep_g
    R['legacy_vs_v2_action_maxabs'] = float((ca.float() - cc.float()).abs().max())
    R['legacy_vs_v2_action_rel'] = float(
        (ca.float() - cc.float()).norm() / ca.float().norm().clamp_min(1e-12))
    note('legacy load: not_restored=%s action maxabs=%.3e rel=%.3e'
         % (rep_g['not_restored'], R['legacy_vs_v2_action_maxabs'],
            R['legacy_vs_v2_action_rel']))

    # E. what r4 / r5 left behind.
    audit = []
    for run in ('a3_lora_r4', 'a3_lora_r5', 'a3_lora_r4_crashed_step500'):
        rd = runs / run
        item = {'run': run, 'exists': rd.is_dir()}
        if not item['exists']:
            audit.append(item)
            continue
        light = rd / 'checkpoints' / 'light'
        item['light_steps'] = sorted(p.name for p in light.iterdir()) if light.is_dir() else []
        item['weights_dir'] = {p.name: p.stat().st_size
                               for p in (rd / 'checkpoints' / 'weights').iterdir()} \
            if (rd / 'checkpoints' / 'weights').is_dir() else None
        item['merged_pt_in_run'] = [p.name for p in rd.glob('*.pt')]
        last = item['light_steps'][-1] if item['light_steps'] else None
        if last:
            d = light / last
            st = json.loads((d / 'trainer_state.json').read_text())
            ap = torch.load(d / 'adapter.pt', map_location='cpu',
                            weights_only=False, mmap=True)
            blob = dict(ap.get('lora') or {})
            item['last'] = last
            item['state_format'] = st.get('format')
            item['state_keys'] = sorted(st)
            item['adapter_format'] = ap.get('format')
            item['adapter_tensors'] = len(blob)
            item['adapter_elements'] = sum(int(v.numel()) for v in blob.values())
            item['adapter_proprio_keys'] = sorted(n for n in blob if 'proprio' in n)
            item['adapter_has_proprio'] = bool(item['adapter_proprio_keys'])
            item['adapter_extra_keys'] = sorted(ap.get('extra_trainable') or {})
            opt = torch.load(d / 'optimizer.pt', map_location='cpu',
                             weights_only=False, mmap=True)
            item['optimizer_groups'] = len(opt.get('param_groups', []))
            item['optimizer_params_per_group'] = [len(g['params'])
                                                 for g in opt.get('param_groups', [])]
        audit.append(item)
        note('audit %s: %s' % (run, json.dumps({k: v for k, v in item.items()
                                                if k != 'light_steps'})))
    R['legacy_runs'] = audit

    # Drift bound: proprio_encoder across the merged checkpoints that do have it.
    chain = []
    for run in ('a3_lora_r1', 'a3_lora_r2', 'a3_lora_r3'):
        for p in sorted((runs / run).glob('*merged*.pt')):
            chain.append(p)
    prev, drift = None, []
    for p in chain:
        try:
            sd = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
        except Exception as exc:
            drift.append({'path': str(p), 'error': repr(exc)})
            continue
        inner = sd.get('model') if isinstance(sd, dict) and 'model' in sd else sd
        keys = [n for n in inner if 'proprio' in str(n)] if isinstance(inner, dict) else []
        cur = {n: inner[n].detach().to('cpu', torch.float32).clone() for n in keys}
        rec = {'path': str(p), 'tensors': sorted(str(n) for n in cur),
               'elements': sum(int(v.numel()) for v in cur.values())}
        if prev:
            for n in cur:
                if n in prev:
                    o, c = prev[n], cur[n]
                    rec.setdefault('delta', {})[str(n)] = {
                        'rel_l2': float((c - o).norm() / o.norm().clamp_min(1e-12)),
                        'max_abs': float((c - o).abs().max()),
                        'rms': float((c - o).pow(2).mean().sqrt()),
                        'ref_rms': float(o.pow(2).mean().sqrt())}
        prev = cur
        drift.append(rec)
        note('proprio %s -> %s' % (p, json.dumps(rec.get('delta', {}))))
    R['proprio_chain'] = drift

    (out / 'test_light_state_v2.json').write_text(json.dumps(R, indent=2))
    note('WROTE ' + str(out / 'test_light_state_v2.json'))


if __name__ == '__main__':
    main()
