import json, datetime
from collections import OrderedDict

led = json.load(open('ROOTCAUSE_EVIDENCE_LEDGER_20261118.json'), object_pairs_hook=OrderedDict)
sat = json.load(open('SEED17_FORCE_SATURATION_CPU_20261118.json'))
cpl = json.load(open('SEED17_FORCE_ACTION_COUPLING_CPU_20261118.json'))
v5 = json.load(open('SEED17_V5_ADAPTER_REPLAY_CPU_20261118.json'))

new = OrderedDict()
for k, v in led.items():
    new[k] = v
    if k == 'verified_local_mechanism':
        new['force_interface_saturation_cpu'] = OrderedDict([
            ("claim", "The deployed seed17 force inputs are far outside the training support; P1 (proprio force columns) is clamped to 5 sigma after normalisation and P2 (16-frame raw-newton history) is fed unclamped at hundreds of sigma."),
            ("P1_mechanism", sat['semantics']['P1_proprio_force_columns']),
            ("P2_mechanism", sat['semantics']['P2_force_history']),
            ("training_source_range_P1_sigma", OrderedDict([
                ("min", sat['training_corpus']['P1_normalised']['min']),
                ("max", sat['training_corpus']['P1_normalised']['max']),
                ("pad_inner_max_sigma", sat['training_corpus']['P1_normalised']['max'][3]),
                ("pad_outer_max_sigma", sat['training_corpus']['P1_normalised']['max'][4]),
                ("frames_any_abs_gt_5", sat['training_corpus']['P1_frames_any_over_5']['frames_any']),
                ("note", "pad_inner/pad_outer never exceed 2.27 sigma anywhere in the 42681-row corpus; only wrist fy/fz cross 5 sigma in 300/320 rows.")])),
            ("training_source_range_P2_sigma", OrderedDict([
                ("min", sat['training_corpus']['P2_normalised']['min']),
                ("max", sat['training_corpus']['P2_normalised']['max'])])),
            ("deployment_seed17_P1_sigma", OrderedDict([
                ("min", sat['deployment']['P1_normalised_raw']['min']),
                ("max", sat['deployment']['P1_normalised_raw']['max']),
                ("frames_any_abs_gt_5", sat['deployment']['P1_frames_any_over_5']['frames_any']),
                ("fraction_any_abs_gt_5", sat['deployment']['P1_frames_any_over_5']['fraction_any']),
                ("first_saturated_frame", sat['deployment']['P1_frames_any_over_5']['first_frame']),
                ("jam_frames_134_311_saturated", sat['deployment']['saturated_frames_jam_134_311']),
                ("jam_fraction_134_311", sat['deployment']['saturated_fraction_jam_134_311']),
                ("pads_both_saturated_longest_run_frames", sat['deployment']['pads_both_saturated_longest'])])),
            ("deployment_seed17_P2_sigma", OrderedDict([
                ("min", sat['deployment']['P2_normalised']['min']),
                ("max", sat['deployment']['P2_normalised']['max'])])),
            ("amplitude_lost_to_P1_clamp", OrderedDict([
                ("jam_clip_removed_L2_ratio_median", sat['deployment']['clip_removed_ratio_jam_median']),
                ("jam_clip_removed_L2_ratio_max", sat['deployment']['clip_removed_ratio_jam_max']),
                ("preclip_norm_median_sigma", sat['deployment']['preclip_norm_jam_median']),
                ("preclip_norm_max_sigma", sat['deployment']['preclip_norm_jam_max'])])),
            ("downstream_mechanism", "force_condition.py keeps the raw amplitude on the residual trunk (_CausalResidualBlock) and does not normalise the cross-attention force K/V, so P2's hundreds-of-sigma history can inject large magnitudes; P1's clamp destroys pad amplitude information by pinning it to a constant 5."),
            ("limits", sat['limits'] + ["Saturation is a measured interface fact; it does not by itself prove causation of contact onset or gripper reversals."]),
            ("source", "SEED17_FORCE_SATURATION_CPU_20261118.json")])
        new['force_action_observational_coupling_cpu'] = OrderedDict([
            ("claim", "Within the recorded seed17 action stream, force-history peak and model-output action amplitude co-vary strongly in the same frame and the recorded executed chunks sit inside the amplified regime, so the amplification is not a re-inference artifact."),
            ("pre_contact_i0_60", cpl['values']['pre_contact_i0_60']),
            ("jam_i134_311", cpl['values']['jam_i134_311']),
            ("whole_episode", cpl['values']['whole_episode']),
            ("same_frame_correlation_pad_max_vs_absmax", cpl['correlation_same_frame']),
            ("lag_correlation_pad_leads_action", cpl['correlation_pad_vs_absmax_by_lag']),
            ("recorded_chunk_absmax", "chunks 14/15/16/17/18/19/20/21/30/31 record 6.75/21.25/30.5/30.1/27.1/43.75/26.75/94.0/31.25/10.9; handed_14d translation saturates in 248/600 rows and 90.4% of jam frames."),
            ("counterexample", "i350 has pad=0 but wrist fz=-88.2 N and raw absmax still 15.25, so wrist force alone can sustain amplification."),
            ("limits", cpl['limits']),
            ("source", "SEED17_FORCE_ACTION_COUPLING_CPU_20261118.json")])
        new['v5_counterfactual_plan_semantics_cpu'] = OrderedDict([
            ("claim", "Replaying the v5 clip counterfactuals through the identified de-normaliser and adapter shows a dose-response in time: at pre-contact f60 clipping changes nothing, while at f64/f70/f143/f159 it changes translation signs, saturation rows and gripper commands."),
            ("denorm_identified", v5['denorm_identification']),
            ("rotation_caveat", v5['rotation_note']),
            ("f60_precontact", OrderedDict([
                ("raw_absmax_baseline", 1.0703125), ("raw_absmax_clip", 1.0703125),
                ("trans_sign_agree_xyz_of_10", [10, 10, 10]), ("gripper_cmd_agree_of_10", 10),
                ("interpretation", "No history entry was out of source range yet, so the clip is a null operation; this is the temporal specificity control.")])),
            ("contact_and_jam_pairs", [
                OrderedDict([("tag", p['tag']), ("baseline_absmax_raw", p['baseline_absmax_raw']),
                             ("clip_absmax_raw", p['clip_absmax_raw']),
                             ("baseline_trans_saturated_rows_of_10", p['baseline_trans_saturated_rows']),
                             ("clip_trans_saturated_rows_of_10", p['clip_trans_saturated_rows']),
                             ("trans_sign_agree_xyz_of_10", p['trans_sign_agree']),
                             ("gripper_cmd_agree_of_10", p['grip_cmd_agree'])]) for p in v5['pairs']]),
            ("interpretation", "These are offline predicted-plan comparisons. A sign disagreement says the plans diverge, not that either plan or the clipped policy is correct or safe."),
            ("limits", v5['limits']),
            ("source", "SEED17_V5_ADAPTER_REPLAY_CPU_20261118.json")])
        continue

arts = new['verified_artifacts']
for a in ['SEED17_FORCE_SATURATION_CPU_20261118.json', 'SEED17_FORCE_ACTION_COUPLING_CPU_20261118.json', 'SEED17_V5_ADAPTER_REPLAY_CPU_20261118.json']:
    if a not in arts:
        arts.append(a)

new['ledger_updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
with open('ROOTCAUSE_EVIDENCE_LEDGER_20261118.json', 'w', encoding='utf-8') as f:
    json.dump(new, f, indent=2, ensure_ascii=False)
print('keys:', list(new.keys()))
print('bytes:', len(json.dumps(new)))
