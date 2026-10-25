"""CPU explanation of the registered correct-union greedy probe selector.

Analyze saved inner-dev counts and exact OOF decisions; never fit a new model.
The constant controls are mathematical, non-deployable oracle controls, not
registered model arms or claimed improvements. Only train-internal B is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from scripts.hanguard import ab_frozen_readout_diagnostics as readouts
from scripts.hanguard import ab_diagnostic_queue as queue
from scripts.hanguard import ab_readout_parallel_queue as ownership

SCHEMA = 'hanguard_greedy_probe_rescue_explanation_v1'


def hard_metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.shape != prediction.shape or not len(y) or not np.isin(y, (0, 1)).all() or not np.isin(prediction, (0, 1)).all():
        raise ValueError('Aligned nonempty binary labels/decisions are required')
    tp = int(((y == 1) & (prediction == 1)).sum())
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    return dict(n=len(y), tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp + tn) / len(y),
        positive_prediction_rate=float(prediction.mean()), positive_label_rate=float(y.mean()),
        recall=tp / (tp + fn) if tp + fn else None,
        specificity=tn / (tn + fp) if tn + fp else None,
        fpr=fp / (tn + fp) if tn + fp else None)


def transitions_by_label(y, candidate, anchor):
    y = np.asarray(y)
    ac, bc = np.asarray(candidate) == y, np.asarray(anchor) == y
    rows = []
    for label in (0, 1):
        mask = y == label
        rescued, harmed = int((mask & ac & ~bc).sum()), int((mask & ~ac & bc).sum())
        rows.append(dict(label=label, label_rows=int(mask.sum()), rescued=rescued, harmed=harmed,
                         both_correct=int((mask & ac & bc).sum()), both_wrong=int((mask & ~ac & ~bc).sum()),
                         net_correct=rescued - harmed))
    return rows


def constant_control(y, anchor, first_greedy_rescue):
    """Fixed hard decisions; never optimize two constant-score thresholds here."""
    y, anchor = np.asarray(y), np.asarray(anchor)
    if not np.array_equal(np.unique(y), np.array([0, 1])):
        raise ValueError('The constant-score AUROC control requires both binary classes')
    return constant_control_from_marginals(hard_metrics(y, anchor), first_greedy_rescue)


def metric_from_counts(value):
    tp, tn, fp, fn = (int(value[key]) for key in ('tp', 'tn', 'fp', 'fn'))
    n = tp + tn + fp + fn
    if n <= 0 or min(tp, tn, fp, fn) < 0 or value.get('n', n) != n:
        raise ValueError('Invalid saved binary confusion counts')
    return dict(n=n, tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp + tn) / n,
        positive_prediction_rate=(tp + fp) / n, positive_label_rate=(tp + fn) / n,
        recall=tp / (tp + fn) if tp + fn else None,
        specificity=tn / (tn + fp) if tn + fp else None,
        fpr=fp / (tn + fp) if tn + fp else None)


def constant_control_from_marginals(anchor, first_greedy_rescue):
    anchor = metric_from_counts(anchor)
    positive, negative = anchor['tp'] + anchor['fn'], anchor['tn'] + anchor['fp']
    if not positive or not negative:
        raise ValueError('The constant-score AUROC control requires both binary classes')
    majority = int(positive >= negative)
    zero = metric_from_counts(dict(tp=0, tn=negative, fp=0, fn=positive))
    one = metric_from_counts(dict(tp=positive, tn=0, fp=negative, fn=0))
    counts = [dict(label=0, label_rows=negative, rescued=anchor['fp'] if majority == 0 else 0,
                  harmed=0 if majority == 0 else anchor['tn']),
              dict(label=1, label_rows=positive, rescued=anchor['fn'] if majority == 1 else 0,
                  harmed=0 if majority == 1 else anchor['tp'])]
    majority_rescue = sum(row['rescued'] for row in counts)
    return dict(non_deployable_oracle_control=True, fixed_hard_decisions=True,
        constant_zero=zero, constant_one=one,
        constant_pair_correctness_union_accuracy=1.,
        constant_score_AUROC=.5, instance_features_used=False,
        no_attainable_model_gain_claimed=True, majority_class=majority,
        majority_rule=one if majority else zero, majority_transitions_by_label=counts,
        hypothetical_majority_first_slot_rescue=majority_rescue,
        original_first_slot_rescue=int(first_greedy_rescue),
        majority_rescue_strictly_exceeds_registered_first_slot=majority_rescue > first_greedy_rescue,
        hypothetical_candidates_were_not_added=True,
        interpretation='Correctness union can reach100% without instance information; choosing the correct constant requires the unknown label. Threshold-optimized constant scores both predict the majority class instead.')


def validate_saved_selection(selection):
    metrics = selection['inner_dev_layer_metrics']
    if set(metrics) != {str(layer) for layer in range(1, 33)}:
        raise ValueError('Saved selection must contain all32 layers')
    expected = max(range(1, 33), key=lambda layer: (metrics[str(layer)]['accuracy'], -metrics[str(layer)]['bce'], layer))
    if expected != selection['best_single_layer']:
        raise ValueError('Saved best layer differs from the registered accuracy/BCE rule')
    selected = [expected] + [row['layer'] for row in selection['rescue_trace']]
    if len(selected) != 4 or len(set(selected)) != 4 or sorted(selected) != selection['selected_layers']:
        raise ValueError('Saved selected layers and trace differ')
    anchor = metric_from_counts(metrics[str(expected)])
    for value in metrics.values():
        hard = metric_from_counts(value)
        if not np.isclose(hard['accuracy'], value['accuracy'], atol=1e-12):
            raise ValueError('Saved accuracy and confusion counts differ')
    covered = anchor['tp'] + anchor['tn'] + sum(row['newly_rescued'] for row in selection['rescue_trace'])
    if covered > anchor['n'] or not np.isclose(covered / anchor['n'], selection['dev_probe_oracle_accuracy'], atol=1e-12):
        raise ValueError('Saved oracle and greedy rescue trace differ')
    return anchor


def marginal_transition_bounds(anchor, candidate, first_rescue=None):
    rows = []
    for label, correct_key, error_key in ((0, 'tn', 'fp'), (1, 'tp', 'fn')):
        a, c = anchor[correct_key], candidate[correct_key]
        n = a + anchor[error_key]
        if c + candidate[error_key] != n:
            raise ValueError('Candidate and anchor class counts differ')
        rows.append(dict(label=label, label_rows=n, rescued_lower=max(0, c - a),
            rescued_upper=min(c, n - a), net_correct=c - a))
    original = [dict(row) for row in rows]
    if first_rescue is not None:
        for label in (0, 1):
            rows[label]['rescued_lower'] = max(original[label]['rescued_lower'], first_rescue - original[1 - label]['rescued_upper'])
            rows[label]['rescued_upper'] = min(original[label]['rescued_upper'], first_rescue - original[1 - label]['rescued_lower'])
    for row in rows:
        if row['rescued_lower'] > row['rescued_upper']:
            raise ValueError('Saved first-step rescue is incompatible with confusion marginals')
        row.update(harmed_lower=row['rescued_lower'] - row['net_correct'],
                   harmed_upper=row['rescued_upper'] - row['net_correct'],
                   exact_joint_decisions_available=False)
    return rows


def analyze_saved_dev(fold, selection):
    anchor = validate_saved_selection(selection)
    best, saved = selection['best_single_layer'], selection['inner_dev_layer_metrics']
    metric_rows, transition_rows, trace_rows = [], [], []
    for layer in range(1, 33):
        metric = metric_from_counts(saved[str(layer)])
        metric.update(fold=fold, scope='inner_dev_saved', layer=layer, anchor_layer=best,
            selected=layer in selection['selected_layers'], selected_anchor=layer == best,
            accuracy_delta_vs_anchor=metric['accuracy'] - anchor['accuracy'],
            saved_AUROC=saved[str(layer)]['auroc'], saved_BCE=saved[str(layer)]['bce'])
        metric_rows.append(metric)
        first = selection['rescue_trace'][0]
        total = 0 if layer == best else (first['newly_rescued'] if layer == first['layer'] else None)
        for bound in marginal_transition_bounds(anchor, metric, total):
            transition_rows.append(dict(fold=fold, scope='inner_dev_saved', layer=layer,
                anchor_layer=best, selected=layer in selection['selected_layers'], selected_anchor=layer == best, **bound))
    covered = anchor['tp'] + anchor['tn']
    errors = [anchor['fp'], anchor['fn']]
    for step, row in enumerate(selection['rescue_trace'], start=1):
        rescued = row['newly_rescued']
        candidate = saved[str(row['layer'])]
        upper = [min(rescued, errors[0], candidate['tn']), min(rescued, errors[1], candidate['tp'])]
        before = covered
        covered += rescued
        for label in (0, 1):
            trace_rows.append(dict(fold=fold, scope='inner_dev_saved', step=step, layer=row['layer'],
                anchor_layer=best, label=label, label_rows=anchor['tn'] + anchor['fp'] if label == 0 else anchor['tp'] + anchor['fn'],
                newly_rescued_lower=max(0, rescued - upper[1 - label]), newly_rescued_upper=upper[label],
                exact_total_newly_rescued=rescued, oracle_covered_before=before,
                oracle_covered_after=covered, oracle_accuracy_after=covered / anchor['n'],
                exact_joint_decisions_available=False, oracle_not_deployable=True))
    return metric_rows, transition_rows, trace_rows, dict(fold=fold, scope='inner_dev_saved',
        anchor_layer=best, anchor_accuracy=anchor['accuracy'],
        selected_probe_oracle_accuracy=selection['dev_probe_oracle_accuracy'],
        anchor_errors=anchor['fp'] + anchor['fn'], selected_set_oracle_rescued=sum(row['newly_rescued'] for row in selection['rescue_trace']),
        rows=anchor['n'], oracle_not_deployable=True)


def analyze_scope(fold, scope, y, predictions, selection, stored_metrics):
    anchor = selection['best_single_layer']
    anchor_predictions = predictions[:, anchor - 1]
    anchor_accuracy = hard_metrics(y, anchor_predictions)['accuracy']
    metrics_rows, transition_rows, trace_rows = [], [], []
    selected = selection['selected_layers']
    for layer in range(1, 33):
        metric = hard_metrics(y, predictions[:, layer - 1])
        metric.update(fold=fold, scope=scope, layer=layer, anchor_layer=anchor,
            selected=layer in selected, selected_anchor=layer == anchor,
            accuracy_delta_vs_anchor=metric['accuracy'] - anchor_accuracy,
            saved_AUROC=stored_metrics[str(layer)]['auroc'], saved_BCE=stored_metrics[str(layer)]['bce'])
        metrics_rows.append(metric)
        for transition in transitions_by_label(y, predictions[:, layer - 1], anchor_predictions):
            transition_rows.append(dict(fold=fold, scope=scope, layer=layer, anchor_layer=anchor,
                selected=layer in selected, selected_anchor=layer == anchor, **transition))
    correct = predictions == y[:, None]
    covered = correct[:, anchor - 1].copy()
    sequence = [anchor] + [step['layer'] for step in selection['rescue_trace']]
    for step, layer in enumerate(sequence):
        newly = (~covered & correct[:, layer - 1]) if step else np.zeros(len(y), dtype=bool)
        before = int(covered.sum())
        covered |= correct[:, layer - 1]
        for label in (0, 1):
            trace_rows.append(dict(fold=fold, scope=scope, step=step, layer=layer, anchor_layer=anchor,
                label=label, label_rows=int((y == label).sum()),
                newly_rescued=int((newly & (y == label)).sum()),
                oracle_covered_before=before, oracle_covered_after=int(covered.sum()),
                oracle_accuracy_after=float(covered.mean()), oracle_not_deployable=True))
        if scope == 'inner_dev_replay' and step and int(newly.sum()) != selection['rescue_trace'][step - 1]['newly_rescued']:
            raise ValueError('Registered greedy dev rescue trace did not replay')
    return metrics_rows, transition_rows, trace_rows, dict(fold=fold, scope=scope,
        anchor_layer=anchor, anchor_accuracy=anchor_accuracy, selected_probe_oracle_accuracy=float(covered.mean()),
        anchor_errors=int((~correct[:, anchor - 1]).sum()), selected_set_oracle_rescued=int((covered & ~correct[:, anchor - 1]).sum()),
        rows=len(y), oracle_not_deployable=True)


def _write_table(output, name, records):
    frame = pd.DataFrame(records)
    frame.to_parquet(output / f'{name}.parquet', index=False)
    frame.to_csv(output / f'{name}.csv', index=False)


def build_audit(readout, output, cpu_threads=4):
    readout, output = Path(readout).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Explanation evidence is immutable; use a fresh output path')
    if cpu_threads != 4:
        raise ValueError('CPU replay uses the fixed four-thread resource cap')
    torch.set_num_threads(cpu_threads)
    protocol = readouts.read_json(readout / 'protocol.json')
    if protocol['representation'] != 'adapted' or protocol['pooling'] != 'last' or protocol['outer_folds'] != list(range(5)):
        raise ValueError('This explanation requires the complete registered adapted_last five folds')
    cache, metadata, manifest = readouts.load_b_cache(Path(protocol['cache_manifest_path']))
    if queue.sha(protocol['cache_manifest_path']) != protocol['cache_manifest_sha256']:
        raise ValueError('Readout cache manifest identity changed')
    inputs = {str(readout / 'protocol.json'): queue.sha(readout / 'protocol.json'),
              str(Path(__file__).resolve()): queue.sha(__file__)}
    for source in (queue.__file__, ownership.__file__):
        inputs[str(Path(source).resolve())] = queue.sha(source)
    for name, digest in protocol['code_sha256'].items():
        path = ROOT / 'scripts/hanguard' / name
        if queue.sha(path) != digest:
            raise ValueError(f'Registered scientific source changed: {name}')
        inputs[str(path)] = digest
    for path in (Path(protocol['cache_manifest_path']), Path(manifest['path']), Path(manifest['metadata_path']),
                 Path(manifest['ab_protocol_path']), Path(manifest['training_protocol_path'])):
        inputs[str(path)] = queue.sha(path)
    partition = readouts.read_json(manifest['ab_protocol_path'])
    inputs[str(Path(manifest['ab_protocol_path']).parent / 'B.parquet')] = partition['artifact_sha256']['B.parquet']
    for name in ('lora_data/train.parquet', 'lora_data/validation.parquet'):
        inputs[str(Path(manifest['ab_protocol_path']).parent / name)] = partition['artifact_sha256'][name]
    for path in (Path(manifest['training_protocol_path']).parent / 'registration.json',
                 Path(manifest['metadata_path']).parent / 'replay_audit.json',
                 Path(manifest['checkpoint_lineage_path']), Path(manifest['checkpoint_lineage_path']).parent / 'adapter.pt'):
        inputs[str(path)] = queue.sha(path)
    probe_oof = readouts.aggregate_stage(readout, protocol, metadata, 'probes')
    head_oof = readouts.aggregate_stage(readout, protocol, metadata, 'heads')
    for name, expected in (('probe_outer_predictions.parquet', probe_oof), ('head_outer_predictions.parquet', head_oof)):
        path = readout / 'analysis' / name
        if not pd.read_parquet(path).equals(expected):
            raise ValueError(f'Bound analysis OOF differs from completed fold predictions: {name}')
        inputs[str(path)] = queue.sha(path)
    results_path = readout / 'analysis/results.json'
    existing_results = readouts.read_json(results_path)
    if existing_results['protocol_sha256'] != inputs[str(readout / 'protocol.json')]:
        raise ValueError('Existing head/probe report protocol identity differs')
    inputs[str(results_path)] = queue.sha(results_path)
    all_metrics, transitions, traces, oracle, constants, consistency = [], [], [], [], [], []
    for fold in range(5):
        readouts.check_stop(protocol)
        inputs.update(ownership.validate_fold(readout, fold, inputs[str(readout / 'protocol.json')] ))
        indices = readouts.nested_indices(metadata, fold)
        entry = next(value for value in partition['folds'] if value['fold'] == fold)
        inputs[entry['assignments_path']] = queue.sha(entry['assignments_path'])
        selection = readouts.read_json(readout / f'fold_{fold}/probes/selection.json')
        expected_provenance = readouts._provenance(readout, protocol, fold, indices, metadata, 'probes')
        if selection['provenance'] != expected_provenance:
            raise ValueError('Saved probe selection has another fold identity')
        y_dev = metadata.iloc[indices['dev']].label.to_numpy(dtype=np.int64)
        anchor = validate_saved_selection(selection)
        if anchor['n'] != len(y_dev) or anchor['tp'] + anchor['fn'] != int(y_dev.sum()):
            raise ValueError('Saved dev marginal labels differ from the registered dev rows')
        m, t, steps, summary = analyze_saved_dev(fold, selection)
        all_metrics.extend(m); transitions.extend(t); traces.extend(steps); oracle.append(summary)
        control = constant_control_from_marginals(anchor, selection['rescue_trace'][0]['newly_rescued'])
        control.update(fold=fold, scope='inner_dev_saved', first_slot_comparison_uses_dev_counts_only=True)
        constants.append(control)
        consistency.append(dict(fold=fold, saved_best_rule_matches=True,
            saved_counts_accuracy_and_oracle_trace_consistent=True, dev_row_label_marginals_match=True,
            dev_joint_predictions_not_saved=True, no_CPU_mask_substitution=True))
        held = probe_oof[probe_oof.outer_fold.eq(fold)]
        held_pred = held[[f'prediction_L{layer}' for layer in range(1, 33)]].to_numpy(dtype=np.int8)
        held_y = held.label.to_numpy(dtype=np.int64)
        complete = readouts.read_json(readout / f'fold_{fold}/probes/complete.json')
        for scope, y, pred, saved in (('outer_held', held_y, held_pred, complete['result']['held_layer_metrics']),):
            m, t, steps, summary = analyze_scope(fold, scope, y, pred, selection, saved)
            all_metrics.extend(m); transitions.extend(t); traces.extend(steps); oracle.append(summary)
            first_rescue = sum(step['newly_rescued'] for step in steps if step['step'] == 1)
            control = constant_control(y, pred[:, selection['best_single_layer'] - 1], first_rescue)
            control.update(fold=fold, scope=scope,
                first_slot_comparison_uses_dev_counts_only=False)
            constants.append(control)
        print(json.dumps(dict(state='CPU_probe_rescue_explanation', completed_fold=fold,
                              dev_uses_authoritative_saved_marginals_and_trace=True, OOF_decisions_exact=True)), flush=True)
    curve = []
    for layer in range(1, 33):
        metric = hard_metrics(probe_oof.label.to_numpy(), probe_oof[f'prediction_L{layer}'].to_numpy())
        metric.update(layer=layer, scope='pooled_OOF_only',
            saved_AUROC=existing_results['layer_metrics'][str(layer)]['auroc'],
            selected_fold_count=sum(row['selected'] for row in all_metrics if row['scope'] == 'outer_held' and row['layer'] == layer))
        curve.append(metric)
    layer_frame = pd.DataFrame(all_metrics)
    shallow = layer_frame[(layer_frame.scope == 'outer_held') & layer_frame.selected & (layer_frame.layer <= 5)]
    dev_controls = [value for value in constants if value['scope'] == 'inner_dev_saved']
    saved_comparisons = [value for value in existing_results['comparisons']
                        if value['candidate'] in ('fusion_selected', 'fusion_selected_best_anchor')
                        and value['baseline'] in ('best_single', 'repeat_best', 'fusion_selected')]
    for name, digest in inputs.items():
        if queue.sha(name) != digest:
            raise ValueError(f'Explanation input changed during CPU replay: {name}')
    output.mkdir(parents=True)
    _write_table(output, 'layers_by_fold', all_metrics)
    _write_table(output, 'layers_OOF_curve', curve)
    _write_table(output, 'layer_rescue_harm_by_label', transitions)
    _write_table(output, 'greedy_trace_by_label', traces)
    queue.dump(output / 'dev_saved_metric_consistency.json', dict(folds=consistency,
        dev_accuracy_positive_rate_and_trace_exact=True, dev_label_joint_rescue_harm_are_bounds=True,
        reason='Original per-row dev decisions were not saved. GPU-probability-valued thresholds are sensitive to tiny CPU/GPU rounding differences. No threshold, prediction, checkpoint, or selected layer was modified.'))
    queue.dump(output / 'constant_oracle_controls.json', dict(controls=constants,
        mathematical_only=True, fixed_zero_plus_one_oracle_accuracy=1.,
        no_new_registered_candidate=True, no_deployable_model_gain=True))
    result = dict(schema=SCHEMA, complete=True, readout=str(readout),
        formal_validation_or_test_opened=False, no_training=True, original_selection_rule_changed=False,
        source_and_input_sha256=inputs, OOF_decisions_exact=True,
        dev_uses_saved_authoritative_confusion_and_trace=True, dev_joint_label_transitions_are_marginal_bounds=True,
        numerical_replay_limitation='A threshold may equal a GPU sample probability; CPU/GPU rounding can flip a boundary row. No CPU mask was substituted or threshold tuned; this is not evidence explaining actual fusion degradation.',
        nested_dev_metrics_are_descriptive_and_not_independent_OOF=True,
        greedy_objective=dict(primary='uncovered correct decisions', ties=['higher dev accuracy', 'lower dev BCE', 'higher layer index'],
            harm_penalty=False, candidate_quality_floor=False),
        oracle_by_fold_scope=oracle, saved_registered_head_comparisons=saved_comparisons,
        shallow_selected_descriptive=dict(layer_range=[1, 5], selected_fold_layer_pairs=len(shallow),
            OOF_accuracy_range=[float(shallow.accuracy.min()), float(shallow.accuracy.max())],
            OOF_positive_prediction_rate_range=[float(shallow.positive_prediction_rate.min()), float(shallow.positive_prediction_rate.max())],
            OOF_AUROC_range=[float(shallow.saved_AUROC.min()), float(shallow.saved_AUROC.max())]),
        constant_majority_hypothetical_first_slot_wins=sum(value['majority_rescue_strictly_exceeds_registered_first_slot'] for value in dev_controls),
        constant_majority_hypothetical_first_slot_folds=len(dev_controls),
        bounded_conclusions=[
            'The registered rescue objective rewards an oracle correctness union and does not penalize candidate errors on anchor-correct rows.',
            'A fixed all-zero/all-one pair attains100% correctness union without instance information; this is a non-deployable metric control, never a model gain.',
            'Actual selected shallow probes predict both classes and possess substantial discrimination; the evidence does not show single-class collapse.',
            'The hypothetical majority-only first-slot comparison tests the objective, without adding candidates or changing the registered rule.',
            'Probe-selection and this frozen readout may be misaligned; the analysis neither proves causality of the entire fusion deficit nor absence of useful raw early-layer information.',
            'These results cannot establish that data are clean or that all multilayer fusion methods fail.'],
        artifact_sha256={path.name: queue.sha(path) for path in output.iterdir() if path.is_file()})
    queue.dump(output / 'results.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run this explanation with CUDA_VISIBLE_DEVICES=""; GPU use is forbidden')
    print(json.dumps(build_audit(args.readout, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
