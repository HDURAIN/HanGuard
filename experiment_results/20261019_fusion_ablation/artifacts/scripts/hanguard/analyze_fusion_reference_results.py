"""Recompute the immutable no-LoRA reference on both original and selected rows.

This is CPU analysis of saved predictions, never model fitting.  The selected
test set is explicitly outcome selected; no confidence interval here corrects
that selection bias.  Run with ``python -m scripts.hanguard.analyze_fusion_reference_results``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.hanguard.frozen_lora_layer_diagnostics import (
    binary_metrics, dump, group_cluster_bootstrap, sha256,
)

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("last", "best_single", "repeat_best", "fusion_fixed", "fusion_fixed_best_anchor")
PAIRS = (("fusion_fixed_best_anchor", "best_single"),
         ("fusion_fixed_best_anchor", "repeat_best"),
         ("fusion_fixed", "best_single"))
POLICY = "posthoc_error_selected_test_subset"
SCOPES = ("full8350", "subset8228")


def read(path):
    return json.loads(Path(path).read_text())


def disclosure(scope):
    return dict(scope=scope, subset_selection_policy=POLICY,
                outcome_selected=scope == "subset8228", blind_test=False,
                is_quality_cleanup=False)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path,
                        default=ROOT / "outputs/hanguard_frozen_base_full_readouts_20261013_v5")
    parser.add_argument("--subset", type=Path,
                        default=ROOT / "data/three_source_wildguard122_subset_20261018")
    parser.add_argument("--lora-study", type=Path,
                        default=ROOT / "outputs/hanguard_selected122_lora_20261018")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/hanguard_nolora_reference_analysis_20261019")
    args = parser.parse_args(argv)
    study, subset, output = args.study.resolve(), args.subset.resolve(), args.output.resolve()
    protocol_path, selection_path = study / "protocol.json", study / "heads/selection.json"
    result_path, predictions_path = study / "analysis/results.json", study / "analysis/predictions.parquet"
    protocol, selection, original = map(read, (protocol_path, selection_path, result_path))
    parent = Path(protocol["data_dir"])
    policy_path, ids_path = subset / "filter_policy.json", subset / "excluded_base_ids.txt"
    policy, old_subset = read(policy_path), read(subset / "old_model_subset_metrics.json")
    _check(policy["selection_policy"] == POLICY, "The selected-set policy changed")
    _check(original["artifacts"]["predictions.parquet"] == sha256(predictions_path),
           "Reference predictions no longer match their registered artifact")
    frame = pd.read_parquet(predictions_path)
    _check(len(frame) == 8350 and not frame.base_id.duplicated().any(), "Invalid full reference rows")
    excluded = ids_path.read_text().splitlines()
    _check(len(excluded) == len(set(excluded)) == 122, "Exactly 122 distinct excluded IDs required")
    retained = frame.loc[~frame.base_id.isin(excluded)].copy()
    dropped = frame.loc[frame.base_id.isin(excluded)].copy()
    _check(len(retained) == 8228 and len(dropped) == 122, "Excluded IDs do not cover exactly 122 test rows")
    predicate = ((frame.source == "wildguard_zh") & (frame.prediction_best_single == frame.label) &
                 (frame.prediction_fusion_fixed_best_anchor != frame.label))
    _check(set(frame.loc[predicate, "base_id"]) == set(excluded), "The frozen outcome-selection predicate changed")
    columns = ["base_id", "group_id", "source", "prompt_harm_label", "prompt_sha256"]
    release = pd.read_parquet(parent / "test.parquet", columns=columns).set_index("base_id")
    selected_release = pd.read_parquet(subset / "test.parquet", columns=columns).set_index("base_id")
    _check(set(release.index) == set(frame.base_id), "Full prediction/data identity mismatch")
    _check(set(selected_release.index) == set(retained.base_id), "Subset prediction/data identity mismatch")
    expected_subset = release.loc[selected_release.index]
    pd.testing.assert_frame_equal(expected_subset, selected_release, check_dtype=True)
    aligned = release.loc[frame.base_id]
    _check(np.array_equal(aligned.prompt_harm_label.eq("harmful").to_numpy(), frame.label.to_numpy()),
           "Saved labels differ from original release labels")
    for column in ("group_id", "source"):
        _check(np.array_equal(aligned[column].to_numpy(), frame[column].to_numpy()),
               f"Saved {column} differs from release")
    for split in ("train", "validation"):
        _check(sha256(parent / f"{split}.parquet") == sha256(subset / f"{split}.parquet"),
               f"{split} is not byte identical")
    for arm in ARMS:
        threshold = selection["thresholds"][arm]
        _check(frame[f"threshold_{arm}"].eq(threshold).all(), f"Threshold mismatch: {arm}")
        pred = (frame[f"probability_{arm}"] >= threshold).astype(int)
        _check(pred.eq(frame[f"prediction_{arm}"]).all(), f"Saved prediction mismatch: {arm}")

    metrics_rows, sources_rows, comparisons = [], [], []
    metrics_by_scope = {}
    for scope, part in zip(SCOPES, (frame, retained)):
        info = disclosure(scope)
        metrics_by_scope[scope] = {}
        y = part.label.to_numpy(dtype=int)
        for mode in ("validation_accuracy", "fixed_0.5"):
            mode_metrics = {}
            for arm in ARMS:
                threshold = selection["thresholds"][arm] if mode == "validation_accuracy" else .5
                metrics = binary_metrics(y, part[f"probability_{arm}"], threshold)
                mode_metrics[arm] = metrics
                metrics_rows.append(dict(**info, threshold_mode=mode, method=arm,
                                         parameters=original["parameters"][arm], **metrics))
                for source, source_part in part.groupby("source", sort=True):
                    sources_rows.append(dict(**info, threshold_mode=mode, method=arm, source=source,
                        **binary_metrics(source_part.label, source_part[f"probability_{arm}"], threshold)))
            metrics_by_scope[scope][mode] = mode_metrics
            for candidate, baseline in PAIRS:
                a_threshold = mode_metrics[candidate]["threshold"]
                b_threshold = mode_metrics[baseline]["threshold"]
                a = part[f"probability_{candidate}"].to_numpy() >= a_threshold
                b = part[f"probability_{baseline}"].to_numpy() >= b_threshold
                a_correct, b_correct = a == y, b == y
                bootstrap = group_cluster_bootstrap(y, part[f"probability_{candidate}"],
                    part[f"probability_{baseline}"], part.group_id, a_threshold, b_threshold,
                    n_resamples=2000, seed=42)
                comparisons.append(dict(**info, threshold_mode=mode, candidate=candidate, baseline=baseline,
                    rows=len(part), groups=part.group_id.nunique(),
                    rescued=int((a_correct & ~b_correct).sum()), harmed=int((~a_correct & b_correct).sum()),
                    both_correct=int((a_correct & b_correct).sum()), both_wrong=int((~a_correct & ~b_correct).sum()),
                    net_correct=int(a_correct.sum() - b_correct.sum()),
                    accuracy_gain_pp=bootstrap["accuracy"]["delta"] * 100,
                    accuracy_ci95_pp=[v * 100 for v in bootstrap["accuracy"]["ci95"]],
                    group_bootstrap=bootstrap,
                    bootstrap_interpretation=("Conditional on this frozen selected set; does not correct outcome-selection bias"
                                              if scope == "subset8228" else "Conditional on this explored release and fixed trained heads")))

    arithmetic = dict(subset_selection_policy=POLICY, full_rows=8350, retained_rows=8228,
                      dropped_rows=122, excluded_fraction=122/8350, models_retrained=False,
                      thresholds_changed=False, same_retained_rows_for_all_models=True,
                      dropped_correct={}, dropped_errors={}, retained_correct={}, retained_errors={})
    for arm in ARMS:
        correct = (frame[f"prediction_{arm}"] == frame.label)
        dropped_correct = int((dropped[f"prediction_{arm}"] == dropped.label).sum())
        retained_correct = int((retained[f"prediction_{arm}"] == retained.label).sum())
        arithmetic["dropped_correct"][arm] = dropped_correct
        arithmetic["dropped_errors"][arm] = len(dropped) - dropped_correct
        arithmetic["retained_correct"][arm] = retained_correct
        arithmetic["retained_errors"][arm] = len(retained) - retained_correct
        _check(int(correct.sum()) == dropped_correct + retained_correct, "Deletion accounting mismatch")
        _check(retained_correct == old_subset["selected_test_subset"]["correct"][arm], "Existing subset result mismatch")
        _check(abs(metrics_by_scope["full8350"]["validation_accuracy"][arm]["accuracy"] -
                   original["metrics"][arm]["accuracy"]) < 1e-12, "Existing full result mismatch")
    arithmetic.update(full_net_correct=51, dropped_rescues=0, dropped_harms=122,
                      retained_net_correct=173, formula="retained_net = full_net - dropped_rescues + dropped_harms = 51 - 0 + 122 = 173",
                      gain_formula="100 * 173 / 8228 = 2.1025765678 pp",
                      interpretation="The subset gain is mechanically favored by removing this old model's harmful transitions; it is descriptive, not unbiased method evidence.")
    full_pair = next(c for c in comparisons if c["scope"] == "full8350" and
                     c["threshold_mode"] == "validation_accuracy" and c["baseline"] == "best_single" and
                     c["candidate"] == "fusion_fixed_best_anchor")
    subset_pair = next(c for c in comparisons if c["scope"] == "subset8228" and
                       c["threshold_mode"] == "validation_accuracy" and c["baseline"] == "best_single" and
                       c["candidate"] == "fusion_fixed_best_anchor")
    _check((full_pair["rescued"], full_pair["harmed"], subset_pair["rescued"], subset_pair["harmed"]) ==
           (177, 126, 177, 4), "Primary error transition accounting mismatch")

    threshold_rows = []
    for scope in SCOPES:
        for arm in ARMS:
            chosen, fixed = (metrics_by_scope[scope][mode][arm] for mode in ("validation_accuracy", "fixed_0.5"))
            threshold_rows.append(dict(**disclosure(scope), method=arm, validation_threshold=chosen["threshold"],
                fixed_threshold=.5, validation_threshold_accuracy=chosen["accuracy"], fixed_threshold_accuracy=fixed["accuracy"],
                accuracy_delta_pp=(chosen["accuracy"] - fixed["accuracy"]) * 100,
                validation_threshold_f1=chosen["f1"], fixed_threshold_f1=fixed["f1"],
                validation_threshold_fpr=chosen["fpr"], fixed_threshold_fpr=fixed["fpr"]))
    probe_rows = [dict(**disclosure("shared_train_validation"), layer=layer,
                       diagnostic="independent_same_capacity_MLP_validation_BCE_not_linear_probe_or_test_score",
                       validation_bce=bce, best_epoch=selection["layer_best_epochs"][layer-1],
                       selected_best_single=layer == selection["best_single_layer"],
                       selection_rule="minimum validation BCE; lowest layer breaks ties")
                  for layer, bce in enumerate(selection["layer_validation_bce"], 1)]

    lora_path = args.lora_study.resolve() / "readouts/analysis/results.json"
    lora = read(lora_path)
    _check(lora["rows"] == 8228 and lora["subset_selection_policy"] == POLICY and lora["use_lora"],
           "Unexpected LoRA supplementary result scope")
    lora_summary = dict(**disclosure("subset8228"), use_lora=True, complete=True,
        metrics=lora["metrics"], adapter_sha256=lora["adapter_sha256"],
        anchor_vs_best_single_gain_pp=100*(lora["metrics"]["fusion_fixed_best_anchor"]["accuracy"]-
                                         lora["metrics"]["best_single"]["accuracy"]),
        scope_note="Supplementary shared frozen-LoRA comparison; no-LoRA reference exclusions stay fixed")
    sources = [Path(__file__).resolve(), ROOT / "scripts/hanguard/frozen_lora_layer_diagnostics.py"]
    inputs = [protocol_path, selection_path, result_path, predictions_path, policy_path, ids_path,
              subset / "old_model_subset_metrics.json", subset / "manifest.json", lora_path]
    inputs += [base / f"{split}.parquet" for base in (parent, subset) for split in ("train", "validation", "test")]
    input_sha = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    source_sha = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    result = dict(schema="hanguard_nolora_reference_analysis_v1", complete=True, cpu_only=True,
        subset_selection_policy=POLICY, blind_test=False, is_quality_cleanup=False, seed=42,
        method_selection="All five heads, layer/checkpoint/threshold selected using unchanged validation only",
        subset_selection="122 test IDs selected using old best_single-correct and anchor-wrong predictions; fixed across all methods",
        metrics=metrics_by_scope, comparisons=comparisons, deletion_arithmetic=arithmetic,
        parameters=original["parameters"], layer_selection=selection["best_single_layer"],
        lora_supplementary=lora_summary, input_sha256=input_sha, source_sha256=source_sha,
        bootstrap=dict(draws=2000, seed=42, unit="group_id cluster", selection_bias_corrected=False),
        completion=dict(existing_core_five_arms="complete_seed42", lora_comparison="complete_seed42",
                        additional_seeds_43_44="pending", six_component_ablations="pending",
                        component_names=["uniform_weights", "no_anchor", "shared_projection", "top4", "two_layers", "mean_pool"]),
        limitations=["The +2.1026 pp selected-set gain is outcome-selected and cannot be presented as unbiased test improvement.",
                     "Full-release +0.6108 pp must accompany the selected-set result.",
                     "One head seed is not cross-seed replication; current outputs do not include new seeds/components.",
                     "Conditional bootstrap intervals do not correct test-set selection or prior exploratory reuse."])
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(sources_rows).to_csv(output / "source_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output / "threshold_sensitivity.csv", index=False)
    error_rows = []
    for item in comparisons:
        row = {k: v for k, v in item.items() if k not in ("group_bootstrap", "accuracy_ci95_pp")}
        row.update(accuracy_ci95_low_pp=item["accuracy_ci95_pp"][0], accuracy_ci95_high_pp=item["accuracy_ci95_pp"][1],
                   bootstrap_draws=2000, bootstrap_seed=42, bootstrap_selection_bias_corrected=False)
        error_rows.append(row)
    pd.DataFrame(error_rows).to_csv(output / "error_complementarity.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(output / "layer_probe.csv", index=False)
    dump(output / "main_results.json", result)
    report = ["# 无 LoRA 融合结果与已有消融证据", "",
        f"子集政策：`{POLICY}`。本文档是已有预测的 CPU 复算，没有重新训练或改变阈值。", "",
        "固定8228子集由旧模型的测试错误选择，122个排除ID对五种方法完全相同。它不是独立盲测，也不是已证明的数据质量修复。", "",
        "| 方法 | 原始8350准确率 | 固定8228准确率 | 参数量 |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        full = metrics_by_scope["full8350"]["validation_accuracy"][arm]["accuracy"] * 100
        kept = metrics_by_scope["subset8228"]["validation_accuracy"][arm]["accuracy"] * 100
        report.append(f"| {arm} | {full:.4f}% | {kept:.4f}% | {original['parameters'][arm]:,} |")
    report += ["", f"锚点融合相对最佳单层：原完整测试 **+{full_pair['accuracy_gain_pp']:.4f}个百分点**，"
        f"条件组bootstrap 95%区间 [{full_pair['accuracy_ci95_pp'][0]:+.4f}, {full_pair['accuracy_ci95_pp'][1]:+.4f}]个百分点；"
        f"固定子集 **+{subset_pair['accuracy_gain_pp']:.4f}个百分点**，"
        f"条件区间 [{subset_pair['accuracy_ci95_pp'][0]:+.4f}, {subset_pair['accuracy_ci95_pp'][1]:+.4f}]个百分点。"
        "子集区间不校正按结果筛选造成的偏差。", "",
        "完整测试救回177条、新增126条错误，净增51条。排除122条只移除了旧锚点融合新增的错误，保留救回177条、新增错误4条："
        "净增51+122=173条，173/8228×100=2.1026个百分点。因此该数字是事后子集上的描述性结果，不能宣称无偏的总体收益。", "",
        "现有核心5臂已完成：最后层、32层独立MLP中验证选择的最佳单层、容量匹配重复单层、固定四层融合、最佳层锚点融合。"
        "三个大容量臂均为1,656,709参数且初始化一致；单层MLP为327,937参数。最佳层为L19，四条融合支路为L8/L16/L24/L32。", "",
        "阈值0.5和验证集最大准确率阈值已同时复算；来源分组、组bootstrap、救回/新增错误见CSV。"
        "layer_probe.csv仅含32层独立MLP的验证BCE与最佳轮次，没有虚构线性probe或逐层测试结果。", "",
        "新种子43/44与6个组件消融尚待运行。本目录不能作为它们已经完成的证据。保持同一训练/验证划分、固定122个排除ID，"
        "新实验均应同时报告原始8350与固定8228两个范围。", "",
        "六组件为uniform_weights（等权）、no_anchor（去直接锚点）、shared_projection（共享投影）、top4（验证选4层）、"
        "two_layers（2层）、mean_pool（L8/L16/L24/L32原始特征均值后MLP）。", "",
        f"LoRA补充对照已完成，同一固定8228子集上锚点相对最佳单层 **+{lora_summary['anchor_vs_best_single_gain_pp']:.4f}个百分点**。", "",
        "复现：` .venv-hanguard/bin/python -m scripts.hanguard.analyze_fusion_reference_results `。输入与分析源码SHA256见main_results.json；"
        "输出清单与SHA256见artifact_manifest.json。", ""]
    (output / "report.md").write_text("\n".join(report))
    filenames = ("main_results.json", "metrics.csv", "source_metrics.csv", "threshold_sensitivity.csv",
                 "error_complementarity.csv", "layer_probe.csv", "report.md")
    dump(output / "artifact_manifest.json", dict(schema="hanguard_nolora_reference_analysis_artifacts_v1",
         subset_selection_policy=POLICY, artifacts={name: sha256(output / name) for name in filenames},
         input_sha256=input_sha, source_sha256=source_sha))
    print(json.dumps(dict(output=str(output), complete=True, rows={"full8350":len(frame), "subset8228":len(retained)},
                         metrics_rows=len(metrics_rows), source_rows=len(sources_rows), comparisons=len(comparisons),
                         anchor_gains_pp={"full8350":full_pair["accuracy_gain_pp"], "subset8228":subset_pair["accuracy_gain_pp"]})))


if __name__ == "__main__":
    main()
