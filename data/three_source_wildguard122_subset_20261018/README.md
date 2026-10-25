# WildGuard 122-row posthoc test subset

This derived release excludes exactly 122 WildGuard test rows on which the old no-LoRA best_single was correct and fusion_fixed_best_anchor was wrong. It is an outcome-dependent exploratory evaluation subset, not data-quality remediation or a new blind benchmark. Train and validation are byte-identical to the parent; retained text, labels, order and split assignments are unchanged.

Both old models are evaluated on the same retained test rows. See `old_model_subset_metrics.json`, `filter_policy.json` and `excluded122.csv` for the full selection lineage and descriptive metrics.
