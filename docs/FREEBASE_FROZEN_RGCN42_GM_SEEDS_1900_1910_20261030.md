# Freebase: isolate GraphMask training randomness, 1900–1910

Research question: with each window's exact original L1-only seed42 RGCN fixed,
how much do GraphMask seeds43/44 change Layer0 five-group retained-message
shares compared with original GraphMask seed42?

- Scope: 11 centers1900–1910 inclusive, the original±20-year windows.
- Reuse the original seed42 model/data/taxonomy and split20260814. No RGCN
  training, data rebuilding, model-weight copying or checkpoint rewriting.
- Original GraphMask42 reports are the baseline. They used validation sampling
  seed100042 and final report sampling seed42. Hold both fixed for new probes.
- New seeds43/44 control only GraphMask initialization, training-root order,
  training-neighbor sampling and stochastic gates. Each layer gets3 epochs,
  beta .03, the original optimizer settings and validation relative Macro-F1
  drop≤.05. Select the sparsest faithful any-stage checkpoint, as before.
- The baseline training modules remain untouched because a separate full
  Freebase seed43 experiment may still be running. A process-local wrapper
  fixes validation sampling, records it, and asserts loaded model parameters
  have requires_grad=False. Native optimization updates probe parameters only.
- Exclude GPU2. Select available cards only from0/1/3/4/5 at dispatch time,
  utilization<60%, free memory≥8GiB, at most4 simultaneous probe jobs.

```bash
bash Freebase/run_frozen_rgcn_graphmask_seeds.sh run
```

Output root:
`runs_graphmask/freebase_frozen_rgcn42_gm_seeds43_44_1900_1910_20261030_v1`

The controller owns only its experiment lock and original seed42 model lock.
It does not acquire or mutate the shared period/relation outputs used by the
ongoing seed43 full experiment. Every worker verifies original stage contracts
and executes graphmask only. New probes have independent per-seed/per-year
directories; compatible stages of the same probe can resume after interruption.
Logs, per-job exit codes, config, Git version and source snapshots are retained.

After22 new jobs finish, summarize33 total baseline/new reports:

```bash
bash Freebase/run_frozen_rgcn_graphmask_seeds.sh summarize
```

This writes `all_relations.tsv` (165 rows) and `comparison_audit.json`. Check
recorded source checkpoint/data fingerprints agree across seeds for each
window, report sampling=42, identical sampled-message counts and validation
fidelity; do not calculate additional weight/data hashes. Keep Layer0-disabled
selected checkpoints visible rather than silently excluding them.

After downloading small reports (no weights), plot locally:

```bash
.venv/bin/python Freebase/plot_frozen_rgcn_graphmask_seeds.py \
  --input-tsv <received>/all_relations.tsv --output-dir <output>/figures
```

Plots: same-relation overlays for seeds42/43/44, five relation panels and
individual relation figures, plus one panel per seed with five group curves.
Raw annual values, no bootstrap or smoothing. User requested checks every10
minutes after launch, until completion and delivery of figures.
