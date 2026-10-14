# DBpedia BHHT L1 sliding experiment v1

User confirmed on 2026-10-14: use the provisional five-class L1 dataset,
without temporal/country auxiliary features. Run the three pilot centers
1900/1950/2000 after the current Wikidata experiment succeeds, validate them,
then expand to all 101 centers on GPUs 1,2,4.

- Research question: how do eight DBpedia relation groups contribute to
  hard-retained Layer 0 sampled messages across annual twentieth-century windows?
- Dataset: `external_data/dbpedia/bhht_l1_provisional_v1`; 55,690 people,
  67,090 original directed facts, 33 original predicates, five L1 classes.
  Other contains 680 formal labels plus 597 provisional fallback labels.
  L2/L3 remain empty; audit states and original dates are retained.
- Windows: inclusive center +/-20 years, centers 1900--2000, stride one.
  Birth >1920 with missing death extends membership through 2026; it does
  not replace the raw missing death value. Use induced edges and retain
  member nodes even when isolated.
- Relations: eight groups in `dbpedia_tie_taxonomy_acquired_subgroups_v2.json`,
  with Influence and Succession separate; reverse predicates are generated
  by the existing preparation protocol. Collapse identical grouped triples.
- Model: L1-only occupation inputs, L1 target, no auxiliary features;
  training neighbor occupations are visible, validation/test occupations are
  unknown, and current seed node occupations are masked by the shared trainer.
- RGCN: seed 42, two layers, fast backend, 16 bases, hidden 128/branch 64,
  dropout 0.2, sampled fanouts 15,10, batch 512, workers 4, Adam lr 0.001,
  weight decay 0.0001, maximum 50 epochs, validation Macro-F1 early stopping
  with patience 6/minimum improvement 0.001.
- Split: stratified within-window 70/10/20, split seed 20260814,
  local minimum class count 3. Source preparation uses seed 42 and minimum
  global class count 20. Do not reuse old 82-class DBpedia checkpoints.
- GraphMask: 3 epochs per layer, batch 32, fanouts inherited from RGCN,
  beta 0.03, learning rate 0.0001, multiplier learning rate 0.01,
  temperature 1/3, location bias 3.0; validation relative Macro-F1
  difference threshold 0.05. Checkpoints can be selected at any stage;
  Layer 0 need not have been enabled in the selected checkpoint.
- Control: the unmasked RGCN on the same test roots. Report accuracy,
  Macro-F1, prediction agreement, global and per-layer hard retention,
  and each group's share of all retained Layer 0 sampled messages.
  These are sampled message statistics rather than unique original-edge counts.
  This single-seed run does not estimate uncertainty with bootstrapping.
- Output roots and all preparation stages are isolated under `bhht_l1` names;
  old datasets, running Wikidata jobs and earlier DBpedia outputs are retained.
- Recovery reuses compatible completed stages. An interrupted unfinished
  RGCN/GraphMask stage restarts; there is no optimizer/epoch resume.

Entry point:

```bash
export DBPEDIA_PYTHON_BIN=/home/dingqiyang/anaconda3/envs/wywikidata/bin/python
bash DBpedia/run_sliding_bhht_l1_v1.sh plan all --gpus 1,2,4
```

Durable queue, in an idle dedicated tmux window:

```bash
python DBpedia/queue_bhht_l1_after_wikidata.py \
  --python "$DBPEDIA_PYTHON_BIN" --gpus 1,2,4 \
  --output-dir artifacts/dbpedia_bhht_l1_queue/20261014_v1
```

The queue requires successful exit, 101 successful dispatch events, and 101
complete predecessor reports. It checks dependencies, free disk space,
immutable inputs/code, and at least 4 GiB free on each selected GPU before
each phase. Utilization is recorded because the server is shared. Insufficient
GPU memory waits; failures stop expansion. The pilot validates exact membership,
raw attributes, induced edges, actual model inputs/classes, fidelity, test roots,
Layer 0 shares and completed-stage provenance before expansion.

Queue `status.json`, per-phase command/log/exit files, GPU telemetry and the
dispatcher manifests preserve the actual deployed Git revision and settings.
The queue stays alive independently of SSH in tmux. Its outer log must be
outside its initially empty output directory. A Git/code change while queued
stops launch to prevent silently changing the confirmed experiment.

Results:
`runs_graphmask/dbpedia_grouped_sliding_1900_2000_pm20_step1_bhht_l1_alive2026_v1/`
contains `matrix_summary.tsv`, `relation_group_summary.tsv`,
`pilot_validation.json` and each center's `multi_group/seed_42` report.
