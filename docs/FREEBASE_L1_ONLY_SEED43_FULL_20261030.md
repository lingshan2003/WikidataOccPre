# Freebase L1-only seed43: full 1900–2000 replication

Research question: does the seed42 annual relation-group pattern persist under
seed43, after the two-window multi-seed audit showed sensitivity?

Use the same frozen source, 101 annual ±20-year life windows, five relation
groups, occupation L1-only input and L1 target, no auxiliary features, and
within-window split seed 20260814. Birth >1920 without death extends membership
through 2026. Global seed43 controls RGCN, GraphMask and neighbor sampling.
RGCN remains max50 epochs with early stopping; GraphMask remains3 epochs/layer,
beta .03, validation relative Macro-F1 drop≤.05 and any-stage checkpoint selection.
Evaluate test accuracy/Macro-F1, prediction agreement, overall/layer retention
and Layer0 group retained-message shares. No new model-selection rule.

```bash
bash Freebase/run_sliding_l1_only_seed43.sh run all --gpus 0,3,4,5
```

GPU IDs must be chosen from current occupancy immediately before launch.
The launcher writes a Git revision, resolved config, code snapshot, logs and
exit status under `artifacts/l1_only_sliding_launches/freebase/`.

The existing seed43 1940/1942 models and probes are reused only after their
existing stage contracts match. Their original roots are deliberately retained
so commands, input stamps and checkpoint references remain unchanged:

- Models: `runs/freebase_l1_only_repeat_1940_1942_seed43_run1_v1`
- Probes/reports: `runs_graphmask/freebase_l1_only_repeat_1940_1942_20261030_v1/seed43_run1`

Before expansion, preserve the old two-window matrices and resolved config in
`before_101_window_expansion_20261030/`. A human-readable result entry may link
`runs_graphmask/freebase_grouped_sliding_1900_2000_pm20_step1_l1_only_seed43_alive2026_v1`
to the report root above. The config uses real original roots; this link is not
required for stage reuse. Never rewrite contracts to manufacture compatibility.

The other99 windows train from scratch. Compatible completed stages can resume;
interrupted optimizer/epoch state is not resumed. Seed42 and other-seed results
remain separate. Run inside shared tmux so SSH disconnection does not stop it.
