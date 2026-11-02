# Global101-window shared GraphMask control

User explicitly authorized this global comparison after the completed decadal-sharing experiment. Scope: Freebase101 centers1900–2000, each retaining its own existing L1-only RGCN42 frozen; one shared GraphMask across all101. Existing independent and decadal reports remain unchanged.

Configuration: config/freebase_global_shared_graphmask_1900_2000_rgcn42_gm42_order42_v1.json. Same trainer and independent RNG protocol as the decadal control: GraphMask initialization42, randomized per-round window order42, validation100042/report42, independent dual variables for every window, KLbeta.03, LR1e-4/dualLR.01. No temporal penalty. Six joint epochs, three per layer. Each window uses min(full32-root-batch counts across101) batches per epoch (expected531). Expected321786 optimizer updates. Actual count recorded. This equalizes windows within the global run; decadal runs used their own group minimums and therefore different per-window training exposure. Report that budget difference rather than claim an exact budget-matched comparison.

Selection is any-stage: prefer all101 validation F1 relative drops<=5%, otherwise all<=10%, minimize mean hard retention within tier. No feasible checkpoint is reported as failure, without silent relaxation; report disabled Layer0 if selected. RGCNs are never retrained or updated. Same shared gate and replacement vectors, separate native source-provenance report envelopes for101 models.

Run in persistent shared tmux via wywikidata Python and CUDA_VISIBLE_DEVICES=<free card>:
`bash Freebase/run_shared_graphmask_1900_1909.sh --config config/freebase_global_shared_graphmask_1900_2000_rgcn42_gm42_order42_v1.json`.
The wrapper name is historical; the config determines the101-window scope. Use --preflight-only before launch to check all source checkpoints, features, mappings, dimensions and frozen parameters; it prints memory usage. Locks reject duplicate runs. Epoch-boundary checkpoints allow --resume after an interrupted process has exited. Logs, manifest, source paths, Git revision, config, status, validation/history and per-year test reports are in runs_graphmask/freebase_global_shared_graphmask_1900_2000_rgcn42_gm42_order42_v1.

Compare global, decadal and independent masks on raw per-year Layer0 shares, conditional group retention, overall sparsity and per-window fidelity; investigate any remaining spikes. Do not select checkpoints for curve smoothness. No additional seed repetitions or dataset experiments are authorized by this configuration.
