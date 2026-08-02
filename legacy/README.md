# Historical baseline

`original_baseline/` contains the project files as received. They expect data
files that are not part of the current reproducible workflow and repeatedly run
full-graph convolution for each mini-batch. Keep them only for source-history
comparison; do not use them for new experiments.

Use `python run.py train --model rgcn` for the maintained R-GCN baseline.
