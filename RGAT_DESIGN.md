# Relation-aware GAT design

This is the new starting point for the occupation-prediction experiment.  The
old `main.py` remains an R-GCN baseline and does not yet invoke this model.

## Data contract

`ExtendedGraphLoader` turns the edge-centric CSV into:

- `nodes`: exactly one row per person, with country, birth/death and all three
  occupation levels;
- `edges`: `source_id`, `relation_id`, `target_id`, plus an explicit reverse
  relation for every original edge.

The loader records inconsistent copies of a person's repeated attributes in
`attribute_conflicts`.  Review this table instead of silently assuming every
copy is identical.

The extended export contains structured values, not natural-language text.
Country is an embedding lookup. Birth/death become standardised numeric
features plus missing indicators. A language-model embedding is only warranted
when a future export has genuine text, such as biographies or descriptions.

## Safe target construction

For the Level-3 task, the target person's full occupation hierarchy must not
be supplied as a feature for seed nodes whose loss is evaluated:

- training: replace each sampled seed's Level-1, Level-2 and Level-3 inputs
  with their explicit unknown IDs;
- validation/test: replace every held-out person's three occupation levels with
  unknown IDs throughout training and evaluation;
- calculate validation/test labels from the unmasked canonical `nodes` table,
  never from the masked feature table.

Known occupations of *other* people are a valid transductive feature only if
they would genuinely be known at inference time. State that assumption in the
experiment report.

## Model

`RelationalGATClassifier` encodes each registered feature in a separate branch,
learns a per-node feature gate, fuses the branches, then applies a residual
relation-aware GAT stack.  New attributes are registered with `FeatureSpec`:

```python
feature_specs = {
    "occupation_level1": FeatureSpec(kind="categorical", cardinality=num_level1_occupations),
    "occupation_level2": FeatureSpec(kind="categorical", cardinality=num_level2_occupations),
    "occupation_level3": FeatureSpec(kind="categorical", cardinality=num_level3_occupations),
    "country": FeatureSpec(kind="categorical", cardinality=num_countries),
    "temporal": FeatureSpec(kind="numeric", input_dim=6),
    # Later: "biography": FeatureSpec(kind="vector", input_dim=768, optional=True),
}
```

The default data tensors are all three occupation levels plus `country` and
`temporal`. Use `--occupation-feature-levels 3` for the Level-3-only neighbour
occupation baseline. Do not provide the target person's own L1/L2 as a shortcut
when the research task is social-neighbour occupation prediction.

## Preparation command

Run the following once before training. It creates `artifacts/graph_data.pt`
plus inspectable CSV/JSON audit artifacts. The default experiment includes
known training neighbours' three-level occupation information under the masking
protocol above.

```bash
python run.py prepare --input Q_R_Q_extended.txt --output-dir artifacts
```

## Training command

After preparation, train a two-layer R-GAT with sampled two-hop neighborhoods:

```bash
python run.py train --model rgat --data artifacts/graph_data.pt --output-dir runs/rgat_level3 \
  --num-neighbors 20,10 --batch-size 512 --epochs 50 --num-workers 4
```

The training loop defaults to validation-loss checkpoint selection and early
stopping (patience 6, minimum loss improvement 0.002). It halves the learning
rate after three bad validation-loss epochs, then evaluates exactly once on the
test nodes. Pass `--early-stop-metric macro_f1` only when macro-F1 is the
explicit selection objective; do not compare the resulting test score against a
loss-selected run as if they used the same protocol.

## Attention export

Export R-GAT attention candidates for one person after training. The script
writes all sampled edges, the top incoming edges for the queried person, and a
prediction summary with feature-fusion gates:

```bash
python run.py explain --data artifacts/graph_data.pt \
  --checkpoint runs/rgat_level3/best_model.pt --node-id Q1000023 \
  --output-dir explanations
```

Attention does not establish causal edge importance. Treat the top-edge CSV as
the candidate set for a later deletion-faithfulness experiment.

## Sampling and explanation

Use two-hop neighbor sampling, with loss calculated only on seed nodes. A
reasonable first server configuration is 512 seeds and fan-outs `[20, 10]`.
Retain the sampler's global `n_id` mapping for each subgraph. On a requested
forward pass the model returns `alpha` per layer/head, `edge_index`, and
`edge_type`; map these local edges through `n_id` before exporting them.

Rank explanations within each destination node and relation-aware sampled
neighborhood. Validate any ranked edge by removing it (or a relation group) and
measuring the drop in the predicted target-class logit. Attention alone is not
a causal importance claim.
