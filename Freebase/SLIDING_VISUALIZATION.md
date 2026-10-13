# Freebase annual Layer 0 figures

Use the completed `born_ge1920_alive2026_v3` multi-group experiment, not the
two-window CPU smoke reports. Default center years are 1901–2000; the report
bundle retains all 101 years, including 1900. Windows are inclusive ±20 years.

The y-axis is each group's **hard-retained sampled Layer 0 messages divided by
all hard-retained sampled Layer 0 messages**. Original and generated reverse
messages are combined. The five accounting shares sum to 100% per year.
These are message composition shares, not within-group retention rates, unique
edge proportions, OII, or causal effects. No bootstrap, confidence intervals,
smoothing, or additional training is used.

The five groups are inherited, intimate partnership, education / mentorship,
influence, and other acquired ties. The stored `influence_succession` key only
contains `influenced_by`, so its display label is **Influence**. Freebase does
not contain the DBpedia succession group. Their taxonomy scopes differ.

Checkpoint selection follows the experiment's `any-stage` default. An open
circle marks a year whose selected checkpoint has Layer 0 gating disabled.
That year's point remains in the curve. No sampled support is a gap with zero
accounting share; an observed group with no retained messages is a plotted zero.
Panels start at zero and use independent y-axis scales.

Freebase extends people born **≥1920** with truly missing death records through
2026 for graph membership only. DBpedia's corresponding v3 uses **>1920**.
Raw death features remain missing. Relations have no event-date filter, and
overlapping windows with separate splits are not independent replications.

## Download without writing files on the server

After the user establishes a password-authenticated SSH ControlMaster connection
while on the university VPN, run this command **locally**. Substitute its actual
control socket path. The exporter executes the local standard-library script in
memory remotely and streams small reports directly into a local archive. It
does not deploy code, edit server files, or load graphs/model weights.

```bash
python3 scripts/download_sliding_reports.py \
  --host siruilai@10.113.178.158 \
  --control-path /actual/ssh/control/socket \
  --remote-root /mnt/network_data/personal_workspace/siruilai/wy/WikidataOccPre \
  --config runs/freebase_grouped_sliding_1900_2000_pm20_step1_born_ge1920_alive2026_v3/grouped_pipeline_resolved_config.json \
  --output artifacts/freebase_sliding_alive2026_v3_reports.tar.gz
```

The actual resolved configuration preserves any training overrides. Downloads
refuse to overwrite an existing archive. The exporter requires all configured
multi-group windows to be complete and pipeline failures to be empty.

Alternatively, after syncing code through Git, explicitly create an archive in
your own server workspace using `python Freebase/package_sliding_results.py
--config runs/.../grouped_pipeline_resolved_config.json --output artifacts/...`.
Then download it using scp. This alternative writes an archive on the server.

## Plot locally

Extract the trusted report archive into a new directory, then run:

```bash
.venv/bin/python Freebase/plot_sliding_relation_groups.py \
  --received-root artifacts/freebase_sliding_alive2026_v3_received \
  --output-dir visualization/freebase_sliding_layer0_20th_century
```

The entry point shares the audited renderer with DBpedia. Validation checks
test roots, sampling seed/fanouts, fidelity, selected checkpoint against training
history, Layer 0 counts, group support, shares and denominators. Outputs are five
separate figures and an overview in PNG/SVG/PDF, plus `layer0_group_annual.tsv`,
`annual_audit.tsv`, and `provenance.json`. Matplotlib is the only plotting
dependency; no PyTorch or weights are needed. The source scripts are retained.
