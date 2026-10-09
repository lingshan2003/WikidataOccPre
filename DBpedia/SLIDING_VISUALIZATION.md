# Annual relation-group figures

Use the completed multi-group annual experiment. Each of the seven groups has
one raw annual curve, with a seven-panel overview. Default center years are
1901–2000 (the twentieth century); 1900 remains in the downloaded bundle.
Window bounds still extend twenty years on either side of each center.

The y-axis is **Layer 0 retained message share**:

`group hard-retained sampled messages / all hard-retained sampled messages in Layer 0`.

Original and generated reverse messages are combined by base group. Seven
accounting shares sum to 100% per year. These are sampled message observations,
not unique graph edges, group-internal retention rates or the reference figure's
OII. No bootstrap, confidence interval, smoothing or sample-size comparison is
performed. Each panel starts at zero and uses its own y-axis scale.

Checkpoint selection remains the experiment's existing `any-stage` policy.
Years with Layer 0 disabled are included and marked by open circles. Missing
sampled support is shown as a gap rather than evidence of zero group retention;
its accounting share is zero in the exported table. A sampled group with all its
messages removed has a genuine plotted zero. An empty annual denominator or a
missing/incomplete annual result fails validation rather than being imputed.

## Server: download lightweight reports

After pushing these scripts locally, run in the server repository:

```bash
git pull
python DBpedia/package_sliding_results.py
```

The packager uses the standard library and needs no GPU or PyTorch. It packages
101 complete windows, summaries, taxonomy/calendar, RGCN metrics, GraphMask
reports, checkpoint-selection metadata and small graph statistics. It excludes
weights, graph tensors, node/edge tables, root-level edge reports and logs. The
archive is placed in ignored `artifacts/` to avoid a tracked root archive blocking
future `git pull`. It refuses to overwrite an existing archive; use `--output`
with a new filename for later exports. Use `--config` for another window run.

Run **on the local computer**, using the same SSH host or alias used previously:

```bash
scp siruilai@dingqiyang:/mnt/network_data/personal_workspace/siruilai/wy/WikidataOccPre/artifacts/dbpedia_sliding_1900_2000_reports.tar.gz /Users/wangyue/RGCN/artifacts/
```

If `dingqiyang` is not resolvable locally, substitute the actual server IP/SSH
alias and add the existing port or jump-host options to `scp`.

## Local: reproducible plotting

```bash
mkdir -p artifacts/dbpedia_sliding_1900_2000_received
tar -xzf artifacts/dbpedia_sliding_1900_2000_reports.tar.gz -C artifacts/dbpedia_sliding_1900_2000_received
.venv/bin/python DBpedia/plot_sliding_relation_groups.py \
  --received-root artifacts/dbpedia_sliding_1900_2000_received \
  --output-dir visualization/dbpedia_sliding_layer0_20th_century
```

Plot dependencies: Python >= 3.9 and Matplotlib. No PyTorch or model loading.
The plotting script checks test roots, sampling seed/fanouts, fidelity threshold,
selected checkpoint against training history, per-layer/group message counts,
group shares and their denominator. It stops if any requested year is missing.
Downloaded absolute server paths in original manifests are retained as provenance;
the bundle also provides normalized relative paths for local access.

Outputs:

- `layer0_<group>_annual.png`, `.svg`, `.pdf`: seven separate figures.
- `layer0_all_groups_annual.png`, `.svg`, `.pdf`: seven-panel overview.
- `layer0_group_annual.tsv`: full-precision annual shares and support counts.
- `annual_audit.tsv`: selected epochs, Layer 0 status, retention, agreement,
  masked accuracy and macro F1.
- `provenance.json`: source configuration, taxonomy and metric definition.

To include center year 1900 explicitly, pass `--start-year 1900`; other ranges
are available with `--start-year` and `--end-year`. The range must have a complete
report for every integer year.

The package/plot workflow was checked with the existing four-window CPU smoke
reports; those smoke curves are not scientific results and are not delivered as
the completed server experiment's figures. Formal figures require its download.
