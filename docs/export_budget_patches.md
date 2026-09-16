# Export patches at saved search budgets

Run from the `swesearch` directory, pointing to the experiment directory containing
`evaluation.json` and `<instance_id>/trajectory.json`:

```sh
poetry run python -m moatless.benchmark.export_predictions evaluations/verified.test.b2 --max-iterations 11 21 31 41 51 --include-empty
```

Replace `evaluations/verified.test.b2` with the actual experiment directory.
This writes `predictions.iterations_11.json`, `predictions.iterations_21.json`, etc.
Each file is a SWE-bench predictions JSON array with `instance_id`,
`model_name_or_path`, and `model_patch`. `--include-empty` retains instances whose
selected patch is empty, which is useful for comparing the same evaluation set.
Each cutoff displays a tqdm progress bar with the processed instance count,
elapsed time, and estimated time remaining. Use `--no-progress` to disable it.

For any single root-inclusive cutoff and a custom output path:

```sh
poetry run python -m moatless.benchmark.export_predictions evaluations/verified.test.b2 --max-iterations 30 --output predictions.at30.json --include-empty
```

`max_iterations` counts the root. Your run with `--max-iterations 51` allows 50
new nodes. For 10/20/30/40 expansions use 11/21/31/41; for literal
`--max-iterations 10/20/30/40` use those exact numbers. The branching limit stays
at the original `--max-expansions 2`.

The exporter prunes by node creation order, restores leaves, rebuilds accumulated
reward and visit counts using only retained nodes, then runs the existing
discriminator and patch generator. Finished nodes are preferred; otherwise the
selector uses leaves, so early patches may be incomplete or empty. Final saved
submissions are ignored when a cutoff is requested. Without the cutoff option,
the original final-submission export behavior is unchanged.

This is reconstruction of a prefix of the saved run, assuming nodes were expanded
once in increasing ID order. A final trajectory cannot recover overwritten states
or repeated reward evaluations from retries. A shorter independent run may also
differ if its policy uses the configured budget. Requests beyond the original
configured limit are rejected; if fewer nodes were saved than the requested
cutoff, a warning is emitted and all available nodes are used. This may indicate
early stopping or an interrupted run, not a completed run at that budget.

No search or benchmark tests are rerun by exporting. Evaluate the output files
with the SWE-bench harness separately to measure correctness.

Regression checks:

```sh
poetry run pytest tests/benchmark/test_export_predictions.py -q
```
