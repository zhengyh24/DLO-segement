# Full WireSeg loss reporting

This version preserves the pilot model and loss mathematics. The added reporting
is detached from backpropagation. Original pilot source remains frozen separately.

Each optimizer opportunity writes the seven raw criterion components and `total`,
their weighted contributions, successful-update count and AMP-skip status to
`logs.jsonl`. `loss_spec.json` records coefficients and inactive components.
Epoch training means weight each batch's criterion value by its number of images;
all opportunities, including any AMP skips, remain visible. Batch size 1 makes this
an exact mean over training images. For larger batch sizes, the criterion's object
loss is normalized within the batch, so it must be called a sample-weighted batch
mean, not an independently recomputed per-image mean.

Every validation epoch computes the same criterion from the same RGB-only forward
already used for segmentation metrics, without a second encoder or decoder call.
GT masks and overlap-validity enter the loss after inference only. The report saves
each image's raw/weighted losses and the arithmetic mean across the full split.
If any image cannot be scored, `losses.complete=false`, the primary mean is null,
and the explicitly named `observed_mean` remains available for diagnosis. Plotting
does not fill in missing validation values or represent successful-only means as
complete validation results. Existing checkpoint selection remains validation
instance IoU then PQ; loss curves do not tune or select on test results.

The epoch accumulator is part of checkpoint progress, so exact mid-epoch resume
preserves the full-epoch means. Curve generation follows the verified run's parent
path and clips parent log observations to the saved cursor. Keep parent logs when
relocating resumed runs, or recreate their recorded path; missing parents fail
explicitly instead of silently presenting a partial history.

## Render charts

The plotting process needs NumPy and Matplotlib (tested with Matplotlib 3.10.6).
It can run in a separate plotting environment; no Matplotlib import is added to
the training process. Use a new output directory for every curve snapshot:

```bash
python sam_path_v1/plot.py --runs B=/absolute/runs/B C=/absolute/runs/C \
  --output /absolute/curves/final --ema-span 200
```

One run is also accepted. For each of `total`, `mask_bce`, `mask_dice`, `object`,
`quality`, `path`, `descriptor` and `geometry`, the output contains a PNG and PDF.
These show all original training-step values with a causal EMA overlay and the
completed-epoch train/validation means. Non-total components also show the actual
weighted contribution. `epoch_overview.png/pdf` summarizes all eight curves.

Each run has `steps.csv` (all observations), `epochs.csv` (complete epoch means),
and `metrics.csv` (validation segmentation metrics). `curve_receipt.json` documents
sources, EMA span, CSV hash, observed step counts, skipped updates and incomplete
validation epochs. A partially written last JSON line is ignored during live
rendering; malformed complete lines, missing/duplicate steps, nonfinite values,
or components that do not reconstruct the logged total are errors.

B has no descriptor or geometry loss: those components are marked inactive and
are not plotted as fictitious observations. C optimizes additional objectives;
B/C total-loss magnitude is not a comparison of segmentation quality. Compare
common loss components and the separately reported segmentation metrics.

CLI evaluation can optionally include losses with `evaluate --losses`. Final
held-out metrics should still be reported for both validation-selected `best.pt`
and `last.pt` with the checkpoint's frozen inference thresholds.
