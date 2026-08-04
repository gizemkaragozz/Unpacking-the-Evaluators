# Additional reviewer analyses

This directory contains two analyses performed in response to reviewer questions.
They reuse existing attribution maps and alignment scores; no explanations or
ground-truth annotations are regenerated.

## Retained-support sensitivity of the BMR--LE taxonomy

`taxonomy_support_sensitivity.py` varies the attribution-support fraction used
by both BMR and LE. This fraction is distinct from the thresholds used for hard
alignment metrics. For every operating point, the script calls the same
`boundary_mass_ratio` and `local_spatial_entropy` functions used by the main
analysis, then recomputes method-level coordinates, rankings, and dataset-relative
behavioral regions.

```bash
python additional_analysis/taxonomy_support_sensitivity.py \
  --npy-root <BRISC-attribution-directory> \
  --output-dir <output-directory> \
  --support-fractions 0.05 0.10 0.15 0.20 0.30 \
  --patch-size 16 16 \
  --workers 4 \
  --figures
```

The BRISC2025 analysis contains 11,251 saved attribution maps from all 15 methods
in the BRISC taxonomy panel. At each retained-support level, method-level BMR and
LE means are min--max normalized within the dataset and the quadrant crosshairs
are placed at 0.5. This makes the construction reproducible and dataset-relative;
the quadrant lines are not universal classification thresholds.

Across retained-support fractions of 5%, 10%, 15%, 20%, and 30%, BMR rank
agreement relative to 10% ranges from 0.867 to 1.000, LE rank agreement from
0.846 to 1.000, and dataset-relative quadrant agreement is 93.3% at every
alternative operating point. The only relative-quadrant change is the already
borderline Guided Grad-CAM method. Absolute coordinates change, so the analysis
supports stable relative organization for 14 methods over the evaluated range;
it does not establish threshold invariance or universal categorical boundaries.

Compact committed outputs are under `results/taxonomy_support/`:

- `taxonomy_support_robustness.csv`: rank and label agreement by support fraction.
- `taxonomy_support_method_summary.csv`: method coordinates and labels.
- `taxonomy_method_rank_changes.csv`: per-method BMR and LE ranks and rank shifts
  from the 10% operating point.
- `figures/brisc_normalized_quadrants.*`: normalized BRISC quadrant panels.
- `figures/brisc_bmr_le_changes.*`: method-level descriptor changes from 10%.
- `figures/brisc_quadrant_assignments.*`: BRISC quadrant assignments.
- `figures/brisc_rank_and_quadrant_agreement.*`: Kendall and label agreement.

The script also regenerates the per-explanation table. With `--figures` and the
optional plotting dependencies installed, it generates BMR/LE trajectories and
dataset-relative quadrant panels.

## Alignment analysis conditioned on fixed GT

`fixed_gt_conditional.py` fixes each available GT representation and compares XAI
methods within each metric. Scores are direction-aligned and standardized only
within dataset, formulation, GT representation, and metric. Consequently, the
analysis does not compare incompatible raw metric scales. The overall conditional
summary averages the standardized scores over the selected metrics.

For lower-is-better metrics, such as Hausdorff distance, the sign is first
reversed. The direction-aligned values are then z-standardized across XAI methods
within each fixed dataset--formulation--GT--metric slice. This is a positive
affine transformation after direction alignment and therefore preserves the XAI
method ordering for every metric. It prevents large-range distance metrics from
dominating bounded overlap or pointing metrics when their standardized scores
are summarized.

```bash
python additional_analysis/fixed_gt_conditional.py \
  --scores <long-form-alignment-scores.csv> \
  --output-dir <output-directory> \
  --metrics iou pg epg distance \
  --hard-threshold 0.10 \
  --figures
```

The input uses the long-form score schema produced by `scripts/compute_alignment.py`.
The default aggregation uses IoU, PG, EPG, and distance. BER can be selected through
`--metrics`, but it is omitted by default because it is the exact complement of EPG.

For Synthetic 2D under the soft formulation, the conditional overall leaders are
GGC for Bounding-box and Boundary GT, and IG for Point and Region GT. Under the hard
formulation, they are GC for Bounding-box, GGC for Boundary, and IG for Point and
Region. The metric-level results further show why the conditional view is useful:
for fixed Region GT under the soft formulation, IoU ranks GC first, while EPG and PG
rank IG first. Complete tables and heatmaps for all four datasets are under
`results/fixed_gt/`.

- `fixed_gt_metric_summary.csv`: raw means, standardized scores, and method ranks.
- `fixed_gt_overall_summary.csv`: conditional multi-metric method rankings.
- `fixed_gt_top_methods.csv`: leading method for each fixed-GT slice.
- `fixed_gt_<dataset>_<formulation>.png`: metric-specific conditional heatmaps.
