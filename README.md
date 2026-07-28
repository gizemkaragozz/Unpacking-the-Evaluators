# Unpacking the Evaluators

This repository contains the reproducibility code accompanying the paper
*Unpacking the Evaluators*. It provides the implementation used to generate the
synthetic data, prepare the four predictive models, construct ground-truth (GT)
representations, characterize attribution maps, compute XAI–GT spatial alignment
scores, and reproduce the ranking analyses.


## Structure

```text
configs/                         Study parameters for 2D and 3D experiments
scripts/compute_alignment.py     Descriptor and alignment computation
scripts/analyze_alignment.py     Ranking and sensitivity analysis
tests/                           Unit tests for the main computations
xai_alignment/                   Reusable implementation
xai_alignment/init_*.py          Data, training, and checkpoint interfaces
2d_synthetic_data_generator.py   Synthetic 2D data generation
3d_synthetic_data_generator.py   Synthetic 3D data generation
```

## Installation

Python 3.10 or later is recommended.

```bash
pip install -e .
```

To generate explanations with the provided Captum helper:

```bash
pip install -e ".[xai]"
```

To download the medical datasets and prepare or train the predictive models:

```bash
pip install -e ".[study,xai]"
```

To generate figures or run the test suite:

```bash
pip install -e ".[plots,dev]"
```

## Synthetic data

```bash
python 2d_synthetic_data_generator.py --out_dir data/synthetic_2d
python 3d_synthetic_data_generator.py --out_dir data/synthetic_3d
```

The number of samples, spatial size, random seed and overlap policy are exposed as command-line arguments. Run either script with
`--help` for the complete interface.

## Data and predictive models

The four initialization modules contain the data readers, preprocessing,
predictive architecture, training procedure, and checkpoint loader associated
with each experiment:

| experiment | initialization module | saved checkpoint |
|---|---|---|
| Synthetic 2D | `xai_alignment.init_synthetic_2d` | dictionary containing `model`, `shapes`, and validation-tuned thresholds |
| BRISC 2D | `xai_alignment.init_brisc_2d` | DenseNet-121 state dictionary |
| Synthetic 3D | `xai_alignment.init_synthetic_3d` | dictionary containing `model_state_dict` and training metadata |
| MosMed 3D | `xai_alignment.init_mosmed_3d` | MONAI DenseNet state dictionary |

For either synthetic dataset, first run the corresponding generator above and
pass its output directory to `create_dataloaders` or `train_model`.

The medical dataset readers use the public Kaggle identifiers
`briscdataset/brisc2025` and
`mathurinache/mosmeddata-chest-ct-scans-with-covid19`. The standard Kaggle
client reads authentication from its normal configuration or environment;
credential values and usernames are not accepted, printed, or stored by this
repository.

```python
from xai_alignment.init_brisc_2d import download_dataset, train_model

data_root = download_dataset("data/brisc2025")
model = train_model(
    data_root,
    "checkpoints/brisc_densenet121.pt",
)
```

Every module also exposes `load_model(checkpoint_path, ...)`, allowing
explanation generation to use an existing saved predictive model without
retraining. The executed MosMed study path loads the reported MONAI architecture
from a raw state-dictionary checkpoint. Its initialization module also supplies a
training route using the same preprocessing, annotation-aware split, architecture,
and checkpoint format.

## Explanation generation

`xai_alignment.explainers.explain` supports Grad-CAM (`GC`), Guided Grad-CAM
(`GGC`), Integrated Gradients (`IG`), and Saliency (`SL`) for a supplied trained
model and target class. The final convolutional layer must be supplied for GC and
GGC. Integrated Gradients uses a zero baseline with 64 steps in 2D and 32 steps
in 3D, as recorded in the configuration files.

`xai_alignment.taxonomy_explainers.explain_taxonomy` exposes the attribution
method implementations used across the output-behavior experiments. Its
dimension-dependent defaults reproduce the recorded 2D and 3D settings while
allowing each parameter to be supplied explicitly.

| parameter | 2D | 3D |
|---|---:|---:|
| Integrated Gradients steps | 64 | 32 |
| SmoothGrad samples | 25 | 16 |
| SmoothGrad noise scale | 0.10 | 0.10 |
| Feature Ablation patch | 16 × 16 | 16 × 16 × 8 |
| Occlusion window | 1 × 16 × 16 | 1 × 16 × 16 × 16 |
| Occlusion stride | 1 × 8 × 8 | 1 × 8 × 8 × 8 |
| Score-CAM channels / batch size | 32 / 16 | 16 / 8 |

## Alignment computation

Prepare a CSV manifest with one row per explanation:

| column | description |
|---|---|
| `dataset` | dataset identifier |
| `sample_id` | sample identifier |
| `method` | XAI method identifier |
| `attribution_path` | path to a NumPy attribution array |
| `region_mask_path` | path to the corresponding binary region mask |

For volumetric experiments, manifest arrays use the study's evaluation order
`(H, W, D)`, with depth on the last axis. Generated Synthetic 3D arrays are
stored as `(D, H, W)` and are transposed with
`np.transpose(array, (1, 2, 0))` before being written to the evaluation
manifest, matching the submitted notebook.

Run:

```bash
python scripts/compute_alignment.py \
  --manifest manifest.csv \
  --config configs/2d.yaml \
  --scores-output outputs/alignment_scores.csv \
  --descriptors-output outputs/descriptors.csv

python scripts/analyze_alignment.py \
  --scores outputs/alignment_scores.csv \
  --config configs/2d.yaml \
  --output-dir outputs/analysis \
  --descriptors outputs/descriptors.csv \
  --figures
```

Use `configs/3d.yaml` for volumetric data.

The metric names are read from the `alignment.metrics` and `analysis.metrics`
lists. IoU, Pointing Game (PG), Energy-based Pointing Game (EPG), Background
Energy Ratio (BER), and distance are implemented; any subset can be selected
without changing the computation code. The supplied configuration includes BER
to reproduce the submitted aggregation. Removing `ber` from
`analysis.metrics` re-runs the aggregation with the remaining metrics while BER
remains available in the descriptive score table.

The distance family follows the study specification. Its hard formulation is
symmetric boundary HD95: the 95th percentile of each directed boundary distance
is computed before taking their maximum. Its soft formulation is the directed,
attribution-weighted mean distance from the GT boundary to the explanation.
For 3D evaluation, all metrics are computed on the slices containing the
corresponding GT, matching the volumetric evaluation protocol.

## Reproducibility details

The region mask is converted into region, morphological-boundary, bounding-box, and point
GT representations. Each connected 2D component is represented by a
distance-transform maximum rasterized as a filled radius-2 disk. Each connected
3D component is represented by its rounded centroid and the six directly adjacent
axial voxels, producing a seven-voxel marker.

Boundary Mass Ratio (BMR) is the fraction of retained attribution mass located on
the one-step inner morphological boundary of the retained support, using a
connectivity-one structuring element. Local Entropy (LE) divides
the retained attribution map into non-overlapping, equal-sized spatial patches,
computes normalized Shannon entropy within each patch, assigns zero to empty or
single-active-element patches, and averages over all patches. Patch dimensions,
support proportion, point-marker construction, hard thresholds, selected metrics,
and analysis parameters are specified in `configs/`.

Hard formulations are evaluated at the explicit retained-support levels
`{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 0.90, 1.00}` listed in
the configuration files. Soft Pointing Game uses `alpha = 4`.

## Analysis outputs

The analysis script writes XAI–GT pair rankings, selected XAI–GT–metric triples,
metric-support tables, bootstrap summaries, soft–hard paired-rank agreement,
threshold-sensitivity summaries, and the submitted ordinal flow paths. The flow
paths use weights 4, 3, 2, and 1 for the two GT ranks and two supporting metric
ranks. With `--figures`, these tables also generate alignment heatmaps,
formulation rank-shift plots, BMR–LE plots, and soft/hard Sankey diagrams.
The heatmaps display raw mean alignment scores; metric direction is represented
by color-map orientation rather than by transforming the displayed values.
