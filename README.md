# Unpacking-the-Evaluators
Unpacking the Evaluators: How Configuration Shapes the Evaluation of Alignment in Explainable AI

## Synthetic Dataset Generation Code

This repository provides supplementary synthetic dataset generation code for the 2D and 3D datasets used in the paper.

### Requirements

#### 2D dataset
- numpy
- opencv-python

#### 3D dataset
- numpy
- scipy

---

### Generate 2D Synthetic Dataset

```bash
python generate_2d_synthetic_shapes.py \
    --out_dir ./synthetic_2d_shapes \
    --n 1400 \
    --image_size 224 \
    --seed 42
```

---

### Generate 3D Synthetic Dataset

```bash
python generate_3d_synthetic_shapes.py \
    --out_dir ./synthetic_3d_shapes \
    --n 1400 \
    --volume_size 64 \
    --seed 42
```

---

### Notes

- The datasets are generated with balanced multi-label combinations.
- Additional evaluation and experimental code will be released upon acceptance.
- The repository is anonymized for double-blind review.
