# Unpacking-the-Evaluators
Unpacking the Evaluators: How Configuration Shapes the Evaluation of Alignment in Explainable AI

This supplementary code provides synthetic dataset generation scripts for the 2D and 3D synthetic datasets used in the paper.

To generate the 2D dataset:
python generate_2d_synthetic_shapes.py --out_dir ./synthetic_2d_shapes --n 1400 --image_size 224 --seed 42

To generate the 3D dataset:
python generate_3d_synthetic_shapes.py --out_dir ./synthetic_3d_shapes --n 1400 --volume_size 64 --seed 42
