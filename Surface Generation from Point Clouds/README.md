# Surface Generation from Point Clouds

3D surface reconstruction from sparse point clouds using:
- a **naive geometric signed-distance approximation**
- a **neural implicit signed-distance model** inspired by **DeepSDF**

This project reconstructs surfaces from point clouds with normals and extracts meshes using **marching cubes**.

---

## Preview

### Naive reconstruction
- nearest-neighbor tangent-plane signed distance
- simple and fast baseline
- works directly on the input point cloud

### Neural reconstruction
- MLP-based signed distance function
- smoother surfaces
- better generalization from sparse samples

---

## Assignment Goal

Given a point cloud \(P = \{p_1, p_2, ..., p_n\}\) with normals, define an implicit signed distance function:

\[
f(x,y,z)
\]

and reconstruct the surface at:

\[
f(x,y,z)=0
\]

The two required methods are:

### 1) Naive geometric reconstruction
For each grid point \(p\), find the nearest point \(p_j\) in the point cloud and compute:

\[
f(p)=n_j \cdot (p-p_j)
\]

where:
- \(p_j\) is the nearest point
- \(n_j\) is the normal of that point

### 2) Neural implicit reconstruction
Train a multilayer perceptron to approximate the signed distance function from sampled 3D points and SDF values.

---

## Repository Structure

```text
.
├── naiveReconstruction.py        # geometric SDF reconstruction
├── model.py                      # neural decoder architecture
├── neuralNetReconstruction.py    # training / validation / evaluation
├── utils.py                      # helper functions, dataset, visualization
├── bunny-500.pts                 # point cloud with normals
├── bunny-1000.pts                # point cloud with normals
├── sphere.pts                    # point cloud with normals
└── README.md
```

---

## Input Data Format

Each `.pts` file contains one point per line:

```text
x y z nx ny nz
```

where:
- `x y z` are the 3D coordinates
- `nx ny nz` are the surface normal components

Included datasets:
- `bunny-500.pts`
- `bunny-1000.pts`
- `sphere.pts`

---

## Installation

Create a virtual environment if you want a clean setup.

### Windows
```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy scikit-image scikit-learn trimesh "pyglet<2" torch
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scikit-image scikit-learn trimesh "pyglet<2" torch
```

Quick install only:
```bash
pip install numpy scikit-image scikit-learn trimesh "pyglet<2" torch
```

---

## Method 1: Naive Reconstruction

Implemented in `naiveReconstruction.py`.

### Idea
A 3D grid is created around the point cloud. For each grid point:
1. find the closest point in the input cloud using `KDTree`
2. use that point's normal
3. compute signed distance to the tangent plane
4. store the result in the implicit field

Then **marching cubes** extracts the zero level set as a mesh.

### Run
```bash
python naiveReconstruction.py --file sphere.pts --method naive
python naiveReconstruction.py --file bunny-500.pts --method naive
python naiveReconstruction.py --file bunny-1000.pts --method naive
```

---

## Method 2: Neural Reconstruction

Implemented in:
- `model.py`
- `neuralNetReconstruction.py`

### Network architecture
The model is an 8-layer fully connected decoder:
- input: `3D point (x, y, z)`
- hidden width: `512`
- after the 4th FC layer, a `509-D` feature is concatenated with the original `3-D` input
- first 7 FC layers use:
  - weight normalization
  - `PReLU` activation with shared learnable slope
  - dropout `p=0.1`
- final layer maps to `1` SDF value
- output activation: `tanh`

### Training data
Training samples are created by perturbing each surface point along its normal:

\[
p_i' = p_i + \epsilon n_i
\]

with:

\[
\epsilon \sim \mathcal{N}(0, 0.05^2)
\]

The target SDF is the sampled offset \(\epsilon\).

### Loss
Training uses **clamped L1 loss**:

\[
L = | clamp(f_\theta(p_i'), \sigma) - clamp(s_i, \sigma) |
\]

with:

\[
\sigma = 0.1
\]

### Optimizer
- `AdamW`
- learning rate: `1e-4`
- weight decay: `1e-4`

### Train
```bash
python neuralNetReconstruction.py --input_pts sphere.pts --checkpoint_folder checkpoints_sphere
python neuralNetReconstruction.py --input_pts bunny-500.pts --checkpoint_folder checkpoints_b500
python neuralNetReconstruction.py --input_pts bunny-1000.pts --checkpoint_folder checkpoints_b1000
```

### Evaluate
```bash
python neuralNetReconstruction.py --input_pts sphere.pts --checkpoint_folder checkpoints_sphere -e
python neuralNetReconstruction.py --input_pts bunny-500.pts --checkpoint_folder checkpoints_b500 -e
python neuralNetReconstruction.py --input_pts bunny-1000.pts --checkpoint_folder checkpoints_b1000 -e
```

---

## Expected Output

Both scripts open a 3D mesh viewer using `trimesh`.

Use it to:
- rotate the object
- zoom in/out

## Implementation Notes

### `naiveReconstruction.py`
- creates a 3D grid around the input cloud
- flattens grid coordinates into query points
- uses `KDTree` for nearest-neighbor search
- computes:
  - nearest point
  - nearest normal
  - signed distance to tangent plane
- reshapes values back into a 3D scalar field

### `model.py`
- defines the decoder network
- includes the skip connection after layer 4
- applies weight normalization, activation, dropout, and final `tanh`

### `neuralNetReconstruction.py`
- handles:
  - training loop
  - validation loop
  - checkpoint saving
  - evaluation on a dense 3D grid

### `utils.py`
- point normalization
- normal normalization
- dataset sampling
- marching cubes mesh extraction
- mesh visualization

---

## Troubleshooting

### Viewer does not open
Install the viewer dependencies:
```bash
pip install trimesh "pyglet<2" scipy
```

### Torch installation issues
Check your Python version and install PyTorch from the official selector if needed.

### Slow naive reconstruction
Reduce the grid resolution in `createGrid(...)` for debugging, then increase it again for final screenshots.

### No CUDA
The code automatically falls back to CPU if GPU is not available.

---

## Results Summary

In general:
- the **naive method** produces rougher surfaces because it depends only on the tangent plane of the nearest point
- the **neural method** produces smoother and more coherent reconstructions by learning a continuous implicit field

For sparse inputs:
- `bunny-500` is more difficult
- `bunny-1000` usually gives better detail
- `sphere` is the easiest case and reconstructs well with both methods

---


## Course Info

Assignment based on **Surface Generation from Point Clouds** for **Generative AI (INF426)**.
