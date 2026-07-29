# TFMe: Time–Frequency–Memory-Enhanced Transformer for Anomaly Detection

TFMe is a reconstruction-based anomaly detector for LiDAR time series collected in car-parking environments. It couples a dual-branch Transformer autoencoder — one branch operating in the time domain, one in the frequency domain — with a multi-head memory module that constrains reconstructions to patterns seen during normal-only training.

Because the model is trained exclusively on normal data and never learns to reconstruct anomalies, a sample is flagged when either branch fails to reconstruct it within a fixed error threshold.

## Architecture

| Component | Role |
| --- | --- |
| `PositionalEncoding` | Standard sinusoidal encoding applied to both branches |
| `MultiHeadMemory` | Learned memory bank queried by cosine similarity, with hard shrinkage to sparsify addressing |
| `TimeBranch` | Transformer encoder over the raw temporal sequence, followed by memory read-out and reconstruction |
| `SpectralBranch` | Real FFT of the input, Transformer encoder over frequency bins, memory read-out, inverse FFT back to the time domain |
| `DualDomainTransformer` | Wraps both branches; returns both reconstructions and both attention maps |

### Memory module

Each input embedding is split across `n_heads` and matched against a bank of `MEMORY_SLOTS` learned prototypes via cosine similarity. The resulting attention is softmaxed, then passed through a hard-shrinkage operator with threshold `λ = 1 / MEMORY_SLOTS`, which zeroes out weakly matched slots and re-normalizes the remainder. The reconstruction is a sparse convex combination of memory items, so inputs unlike anything in the training distribution cannot be reconstructed accurately.

### Detection rule

Per-sample mean squared reconstruction error is computed independently for each branch and compared against a 3-sigma threshold calibrated on the training set:

| Branch | Threshold |
| --- | --- |
| Time | `0.007686922559514642` |
| Frequency | `0.005322358105331659` |

A sample is labelled anomalous if **either** branch exceeds its threshold.

## Data

`dataset.zip` contains `dataset.npy`, a `float32` array of shape `(71045, 340)`. The first `305` columns hold the LiDAR features used by the model; the remaining columns are ignored by the notebook.

Each row is reshaped to `(5, 61)` — 5 consecutive timesteps × 61 azimuth bins — giving one sample per row. Features are scaled with a `MinMaxScaler` fitted on the training split only.

| Split | Row range | Samples |
| --- | --- | --- |
| Train | `0 : 50335` | 50,335 |
| Validation | `50335 : 60341` | 10,006 |
| Test | `60341 : 71045` | 10,704 |

The test split contains normal samples only, so the evaluation in the notebook reports false positives and true negatives.

## Configuration

| Parameter | Value |
| --- | --- |
| `N_LAYERS` | 4 |
| `D_MODEL` | 128 |
| `N_HEAD` | 8 |
| `FFN` | 512 |
| `MEMORY_SLOTS` | 250 |
| `TIMESTEPS` | 5 |
| `FEATURES` | 61 |
| `BATCH_SIZE` | 256 |
| `EPOCHS` | 500 |
| `LEARNING_RATE` | 1e-4 |

Training uses MSE reconstruction loss with early stopping (patience-based, restoring best weights).

## Prerequisites

Conda is recommended for managing the environment.

```
python           3.8.20
pytorch          2.4.1+cu118
cudatoolkit      11.8.0
cudnn            9.10.1.4
numpy            1.24.1
scikit-learn     1.3.2
```

A CUDA device is used automatically when available; the notebook falls back to CPU otherwise.

## Usage

1. Extract the dataset into the repository root:

   ```bash
   unzip dataset.zip
   ```

2. Open `TFMe_Model_Architecture.ipynb` and run the cells in order. The notebook loads and splits the data, defines the full architecture, restores the pretrained weights from `best.pth`, and evaluates the detector on the test split.

To use the model directly:

```python
import torch

model = DualDomainTransformer()
model.load_state_dict(torch.load("best.pth", map_location=DEVICE))
model.to(DEVICE).eval()

with torch.no_grad():
    time_recon, freq_recon, time_att, freq_att = model(inputs)   # inputs: (B, 5, 61)
```

## Repository Contents

| File | Description |
| --- | --- |
| `TFMe_Model_Architecture.ipynb` | Full model definition, pretrained-weight loading, and evaluation walkthrough |
| `best.pth` | Pretrained `DualDomainTransformer` checkpoint |
| `dataset.zip` | Compressed `dataset.npy` feature array |
