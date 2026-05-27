# Classification of Motor Imagery EEG Signals Using a Lightweight CNN-Transformer Hybrid Architecture

---

## Abstract

Brain-computer interfaces (BCIs) based on motor imagery (MI) remain limited by the difficulty of decoding low-amplitude, non-stationary electroencephalographic (EEG) signals recorded from a reduced channel set. We propose **EEGCNNTransformer**, a lightweight hybrid architecture that couples a depthwise-separable convolutional backbone with a single-layer multi-head self-attention encoder. The convolutional stage extracts local spatiotemporal features from eight motor-cortex channels, while the Transformer encoder captures long-range temporal dependencies through a learnable CLS token and sinusoidal positional encoding. The model is evaluated on the PhysioNet EEG Motor Movement/Imagery Dataset (109 subjects) under a strict 5-fold subject-wise cross-validation protocol. Training incorporates Focal Loss, Exponential Moving Average (EMA) weight averaging, Test-Time Augmentation (TTA), and a two-stage subject-specific fine-tuning procedure. An architecture sweep and ablation study quantify the contribution of each design choice. All experiments are fully reproducible with a fixed random seed.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Architecture](#architecture)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Training Protocol](#training-protocol)
- [Evaluation Methodology](#evaluation-methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Repository Structure

```
EEG_Clasificador/
├── data/
│   └── raw/                          # PhysioNet raw EEG recordings (downloaded at runtime)
├── models/
│   ├── 00_folds/
│   │   ├── Kfold5.json               # Subject-wise 5-fold partition definitions
│   │   ├── fold_distribution.pdf     # Fold balance visualization
│   │   └── plot_fold_distribution.py # Script to regenerate fold plots
│   └── 04_hybrid/
│       ├── cnntransformer2c.ipynb    # Main notebook: binary MI classification (L/R)
│       └── ModeloW/
│           └── nb2_h6/              # Selected architecture (2 blocks, 6 heads)
│               ├── *_GLOBAL_*.{png,pdf}  # Aggregate visualizations across folds
│               ├── *.csv                 # Computational and summary metrics
│               └── nb2_h6/fold{1..5}/
│                   └── consumption_fold*_nb2_h6.json  # Per-fold resource reports
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Architecture

**EEGCNNTransformer** consists of three sequential stages:

### 1. Convolutional Backbone

| Component          | Details                                                                    |
| ------------------ | -------------------------------------------------------------------------- |
| Stem convolution   | `Conv1d` with 129 kernels spanning the full temporal receptive field       |
| Feature extraction | Depthwise-separable convolutions for parameter-efficient spatial filtering |
| Normalization      | `GroupNorm` after each convolutional block                                 |
| Activation         | ELU (Exponential Linear Unit)                                              |

### 2. Transformer Encoder

| Component               | Details                                               |
| ----------------------- | ----------------------------------------------------- |
| Self-attention          | Multi-head attention (`n_heads = 6`, `d_model = 144`) |
| Positional encoding     | Sinusoidal (fixed, not learned)                       |
| Classification token    | Learnable CLS token prepended to the sequence         |
| Feed-forward activation | GELU                                                  |
| Depth                   | Single encoder layer (`n_layers = 1`)                 |

### 3. Classification Head

| Component         | Details                                  |
| ----------------- | ---------------------------------------- |
| Pre-normalization | `LayerNorm` on CLS token output          |
| Projection        | Linear layer mapping to 2 output classes |

**Total trainable parameters:** ~90 K (lightweight by design for BCI deployment feasibility).

---

## Dataset and Preprocessing

### Source

**PhysioNet EEG Motor Movement/Imagery Dataset** [[1]](#references)

- 109 subjects, 64-channel EEG at 160 Hz
- Motor imagery runs 4, 8, 12 (left hand vs. right hand imagination)

### Channel Selection

Eight electrodes over the primary motor and supplementary motor areas:

`C3  C4  Cz  CP3  CP4  FC3  FC4  FCz`

### Preprocessing Pipeline

| Step              | Details                                                                      |
| ----------------- | ---------------------------------------------------------------------------- |
| Notch filter      | 60 Hz line-noise removal                                                     |
| Epoching          | -1.0 s to +5.0 s relative to event onset (6 s window, 960 samples)           |
| Normalization     | Channel-wise z-score using training-set statistics only                      |
| Data augmentation | Temporal jitter (35%), Gaussian noise injection (35%), channel dropout (15%) |

---

## Training Protocol

### Optimization

| Parameter          | Value                                          |
| ------------------ | ---------------------------------------------- |
| Optimizer          | AdamW (weight decay = 1e-2)                    |
| Base learning rate | 5e-4                                           |
| LR schedule        | Cosine annealing with linear warmup (4 epochs) |
| Loss function      | Focal Loss (gamma = 1.5)                       |
| Batch size         | 64                                             |
| Max epochs         | 60                                             |
| Early stopping     | Patience = 8 epochs on validation loss         |
| Class balancing    | `WeightedRandomSampler` per training fold      |

### Regularization and Ensembling

| Technique | Details                                                 |
| --------- | ------------------------------------------------------- |
| Dropout   | 0.2 (applied to attention and feed-forward layers)      |
| EMA       | Exponential Moving Average of weights (decay = 0.9995)  |
| TTA       | Test-Time Augmentation via temporal shifts at inference |

### Subject-Specific Fine-Tuning (Two-Stage)

1. **Stage 1 (frozen backbone):** Only the classification head is updated using subject-specific data.
2. **Stage 2 (full model):** All parameters are unfrozen and fine-tuned end-to-end with a reduced learning rate.

---

## Evaluation Methodology

### Cross-Validation

A strict **5-fold subject-wise split** ensures that no subject appears in both training and test partitions simultaneously. Fold assignments are stored in `models/00_folds/Kfold5.json` for full reproducibility.

### Metrics

| Metric             | Scope                                             |
| ------------------ | ------------------------------------------------- |
| Accuracy           | Per-fold and mean across folds                    |
| F1-score           | Macro and weighted averages                       |
| Precision / Recall | Per-class and macro                               |
| Confusion matrix   | Per-fold and global (aggregated across all folds) |

### Interpretability Analyses

- **Attention maps:** Visualization of self-attention weights across temporal positions
- **Saliency topomaps:** Channel-level importance via gradient-based saliency
- **Statistical topomaps:** Log10 p-value maps from permutation tests across channels
- **t-SNE embeddings:** 2D projection of CLS token representations

### Architecture Search and Ablation

- **Architecture sweep:** Grid over `n_blocks` {0, 2, 4} x `n_heads` {2, 4, 6} evaluated under 5-fold CV
- **Ablation study:** Systematic removal of individual components (depthwise convolutions, positional encoding, CLS token, GroupNorm/BatchNorm substitution, ELU/ReLU substitution)

---

## Installation

### Requirements

- Python >= 3.8
- CUDA-compatible GPU (recommended)

### Setup

```bash
git clone https://github.com/ASP-ML/MI_cnn_transformer_hybrid/tree/main.git
cd EEG_Clasificador
pip install -r requirements.txt
```

### Dependencies

| Package      | Version | Purpose                                         |
| ------------ | ------- | ----------------------------------------------- |
| PyTorch      | >= 2.0  | Deep learning framework                         |
| MNE          | >= 1.5  | EEG signal processing and PhysioNet data access |
| scikit-learn | >= 1.3  | Metrics, cross-validation utilities             |
| NumPy        | >= 1.24 | Numerical computation                           |
| SciPy        | >= 1.10 | Statistical analysis                            |
| matplotlib   | >= 3.7  | Visualization                                   |
| pandas       | >= 2.0  | Tabular data handling                           |
| tqdm         | >= 4.66 | Progress bars                                   |

---

## Usage

### Running the Full Pipeline

Open and execute the main notebook:

```
models/04_hybrid/cnntransformer2c.ipynb
```

The notebook implements the end-to-end pipeline:

1. **Data loading** - Automatic download from PhysioNet via MNE
2. **Preprocessing** - Channel selection, filtering, epoching, normalization
3. **5-fold training** - Global model training with EMA and early stopping
4. **Subject fine-tuning** - Two-stage adaptation per subject
5. **Evaluation** - Metrics computation, confusion matrices, interpretability plots
6. **Resource logging** - Per-fold computational consumption reports (GPU time, memory)

### Fold Definitions

Subject-to-fold assignments are pre-computed and stored in:

```
models/00_folds/Kfold5.json
```

To visualize the fold distribution:

```bash
python models/00_folds/plot_fold_distribution.py
```

---

## Reproducibility

All experiments are deterministically reproducible:

```python
RANDOM_STATE = 42
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Fold partitions, hyperparameters, and augmentation seeds are fixed. Raw data is downloaded from the canonical PhysioNet source to avoid dataset versioning issues.

---

## Citation

If you use this code, models, or methodology in your research, please cite:

```bibtex

```

**APA format:**



---

## License

This project is released under the [MIT License](LICENSE).

Copyright (c) 2025 Joel A. Cuascota
