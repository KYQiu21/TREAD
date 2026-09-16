# TREAD

[![Python](https://img.shields.io/badge/Python-≥3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**TREAD** (**T**ransfer learning-based **RE**peat **A**nnotation using Protein Embe**D**dings) annotates tandem-repeat regions directly from protein sequences using residue-level ProtT5 embeddings and trained neural-network models.

> **Sequence in → repeat annotations out**  
> TREAD automatically generates ProtT5 embeddings, performs residue-level inference, and outputs repeat segments, residue-wise scores, and profile plots. Users do **not** need to pre-compute embeddings or load model checkpoints manually.

### Two core workflows

| Workflow | What it does | Command |
| --- | --- | --- |
| **[Predict](#prediction)** | Annotate repeat regions in one or more protein sequences using bundled or custom models | `tread predict` |
| **[Train](#training)** | Train a binary or multitask residue-level model from your own annotations | `tread train` |

**Jump to:** [Quick start](#quick-start) · [Prediction](#prediction) · [Training](#training) · [Reproducibility](#reproducibility) · [Citation](#citation)

### Try TREAD online

[Google Colab](https://colab.research.google.com/drive/1gbtb5BtevWE9vChJrYgiNW2mQSW_kN8j) · [Hugging Face Space](https://huggingface.co/spaces/kevinky/TREAD)

The online interfaces provide convenient interactive demos and quick analyses. For reproducible analysis of user-provided FASTA files and for custom training, the local command-line implementation is recommended.

---

<a id="quick-start"></a>

## 🚀 Quick start

TREAD requires **Python 3.10 or newer**. A CUDA-capable GPU is recommended for faster ProtT5 embedding, but CPU execution is fully supported.

### 1. Install

Clone the repository:

```bash
git clone https://github.com/KYQiu21/TREAD.git
cd TREAD
```

Create a clean conda environment:

```bash
conda create -p /path/to/new/conda_environment python=3.10
conda activate /path/to/new/conda_environment
```

Install TREAD and its dependencies:

```bash
python -m pip install --upgrade pip
pip install .
```

### 2. Run a prediction

Run the bundled general repeat model on the included example:

```bash
tread predict predict_example/Q60773.fasta
```

### 3. Inspect the results

Results are written to `tread_results/` by default:

```text
tread_results/
├── segments.tsv
├── residue_scores.tsv
└── plots/
    └── 0001_sp_Q60773_CDN2D_MOUSE_repeat_profile.png
```

On the first run, the ProtT5 encoder (`Rostlab/prot_t5_xl_uniref50`) is downloaded automatically by Hugging Face Transformers and then cached locally. The trained TREAD checkpoints are bundled with the package.

To inspect the available commands:

```bash
tread --help
tread predict --help
tread train --help
```

---

<a id="prediction"></a>

## 🧬 Prediction

TREAD accepts FASTA files containing one or more protein sequences.

For multi-sequence FASTA files, all sequences are processed in one run. Tabular outputs from all proteins are combined, while each sequence receives its own repeat-profile plot.

```bash
tread predict predict_example/multi.fasta -o multi_results
```

### Bundled models

TREAD currently ships with two ready-to-use models:

| Model | CLI name | Purpose | Default |
| --- | --- | --- | --- |
| General repeat model | `repeatsdb` | Predicts repeat regions and repeat-fold probabilities | Yes |
| Beta-propeller blade model | `propeller-blade` | Specialized beta-propeller blade annotation | No |

Examples:

```bash
# Six RepeatsDB folds annotation
tread predict predict_example/O95834.fasta --model repeatsdb

# Beta-propeller blade annotation
tread predict predict_example/O95834.fasta --model propeller-blade
```

A custom model produced by `tread train` can be used instead of a bundled model:

```bash
tread predict query.fasta --weights my_model/model.pt -o predictions
```

### Common prediction options

| Option | Default | Description |
| --- | ---: | --- |
| `-o`, `--outdir` | `tread_results` | Output directory |
| `--model` | `repeatsdb` | Bundled model to use when `--weights` is not supplied |
| `--weights` | — | Custom checkpoint produced by `tread train` |
| `--device` | `auto` | `auto`, `cpu`, or `cuda`; `auto` uses CUDA when available |
| `--threshold` | `0.8` | Residue probability threshold for segment extraction |
| `--min-length` | `15` | Minimum predicted repeat-segment length |
| `--embedding-chunk-length` | `1000` | Maximum ProtT5 chunk length for long proteins |
| `--embedding-overlap` | `100` | Overlap between ProtT5 chunks |
| `--no-residue-scores` | off | Skip `residue_scores.tsv` |
| `--no-plot` | off | Skip per-sequence PNG profile plots |

For the complete prediction interface:

```bash
tread predict --help
```

### Long proteins

Long proteins are embedded as overlapping ProtT5 chunks and reconstructed into a full-length residue-embedding matrix before TREAD inference.

For example, the included human ankyrin-2 sequence (`ANK2_HUMAN`; 3957 residues) can be analyzed directly:

```bash
tread predict examples/Q01484.fasta -o ank2_results
```

The default embedding settings use chunks of 1000 residues with an overlap of 100 residues. These values can be changed when needed:

```bash
tread predict protein.fasta \
    --embedding-chunk-length 800 \
    --embedding-overlap 100
```

### Output files

**`segments.tsv`**

One row is written for each predicted repeat segment. Coordinates are **1-based and inclusive**.

| Column | Description |
| --- | --- |
| `sequence_id` | FASTA identifier |
| `sequence_length` | Protein length |
| `repeat_index` | Repeat segment number within the sequence |
| `start`, `end` | 1-based inclusive segment coordinates |
| `length` | Segment length |
| `mean_repeat_score` | Mean repeat probability across the segment |
| `repeat_class` | Predicted repeat class/type when available |
| `class_score` | Corresponding class probability |

**`residue_scores.tsv`**

One row is written per residue. The table always contains `sequence_id`, `position`, and `repeat_probability`.

Multitask models additionally write one probability column for each repeat type.

**Profile plots**

TREAD writes one PNG repeat-probability profile per input sequence. Predicted repeat segments are shaded, and the decision threshold is shown as a dashed line.

Plots can be disabled with:

```bash
tread predict predict_example/Q60773.fasta --no-plot
```

---

<a id="training"></a>

## 🛠 Training

TREAD can train residue-level models directly from protein sequences and user-provided annotations.

Two training modes are supported:

- **Binary:** residue-level repeat detection.
- **Multitask:** joint repeat segmentation and user-defined repeat-type prediction.

ProtT5 embeddings are generated automatically. **Users do not need to prepare embedding matrices, HDF5 files, serialized datasets, or cross-validation folds.**

Because the framework operates on residue-level protein embeddings and user-provided annotations, it can in principle also be adapted to sequence or structural motifs beyond the repeat annotations studied in this work.

### Binary training

Prepare:

1. a FASTA file containing all positive and negative proteins; and
2. a tab-separated annotation file describing repeat intervals.

For example:

```text
sequence_id    start    end
protein_A      20       73
protein_A      80       122
protein_B      15       48
```

Coordinates are **1-based and inclusive**.

A sequence present in the FASTA file but absent from the annotation table is treated as a fully negative protein.

Train a model with:

```bash
tread train train_example/propeller/propeller.fasta \
            train_example/propeller/propeller_annotations.tsv \
            -o my_propeller_model
```

TREAD splits whole proteins into training and validation sets **before** window generation, preventing windows from the same protein from leaking across the split.

### Multitask training with repeat types

For joint repeat segmentation and repeat-type prediction, add a `repeat_type` column to the annotation table:

```text
sequence_id    start    end    repeat_type
protein_A      20       73     alpha_solenoid
protein_A      80       122    alpha_solenoid
protein_B      15       48     beta_propeller
```

Train with:

```bash
tread train train_example/repeatsdb/repeatsdb.fasta \
            train_example/repeatsdb/repeatsdb_annotations.tsv \
            --task multitask \
            -o my_repeatsdb_model
```

TREAD automatically discovers the unique `repeat_type` values and creates one type-prediction head per type. The type order is stored in `model.pt` and reused during prediction.

The same linear-edge targets are applied to both the segmentation target and the annotated repeat-type target.

Segmentation loss is evaluated for all non-padding residues, whereas repeat-type loss is evaluated only at residues covered by an annotation. FASTA sequences without annotations remain valid negative proteins.

The default multitask loss weights are **1** for segmentation and **10** for repeat type.

<details>
<summary><strong>Label schemes and linear-edge targets</strong></summary>

<br>

By default, TREAD reproduces the original **linear-edge** target scheme used for the bundled models. This scheme is designed to make peak behaviours explicit and easy to extract.

Inside each annotated repeat interval, the central region has target 1.0, while the first and last 10% of residues are linearly tapered from 0.5 toward 1.0. Residues outside annotated intervals have target 0.0.

Users can adjust `--edge-ratio` and `--edge-min` to improve segment extraction.

The annotation itself is important for this process. For example, when annotated repeat segments are directly adjacent without gaps, linear-edge labels assigned during training can be crucial for successful segment extraction during inference.

| Option | Default | Description |
| --- | ---: | --- |
| `--label-scheme` | `linear` | `linear` for soft repeat edges or `noedge` for hard 0/1 labels |
| `--edge-ratio` | `0.1` | Fraction of each annotated interval tapered at each end |
| `--edge-min` | `0.5` | Target value at the outermost annotated residues |

Examples:

```bash
# Default: original TREAD linear-edge labels
tread train proteins.fasta annotations.tsv \
    --label-scheme linear \
    --edge-ratio 0.1 \
    --edge-min 0.5

# Hard 0/1 labels
tread train proteins.fasta annotations.tsv --label-scheme noedge
```

</details>

<details>
<summary><strong>Training options</strong></summary>

<br>

| Option | Default | Description |
| --- | ---: | --- |
| `-o`, `--outdir` | `tread_model` | Training output directory |
| `--task` | `binary` | `binary` or `multitask` |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--validation-fraction` | `0.1` | Fraction of proteins reserved for validation |
| `--seed` | `42` | Random seed |
| `--epochs` | `30` | Maximum number of training epochs |
| `--patience` | `5` | Early-stopping patience based on validation loss |
| `--batch-size` | `32` | Training batch size |
| `--learning-rate` | `1e-5` | Adam learning rate |
| `--window-size` | `64` | Residue window length |
| `--train-overlap` | `32` | Overlap between training windows |
| `--pos-weight` | — | Optional positive-class weight for BCE loss |
| `--seg-loss-weight` | `1.0` | Multitask segmentation-loss weight |
| `--type-loss-weight` | `10.0` | Multitask repeat-type-loss weight |
| `--embedding-chunk-length` | `1000` | Maximum ProtT5 chunk length |
| `--embedding-overlap` | `100` | Overlap between ProtT5 chunks |

</details>

<details>
<summary><strong>Architecture options</strong></summary>

<br>

The default architecture can also be changed directly from the command line.

We recommend performing a grid search on the target dataset to determine appropriate hyperparameter settings.

| Option | Default | Description |
| --- | ---: | --- |
| `--out-channel` | `64` | Convolutional channel width |
| `--hidden-dim` | `64` | Hidden dimension before the prediction head |
| `--num-block` | `2` | Number of residual blocks |
| `--dropout` | `0.2` | Dropout probability |
| `--kernel-size-conv1` | `7` | Stem convolution kernel size |
| `--kernel-size-block` | `7` | Residual-block convolution kernel size |
| `--input-noise-std` | `0.0` | Gaussian noise SD applied to non-padding embeddings during training |

</details>

For the complete training interface:

```bash
tread train --help
```

Example input formats used in the manuscript are provided under:

```text
train_example/propeller/
train_example/repeatsdb/
```

---

<a id="reproducibility"></a>

## 📦 Training outputs and reproducibility

A training run produces:

```text
my_repeat_model/
├── model.pt
├── training_history.tsv
├── training_summary.json
├── train_ids.txt
├── validation_ids.txt
└── embedding_cache/
```

| File | Purpose |
| --- | --- |
| `model.pt` | Trained parameters plus model and training metadata required for inference |
| `training_history.tsv` | Per-epoch training and validation metrics |
| `training_summary.json` | Data counts, hyperparameters, architecture, label scheme, software versions, best epoch, validation metrics, and output paths |
| `train_ids.txt` | Protein IDs used for training |
| `validation_ids.txt` | Protein IDs reserved for validation |
| `embedding_cache/` | Cached ProtT5 embeddings used during training |

The embedding cache avoids recomputing ProtT5 representations during training and can be deleted after training is complete.

A trained checkpoint can be used directly for inference:

```bash
tread predict query.fasta --weights my_repeat_model/model.pt -o predictions
```

### CPU and GPU execution

TREAD automatically uses CUDA when available and otherwise runs on CPU:

```bash
tread predict proteins.fasta --device auto
```

A device can also be selected explicitly:

```bash
tread predict proteins.fasta --device cuda
tread predict proteins.fasta --device cpu
```

ProtT5 embedding is the computationally expensive step, so GPU execution is recommended for large datasets and long proteins.

---

## 💻 Development

Install TREAD in editable mode with the optional test dependency:

```bash
pip install -e ".[dev]"
pytest
```

---

<a id="citation"></a>

## 📖 Citation

If you use TREAD in your research, please kindly cite:

```text
Qiu, K., Ludwiczak, J., Lupas, A. N., & Dunin-Horkawicz, S. (2026).
Beyond profiles: supervised repeat annotation using protein embeddings.
bioRxiv, 2026-05.
https://www.biorxiv.org/content/10.64898/2026.05.19.725729v1.full
```

---

## License

TREAD is distributed under the MIT License. See `LICENSE` for details.
