# TREAD

**TREAD** (**T**ransfer learning-based **RE**peat **A**nnotation using Protein Embe**D**dings) annotates tandem-repeat regions directly from protein sequences using residue-level ProtT5 embeddings and trained neural-network models.

The command-line interface accepts single- or multi-sequence FASTA files, generates ProtT5 embeddings automatically, runs TREAD inference, and writes repeat segments, residue-wise scores, and profile plots. Users do **not** need to pre-compute embeddings or load model checkpoints manually.

## Online demos

The following Google Colab notebook and Hugging Face Space serve as convenient interactive demos or quick analysis. The local command-line implementation is the recommended reproducible interface for analyzing user-provided FASTA files and custom training.

- Google Colab: https://colab.research.google.com/drive/1gbtb5BtevWE9vChJrYgiNW2mQSW_kN8j
- Hugging Face Space: https://huggingface.co/spaces/kevinky/TREAD

## Quick start

TREAD requires **Python 3.10 or newer**. A CUDA-capable GPU is recommended for faster ProtT5 embedding, but CPU execution is supported.

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

Install all dependencies:

```bash
python -m pip install --upgrade pip
pip install .
```

Check the installation:

```bash
tread --help
tread predict --help
tread train --help
```

Run the bundled general repeat model on the included example:

```bash
tread predict predict_example/Q60773.fasta
```

Results are written to `tread_results/` by default:

```text
tread_results/
├── segments.tsv
├── residue_scores.tsv
└── plots/
    └── 0001_sp_Q60773_CDN2D_MOUSE_repeat_profile.png
```

On the first run, the ProtT5 encoder (`Rostlab/prot_t5_xl_uniref50`) is downloaded automatically by Hugging Face Transformers and then cached locally. The trained TREAD checkpoints are bundled with the package.

## Bundled models

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

## Prediction

TREAD accepts `fasta` files containing one or more sequences. 

If multiple sequences are analyzed, all sequences are processed in one run. All tabular outputs reporting detected segments in all proteins are combined, while each sequence receives its own profile plot.

```bash
tread predict predict_example/multi.fasta -o multi_results
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

For all prediction options:

```bash
tread predict --help
```

### Long proteins

Long proteins are embedded as overlapping ProtT5 chunks and reconstructed into a full-length residue-embedding matrix before TREAD inference.

For example, the included example, human ankyrin-2 sequence (ANK2_HUMAN; 3957 residues), can be analyzed directly:

```bash
tread predict examples/Q01484.fasta -o ank2_results
```

The default embedding settings are 1000 residues per chunk with 100-residue overlap. They can be changed when needed:

```bash
tread predict protein.fasta \
    --embedding-chunk-length 800 \
    --embedding-overlap 100
```

### Output files

### 1. `segments.tsv`

One row per predicted repeat segment. Coordinates are **1-based and inclusive**.

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

### 2. `residue_scores.tsv`

One row per residue. The table always contains `sequence_id`, `position`, and `repeat_probability`. Multitask models additionally write one probability column per repeat type.

### 3. Profile plots

TREAD writes one PNG repeat-probability profile per input sequence. Predicted repeat segments are shaded and the decision threshold is shown as a dashed line.

Disable plots with:

```bash
tread predict predict_example/Q60773.fasta --no-plot
```

## Train a model on your own annotations

TREAD can train either:

- a **binary** residue-level repeat detector; or
- a **multitask** model that jointly predicts repeat regions and user-defined repeat types.
- in principle, a model designed to predict different sequence/structural motifs apart from sequence repeats studied in this work.

ProtT5 embeddings are generated automatically. **Users do not need to prepare embedding matrices, HDF5 files, serialized datasets, or cross-validation folds.**

### Binary training

Prepare **a FASTA file** containing all positive and negative proteins, plus **a tab-separated annotation file**:

```text
sequence_id    start    end
protein_A      20       73
protein_A      80       122
protein_B      15       48
```

Coordinates are **1-based and inclusive**. A sequence present in the FASTA file but absent from the annotation table is treated as a fully negative protein.

Train a model with:

```bash
tread train train_example/propeller/propeller.fasta \
            train_example/propeller/propeller_annotations.tsv \
            -o my_propeller_model
```

TREAD splits whole proteins into training and validation sets **before** window generation, preventing windows from the same protein from leaking across the split.

### Linear-edge labels

By default, TREAD reproduces the original **linear-edge** target scheme used for the bundled models. This scheme is designed to make the peak behaviours explicit and easy to extract.

Inside each annotated repeat interval, the central region has target 1.0, while the first and last 10% of residues are linearly tapered from 0.5 toward 1.0. Residues outside annotated intervals are 0.0.

User can adjust `--edge-ratio` and `--edge-min` to improve the segment extraction process. The annotation itself is the key to this process. For example, for a repeat protein where the annotated repeat segments are exactly adjacent to each other without any gaps in between, the linear-edge labels assigned during training is crucial for successful segment extraction during inference.

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

### Multitask training with repeat types

For joint repeat segmentation and repeat-type prediction, add a `repeat_type` column:

```text
sequence_id    start    end    repeat_type
protein_A      20       73     alpha_solenoid
protein_A      80       122    alpha_solenoid
protein_B      15       48     beta_propeller
```

Train with:

```bash
tread train train_example/repeatsdb/repeatsdb.fasta \
      train_example/repeatsdb/repeatsdb_annotations.tsv  \
      --task multitask \
      -o my_repeatsdb_model
```

TREAD discovers the unique `repeat_type` values automatically and creates one type-prediction head per type. The type order is stored in `model.pt` and reused during prediction.

The same linear-edge targets are applied to the segmentation target and the annotated repeat-type target. Segmentation loss is evaluated for all non-padding residues; repeat-type loss is evaluated only at residues covered by an annotation. FASTA sequences without annotations remain valid negative proteins.

The default multitask loss weights are 1 for segmentation and 10 for repeat type.

### Common training options

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

### Architecture options

The default architecture can also be changed from the command line.

We recommend performing a grid search on your specific dataset to determine the optimal hyperparameter settings.

| Option | Default | Description |
| --- | ---: | --- |
| `--out-channel` | `64` | Convolutional channel width |
| `--hidden-dim` | `64` | Hidden dimension before the prediction head |
| `--num-block` | `2` | Number of residual blocks |
| `--dropout` | `0.2` | Dropout probability |
| `--kernel-size-conv1` | `7` | Stem convolution kernel size |
| `--kernel-size-block` | `7` | Residual-block convolution kernel size |
| `--input-noise-std` | `0.0` | Gaussian noise SD applied to non-padding embeddings during training |

For the complete training interface:

```bash
tread train --help
```

Format examples used in the manuscript are included under `train_example/propeller/` and `train_example/repeatsdb/`.

## Training outputs and reproducibility

A training run writes:

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

Use a trained checkpoint directly for inference:

```bash
tread predict query.fasta --weights my_repeat_model/model.pt -o predictions
```

## CPU and GPU execution

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

## Development installation

Install the package in editable mode with the optional test dependency:

```bash
pip install -e ".[dev]"
pytest
```

## Citation

If you use TREAD in your research, please kindly cite this manuscript:
```
Qiu, K., Ludwiczak, J., Lupas, A. N., & Dunin-Horkawicz, S. (2026). Beyond profiles: supervised repeat annotation using protein embeddings. bioRxiv, 2026-05.
```
## License

TREAD is distributed under the MIT License. See `LICENSE` for details.
