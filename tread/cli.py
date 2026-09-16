import argparse
import csv
import math
import re
import shlex
import sys
from pathlib import Path

from . import __version__
from .embedding import ProtT5Embedder, resolve_device
from .inference import (
    MODEL_SPECS,
    load_custom_tread_model,
    load_tread_model,
    predict_embedding,
    segment_prediction,
)
from .io import read_fasta
from .training import train_binary_model, train_multitask_model
from .utils import get_ranges, plot_profile


def _probability(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def _fraction(value):
    value = float(value)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and smaller than 1")
    return value


def _edge_ratio(value):
    value = float(value)
    if not 0.0 <= value <= 0.5:
        raise argparse.ArgumentTypeError("must be between 0 and 0.5")
    return value


def _positive_probability(value):
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return value


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def _nonnegative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _safe_plot_stem(identifier):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", identifier).strip("._")
    return stem[:160] or "sequence"


def _write_segments(path, rows):
    fieldnames = [
        "sequence_id", "sequence_length", "repeat_index", "start", "end",
        "length", "mean_repeat_score", "repeat_class", "class_score",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_residue_scores(path, record_id, prediction, type_names, mode="a"):
    path = Path(path)
    exists = path.exists() and mode == "a"
    fieldnames = ["sequence_id", "position", "repeat_probability"] + [
        name.replace(" ", "_").replace("-", "_") + "_probability"
        for name in type_names
    ]
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if not exists or mode == "w":
            writer.writeheader()
        for i, repeat_prob in enumerate(prediction.repeat_probability, start=1):
            row = {
                "sequence_id": record_id,
                "position": i,
                "repeat_probability": float(repeat_prob),
            }
            if prediction.type_probabilities is not None:
                for name, probs in zip(type_names, prediction.type_probabilities):
                    key = name.replace(" ", "_").replace("-", "_") + "_probability"
                    row[key] = float(probs[i - 1])
            writer.writerow(row)


def run_predict(args):
    records = read_fasta(args.fasta)
    device = resolve_device(args.device)
    if args.embedding_overlap >= args.embedding_chunk_length:
        raise ValueError("--embedding-overlap must be smaller than --embedding-chunk-length")

    print(f"TREAD: {len(records)} sequence(s) loaded from {args.fasta}")
    print(f"Compute device: {device}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    segments_path = outdir / "segments.tsv"
    residue_path = outdir / "residue_scores.tsv"
    if residue_path.exists():
        residue_path.unlink()
    plot_dir = None
    if not args.no_plot:
        plot_dir = outdir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ProtT5 (first run may download model files)...")
    embedder = ProtT5Embedder(device=device)
    if args.weights:
        print(f"Loading custom TREAD checkpoint: {args.weights}")
        loaded = load_custom_tread_model(args.weights, device)
    else:
        print(f"Loading bundled TREAD model: {args.model}")
        loaded = load_tread_model(args.model, device)

    all_segments = []
    for n, record in enumerate(records, start=1):
        print(f"[{n}/{len(records)}] {record.identifier} ({len(record.sequence)} aa)")
        if len(record.sequence) > args.embedding_chunk_length:
            step = args.embedding_chunk_length - args.embedding_overlap
            n_chunks = 1 + math.ceil((len(record.sequence) - args.embedding_chunk_length) / step)
            print(
                f"  long sequence detected: ProtT5 embedding will use {n_chunks} "
                f"overlapping chunks ({args.embedding_chunk_length} aa, "
                f"{args.embedding_overlap} aa overlap)"
            )

        embedding = embedder.embed(
            record.sequence,
            chunk_length=args.embedding_chunk_length,
            overlap=args.embedding_overlap,
        )
        prediction = predict_embedding(loaded.model, embedding)
        ranges = get_ranges(
            prediction.repeat_probability,
            cutoff1=args.threshold,
            min_len=args.min_length,
            cutoff2=args.threshold,
            frac2=0.5,
        )
        segments = segment_prediction(
            prediction,
            threshold=args.threshold,
            min_length=args.min_length,
            type_names=loaded.type_names,
            binary_class_name=loaded.binary_class_name,
        )

        if plot_dir is not None:
            plot_name = f"{n:04d}_{_safe_plot_stem(record.identifier)}_repeat_profile.png"
            plot_path = plot_dir / plot_name
            plot_profile(
                prediction.repeat_probability,
                ranges,
                save=True,
                save_path=plot_path,
                show=False,
                threshold=args.threshold,
                title=f"TREAD repeat profile: {record.identifier}",
            )

        for row in segments:
            row["sequence_id"] = record.identifier
            row["sequence_length"] = len(record.sequence)
            if row["class_score"] is None:
                row["class_score"] = ""
            all_segments.append(row)

        if not args.no_residue_scores:
            _write_residue_scores(
                residue_path, record.identifier, prediction, loaded.type_names
            )
        print(f"  predicted repeat segments: {len(segments)}")
        if plot_dir is not None:
            print(f"  plot: {plot_path}")

    _write_segments(segments_path, all_segments)
    print(f"\nDone. Segment annotations: {segments_path}")
    if not args.no_residue_scores:
        print(f"Residue-wise scores: {residue_path}")
    if plot_dir is not None:
        print(f"Plots: {plot_dir}")


def run_train(args):
    if args.train_overlap >= args.window_size:
        raise ValueError("--train-overlap must be smaller than --window-size")
    if args.embedding_overlap >= args.embedding_chunk_length:
        raise ValueError("--embedding-overlap must be smaller than --embedding-chunk-length")

    common = dict(
        fasta=args.fasta,
        annotations=args.annotations,
        outdir=args.outdir,
        device=args.device,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        window_size=args.window_size,
        train_overlap=args.train_overlap,
        embedding_chunk_length=args.embedding_chunk_length,
        embedding_overlap=args.embedding_overlap,
        out_channel=args.out_channel,
        hidden_dim=args.hidden_dim,
        num_block=args.num_block,
        dropout=args.dropout,
        kernel_size_conv1=args.kernel_size_conv1,
        kernel_size_block=args.kernel_size_block,
        input_noise_std=args.input_noise_std,
        label_scheme=args.label_scheme,
        edge_ratio=args.edge_ratio,
        edge_min=args.edge_min,
        command=" ".join(shlex.quote(arg) for arg in sys.argv),
    )

    if args.task == "binary":
        train_binary_model(
            **common,
            pos_weight=args.pos_weight,
        )
    elif args.task == "multitask":
        train_multitask_model(
            **common,
            seg_pos_weight=args.pos_weight,
            seg_loss_weight=args.seg_loss_weight,
            type_loss_weight=args.type_loss_weight,
        )
    else:
        raise ValueError(f"Unsupported training task: {args.task}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="tread",
        description="TREAD: protein repeat annotation and supervised model training",
        epilog=(
            "Examples:\n"
            "  tread predict protein.fasta\n"
            "  tread predict protein.fasta --model propeller-blade\n"
            "  tread predict protein.fasta --weights my_model/model.pt\n"
            "  tread train proteins.fasta annotations.tsv -o my_model\n"
            "  tread train proteins.fasta annotations.tsv --task multitask -o my_multitask_model\n\n"
            "Run 'tread predict --help' or 'tread train --help' for detailed options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"TREAD {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser(
        "predict",
        help="predict repeat regions from one or more protein sequences",
        description=(
            "Predict residue-wise repeat probabilities and repeat segments for "
            "all sequences in a protein FASTA file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    predict.add_argument("fasta", help="input protein FASTA file")
    predict.add_argument("-o", "--outdir", default="tread_results", help="output directory")
    predict.add_argument(
        "--model", choices=sorted(MODEL_SPECS), default="repeatsdb",
        help="bundled TREAD model to use when --weights is not supplied",
    )
    predict.add_argument(
        "--weights", default=None,
        help="custom TREAD checkpoint produced by 'tread train'; overrides --model",
    )
    predict.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="compute device; auto uses CUDA when available",
    )
    predict.add_argument("--threshold", type=_probability, default=0.8, help="residue probability threshold for segment extraction")
    predict.add_argument("--min-length", type=_positive_int, default=15, help="minimum predicted repeat-segment length")
    predict.add_argument("--embedding-chunk-length", type=_positive_int, default=1000, help="maximum ProtT5 chunk length for long proteins")
    predict.add_argument("--embedding-overlap", type=_nonnegative_int, default=100, help="overlap between ProtT5 chunks for long proteins")
    predict.add_argument("--no-residue-scores", action="store_true", help="write only segment annotations, not the residue-wise score table")
    predict.add_argument("--no-plot", action="store_true", help="do not generate per-sequence repeat-profile PNG plots")
    predict.set_defaults(func=run_predict)

    train = subparsers.add_parser(
        "train",
        help="train a TREAD model from user-supplied repeat annotations",
        description=(
            "Train either a binary residue-level repeat detector or a multitask "
            "repeat detector with repeat-type heads. Annotation coordinates are "
            "1-based and inclusive. FASTA sequences absent from the annotation table "
            "are treated as negatives."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train.add_argument("fasta", help="training protein FASTA file")
    train.add_argument("annotations", help="annotation TSV; binary requires sequence_id,start,end and multitask also requires repeat_type")
    train.add_argument("-o", "--outdir", default="tread_model", help="training output directory")
    train.add_argument("--task", choices=("binary", "multitask"), default="binary", help="training task")
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="compute device")
    train.add_argument("--validation-fraction", type=_fraction, default=0.1, help="fraction of proteins reserved for validation")
    train.add_argument("--seed", type=int, default=42, help="random seed")
    train.add_argument("--epochs", type=_positive_int, default=30, help="maximum training epochs")
    train.add_argument("--patience", type=_positive_int, default=5, help="early-stopping patience based on validation loss")
    train.add_argument("--batch-size", type=_positive_int, default=32, help="training batch size")
    train.add_argument("--learning-rate", type=_positive_float, default=1e-5, help="Adam learning rate")
    train.add_argument("--window-size", type=_positive_int, default=64, help="residue window length")
    train.add_argument("--train-overlap", type=_nonnegative_int, default=32, help="overlap between training windows")
    train.add_argument("--embedding-chunk-length", type=_positive_int, default=1000, help="maximum ProtT5 chunk length for long proteins")
    train.add_argument("--embedding-overlap", type=_nonnegative_int, default=100, help="overlap between ProtT5 chunks for long proteins")
    train.add_argument("--pos-weight", type=_positive_float, default=None, help="optional positive-class weight for BCE loss")
    train.add_argument(
        "--label-scheme", choices=("linear", "noedge"), default="linear",
        help="residue target scheme inside annotated repeat intervals; linear reproduces the original TREAD soft-edge labels",
    )
    train.add_argument(
        "--edge-ratio", type=_edge_ratio, default=0.1,
        help="fraction of each annotated interval tapered at both ends when --label-scheme linear",
    )
    train.add_argument(
        "--edge-min", type=_positive_probability, default=0.5,
        help="target value at the outermost annotated residues when --label-scheme linear",
    )
    train.add_argument("--seg-loss-weight", type=_positive_float, default=1.0, help="multitask segmentation-loss weight")
    train.add_argument("--type-loss-weight", type=_positive_float, default=10.0, help="multitask repeat-type-loss weight")
    train.add_argument("--out-channel", type=_positive_int, default=64, help="convolutional channel width")
    train.add_argument("--hidden-dim", type=_positive_int, default=64, help="hidden dimension before the prediction head")
    train.add_argument("--num-block", type=_positive_int, default=2, help="number of residual blocks")
    train.add_argument("--dropout", type=_probability, default=0.2, help="dropout probability")
    train.add_argument("--kernel-size-conv1", type=_positive_int, default=7, help="stem convolution kernel size")
    train.add_argument("--kernel-size-block", type=_positive_int, default=7, help="residual-block convolution kernel size")
    train.add_argument("--input-noise-std", type=_nonnegative_float, default=0.0, help="Gaussian noise SD applied to non-padding embeddings during training")
    train.set_defaults(func=run_train)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
