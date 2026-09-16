from __future__ import annotations

import csv
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import FastaRecord


@dataclass(frozen=True)
class Annotation:
    sequence_id: str
    start: int  # 1-based inclusive
    end: int    # 1-based inclusive


def _interval_labels(length: int, scheme: str = "linear", edge_ratio: float = 0.1, edge_min: float = 0.5) -> np.ndarray:
    """Create residue targets for one annotated repeat interval.

    The default ``linear`` scheme reproduces the original TREAD label
    assignment: the first and last ``int(length * edge_ratio)`` residues are
    linearly ramped from ``edge_min`` toward 1.0, while the interval core is
    1.0. ``noedge`` assigns 1.0 to the full interval.
    """
    if length <= 0:
        raise ValueError("Annotated interval length must be positive")
    if scheme not in {"linear", "noedge"}:
        raise ValueError("label scheme must be 'linear' or 'noedge'")
    if not 0.0 <= edge_ratio <= 0.5:
        raise ValueError("edge_ratio must be between 0 and 0.5")
    if not 0.0 < edge_min <= 1.0:
        raise ValueError("edge_min must be greater than 0 and at most 1")

    labels = np.ones(length, dtype=np.float32)
    if scheme == "noedge":
        return labels

    edge_length = int(length * edge_ratio)
    for i in range(edge_length):
        value = edge_min + (1.0 - edge_min) * (i / edge_length)
        labels[i] = value
        labels[-(i + 1)] = value
    return labels


def read_binary_annotations(
    path,
    records: Sequence[FastaRecord],
    label_scheme: str = "linear",
    edge_ratio: float = 0.1,
    edge_min: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Read 1-based inclusive repeat annotations and build residue-wise labels.

    FASTA records absent from the annotation table are treated as fully negative.
    The TSV must contain: sequence_id, start, end. By default, annotated
    intervals use the original TREAD linear-edge soft labels.
    """
    record_map = {record.identifier: record for record in records}
    if len(record_map) != len(records):
        raise ValueError("FASTA identifiers must be unique for training")

    labels = {
        record.identifier: np.zeros(len(record.sequence), dtype=np.float32)
        for record in records
    }

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sequence_id", "start", "end"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Annotation TSV must contain columns: sequence_id, start, end"
            )

        for line_number, row in enumerate(reader, start=2):
            sequence_id = (row.get("sequence_id") or "").strip()
            if sequence_id not in record_map:
                raise ValueError(
                    f"Annotation line {line_number}: sequence_id '{sequence_id}' "
                    "is not present in the FASTA file"
                )
            try:
                start = int(row["start"])
                end = int(row["end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Annotation line {line_number}: start/end must be integers"
                ) from exc

            seq_len = len(record_map[sequence_id].sequence)
            if start < 1 or end < start or end > seq_len:
                raise ValueError(
                    f"Annotation line {line_number}: invalid interval {start}-{end} "
                    f"for '{sequence_id}' (length {seq_len}). Coordinates must be "
                    "1-based and inclusive."
                )

            # Convert 1-based inclusive coordinates to Python [start, end).
            interval_target = _interval_labels(
                end - start + 1,
                scheme=label_scheme,
                edge_ratio=edge_ratio,
                edge_min=edge_min,
            )
            # Maximum composition is robust to overlapping user annotations and
            # preserves the strongest repeat target at each residue.
            current = labels[sequence_id][start - 1 : end]
            labels[sequence_id][start - 1 : end] = np.maximum(current, interval_target)

    if not any(label.any() for label in labels.values()):
        raise ValueError("No positive repeat residues were found in the annotations")
    return labels


def split_sequence_ids(
    labels: Dict[str, np.ndarray], validation_fraction: float = 0.1, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Deterministic protein-level train/validation split, stratified when possible."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(labels) < 2:
        raise ValueError("At least two protein sequences are required for training")

    rng = random.Random(seed)
    positive = [seq_id for seq_id, y in labels.items() if bool(y.any())]
    negative = [seq_id for seq_id, y in labels.items() if not bool(y.any())]
    rng.shuffle(positive)
    rng.shuffle(negative)

    def split_group(group: List[str]) -> Tuple[List[str], List[str]]:
        if not group:
            return [], []
        if len(group) == 1:
            return group[:], []
        n_val = max(1, int(round(len(group) * validation_fraction)))
        n_val = min(n_val, len(group) - 1)
        return group[n_val:], group[:n_val]

    pos_train, pos_val = split_group(positive)
    neg_train, neg_val = split_group(negative)
    train_ids = pos_train + neg_train
    val_ids = pos_val + neg_val

    # Very small datasets may have only one class with one sequence. Guarantee a
    # non-empty validation set while preserving protein-level separation.
    if not val_ids:
        rng.shuffle(train_ids)
        val_ids = [train_ids.pop()]

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    if not train_ids:
        raise ValueError("Training split is empty; provide more sequences")
    return train_ids, val_ids


def _safe_embedding_name(sequence_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", sequence_id).strip("._")[:80] or "sequence"
    digest = hashlib.sha1(sequence_id.encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}.npy"


def generate_embedding_cache(
    records: Sequence[FastaRecord],
    embedder,
    cache_dir,
    chunk_length: int = 1000,
    overlap: int = 100,
) -> Dict[str, Path]:
    """Generate (or reuse) per-protein ProtT5 embedding files."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    for idx, record in enumerate(records, start=1):
        path = cache_dir / _safe_embedding_name(record.identifier)
        paths[record.identifier] = path
        if path.exists():
            array = np.load(path, mmap_mode="r")
            if array.ndim == 2 and array.shape[0] == len(record.sequence) and array.shape[1] == 1024:
                print(f"[{idx}/{len(records)}] embedding cached: {record.identifier}")
                continue
            path.unlink()

        print(f"[{idx}/{len(records)}] embedding: {record.identifier} ({len(record.sequence)} aa)")
        embedding = embedder.embed(
            record.sequence, chunk_length=chunk_length, overlap=overlap
        )
        np.save(path, embedding.numpy().astype(np.float32, copy=False))

    return paths


@dataclass(frozen=True)
class WindowSpec:
    sequence_id: str
    start: int
    end: int


def build_windows(
    sequence_ids: Iterable[str],
    labels: Dict[str, np.ndarray],
    window_size: int = 64,
    overlap: int = 32,
) -> List[WindowSpec]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")
    step = window_size - overlap
    windows: List[WindowSpec] = []
    for sequence_id in sequence_ids:
        length = len(labels[sequence_id])
        for start in range(0, length, step):
            end = min(start + window_size, length)
            windows.append(WindowSpec(sequence_id, start, end))
            if end == length:
                break
    return windows


class BinaryWindowDataset(Dataset):
    """Windowed residue-level dataset backed by cached .npy embeddings."""

    def __init__(
        self,
        windows: Sequence[WindowSpec],
        embedding_paths: Dict[str, Path],
        labels: Dict[str, np.ndarray],
        window_size: int = 64,
    ):
        self.windows = list(windows)
        self.embedding_paths = embedding_paths
        self.labels = labels
        self.window_size = int(window_size)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        spec = self.windows[index]
        embedding = np.load(self.embedding_paths[spec.sequence_id], mmap_mode="r")
        x_slice = np.asarray(embedding[spec.start : spec.end], dtype=np.float32)
        y_slice = self.labels[spec.sequence_id][spec.start : spec.end]

        x = np.zeros((self.window_size, embedding.shape[1]), dtype=np.float32)
        y = np.zeros(self.window_size, dtype=np.float32)
        mask = np.zeros(self.window_size, dtype=np.bool_)
        valid = spec.end - spec.start
        x[:valid] = x_slice
        y[:valid] = y_slice
        mask[:valid] = True

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


@dataclass(frozen=True)
class MultitaskTargets:
    segmentation: Dict[str, np.ndarray]
    types: Dict[str, np.ndarray]
    type_names: Tuple[str, ...]


def read_multitask_annotations(
    path,
    records: Sequence[FastaRecord],
    label_scheme: str = "linear",
    edge_ratio: float = 0.1,
    edge_min: float = 0.5,
) -> MultitaskTargets:
    """Read repeat intervals with repeat types and construct multitask targets.

    The TSV must contain: sequence_id, start, end, repeat_type.
    Coordinates are 1-based and inclusive. Annotated residues are positive for
    segmentation and for the corresponding repeat type. By default, the
    annotated interval uses the original TREAD linear-edge soft labels. All
    other residues are segmentation negatives. Repeat-type loss is intended to
    be evaluated only on annotated repeat residues.
    """
    record_map = {record.identifier: record for record in records}
    if len(record_map) != len(records):
        raise ValueError("FASTA identifiers must be unique for training")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    parsed_rows = []
    type_name_set = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sequence_id", "start", "end", "repeat_type"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Multitask annotation TSV must contain columns: "
                "sequence_id, start, end, repeat_type"
            )

        for line_number, row in enumerate(reader, start=2):
            sequence_id = (row.get("sequence_id") or "").strip()
            repeat_type = (row.get("repeat_type") or "").strip()
            if sequence_id not in record_map:
                raise ValueError(
                    f"Annotation line {line_number}: sequence_id '{sequence_id}' "
                    "is not present in the FASTA file"
                )
            if not repeat_type:
                raise ValueError(
                    f"Annotation line {line_number}: repeat_type must not be empty"
                )
            try:
                start = int(row["start"])
                end = int(row["end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Annotation line {line_number}: start/end must be integers"
                ) from exc

            seq_len = len(record_map[sequence_id].sequence)
            if start < 1 or end < start or end > seq_len:
                raise ValueError(
                    f"Annotation line {line_number}: invalid interval {start}-{end} "
                    f"for '{sequence_id}' (length {seq_len}). Coordinates must be "
                    "1-based and inclusive."
                )
            parsed_rows.append((sequence_id, start, end, repeat_type))
            type_name_set.add(repeat_type)

    if not parsed_rows:
        raise ValueError("No repeat annotations were found")

    # A deterministic order is stored in the checkpoint and defines the output
    # head order for this trained model.
    type_names = tuple(sorted(type_name_set))
    type_to_index = {name: index for index, name in enumerate(type_names)}

    segmentation = {
        record.identifier: np.zeros(len(record.sequence), dtype=np.float32)
        for record in records
    }
    types = {
        record.identifier: np.zeros(
            (len(record.sequence), len(type_names)), dtype=np.float32
        )
        for record in records
    }

    for sequence_id, start, end, repeat_type in parsed_rows:
        start0 = start - 1
        end0 = end
        interval_target = _interval_labels(
            end - start + 1,
            scheme=label_scheme,
            edge_ratio=edge_ratio,
            edge_min=edge_min,
        )
        current_seg = segmentation[sequence_id][start0:end0]
        segmentation[sequence_id][start0:end0] = np.maximum(
            current_seg, interval_target
        )
        type_index = type_to_index[repeat_type]
        current_type = types[sequence_id][start0:end0, type_index]
        types[sequence_id][start0:end0, type_index] = np.maximum(
            current_type, interval_target
        )

    return MultitaskTargets(segmentation, types, type_names)


def split_multitask_sequence_ids(
    targets: MultitaskTargets, validation_fraction: float = 0.1, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Protein-level split stratified by the set of repeat types per protein."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(targets.segmentation) < 2:
        raise ValueError("At least two protein sequences are required for training")

    groups: Dict[Tuple[int, ...], List[str]] = {}
    for sequence_id, type_target in targets.types.items():
        signature = tuple(
            index for index in range(type_target.shape[1])
            if bool(type_target[:, index].any())
        )
        groups.setdefault(signature, []).append(sequence_id)

    rng = random.Random(seed)
    train_ids: List[str] = []
    val_ids: List[str] = []
    for signature in sorted(groups):
        group = groups[signature][:]
        rng.shuffle(group)
        if len(group) == 1:
            train_ids.extend(group)
            continue
        n_val = max(1, int(round(len(group) * validation_fraction)))
        n_val = min(n_val, len(group) - 1)
        val_ids.extend(group[:n_val])
        train_ids.extend(group[n_val:])

    if not val_ids:
        rng.shuffle(train_ids)
        val_ids = [train_ids.pop()]
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    if not train_ids:
        raise ValueError("Training split is empty; provide more sequences")
    return train_ids, val_ids


class MultitaskWindowDataset(Dataset):
    """Windowed segmentation + repeat-type targets backed by cached embeddings."""

    def __init__(
        self,
        windows: Sequence[WindowSpec],
        embedding_paths: Dict[str, Path],
        targets: MultitaskTargets,
        window_size: int = 64,
    ):
        self.windows = list(windows)
        self.embedding_paths = embedding_paths
        self.targets = targets
        self.window_size = int(window_size)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        spec = self.windows[index]
        embedding = np.load(self.embedding_paths[spec.sequence_id], mmap_mode="r")
        x_slice = np.asarray(embedding[spec.start : spec.end], dtype=np.float32)
        seg_slice = self.targets.segmentation[spec.sequence_id][spec.start : spec.end]
        type_slice = self.targets.types[spec.sequence_id][spec.start : spec.end]

        x = np.zeros((self.window_size, embedding.shape[1]), dtype=np.float32)
        seg_y = np.zeros(self.window_size, dtype=np.float32)
        type_y = np.zeros((self.window_size, len(self.targets.type_names)), dtype=np.float32)
        valid_mask = np.zeros(self.window_size, dtype=np.bool_)
        type_mask = np.zeros(
            (self.window_size, len(self.targets.type_names)), dtype=np.bool_
        )

        valid = spec.end - spec.start
        x[:valid] = x_slice
        seg_y[:valid] = seg_slice
        type_y[:valid] = type_slice
        valid_mask[:valid] = True
        # Type classification is meaningful only for annotated repeat residues.
        repeat_mask = seg_slice > 0.0
        type_mask[:valid] = repeat_mask[:, None]

        return (
            torch.from_numpy(x),
            torch.from_numpy(seg_y),
            torch.from_numpy(type_y),
            torch.from_numpy(valid_mask),
            torch.from_numpy(type_mask),
        )
