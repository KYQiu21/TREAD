from __future__ import annotations

import csv
import json
import platform
from copy import deepcopy
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import __version__
from .embedding import DEFAULT_PROTT5_MODEL, ProtT5Embedder, resolve_device
from .io import read_fasta
from .model import DMDModel
from .training_data import (
    BinaryWindowDataset,
    MultitaskWindowDataset,
    build_windows,
    generate_embedding_cache,
    read_binary_annotations,
    read_multitask_annotations,
    split_sequence_ids,
    split_multitask_sequence_ids,
)


def _masked_bce_loss(logits, targets, mask, pos_weight=None):
    weight = None
    if pos_weight is not None:
        weight = torch.tensor(float(pos_weight), device=logits.device)
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=weight
    )
    valid = mask.bool()
    if not valid.any():
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return loss[valid].mean()


def _binary_metrics(logits: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    pred = probs >= threshold
    truth = targets > 0.0
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


def _run_epoch(model, loader, device, optimizer=None, pos_weight=None):
    training = optimizer is not None
    model.train(training)
    losses = []
    all_logits = []
    all_targets = []

    with torch.set_grad_enabled(training):
        for x, y, mask in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device)

            _, logits = model(x)
            loss = _masked_bce_loss(logits, y, mask, pos_weight=pos_weight)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            losses.append(float(loss.detach().cpu()))
            valid = mask.detach().cpu().numpy().astype(bool)
            logits_np = logits.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            all_logits.append(logits_np[valid])
            all_targets.append(y_np[valid])

    if not losses:
        raise RuntimeError("No training windows were generated")
    logits = np.concatenate(all_logits) if all_logits else np.empty(0)
    targets = np.concatenate(all_targets) if all_targets else np.empty(0)
    metrics = _binary_metrics(logits, targets)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def _write_ids(path, sequence_ids):
    path = Path(path)
    with path.open("w") as handle:
        for sequence_id in sequence_ids:
            handle.write(f"{sequence_id}\n")


def _write_history(path, history):
    path = Path(path)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_f1",
        "val_precision",
        "val_recall",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(history)


def train_binary_model(
    fasta,
    annotations,
    outdir,
    device="auto",
    validation_fraction=0.1,
    seed=42,
    epochs=30,
    patience=5,
    batch_size=32,
    learning_rate=1e-5,
    window_size=64,
    train_overlap=32,
    embedding_chunk_length=1000,
    embedding_overlap=100,
    out_channel=64,
    hidden_dim=64,
    num_block=2,
    dropout=0.2,
    kernel_size_conv1=7,
    kernel_size_block=7,
    input_noise_std=0.0,
    pos_weight=None,
    label_scheme="linear",
    edge_ratio=0.1,
    edge_min=0.5,
    command=None,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    records = read_fasta(fasta)
    labels = read_binary_annotations(
        annotations, records, label_scheme=label_scheme, edge_ratio=edge_ratio, edge_min=edge_min
    )
    train_ids, val_ids = split_sequence_ids(labels, validation_fraction, seed)
    _write_ids(outdir / "train_ids.txt", train_ids)
    _write_ids(outdir / "validation_ids.txt", val_ids)

    n_positive = sum(bool(y.any()) for y in labels.values())
    n_negative = len(labels) - n_positive
    print(f"TREAD training: {len(records)} proteins ({n_positive} annotated, {n_negative} unannotated negatives)")
    print(f"Train/validation proteins: {len(train_ids)}/{len(val_ids)}")
    print(
        f"Label scheme: {label_scheme} "
        f"(edge ratio={edge_ratio:g}, edge minimum={edge_min:g})"
    )
    print(f"Compute device: {device}")

    cache_dir = outdir / "embedding_cache"
    print("Loading ProtT5 and preparing residue embeddings...")
    embedder = ProtT5Embedder(device=device)
    embedding_paths = generate_embedding_cache(
        records,
        embedder,
        cache_dir,
        chunk_length=embedding_chunk_length,
        overlap=embedding_overlap,
    )
    # Free the large language model before the supervised model is trained.
    del embedder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_windows = build_windows(
        train_ids, labels, window_size=window_size, overlap=train_overlap
    )
    val_windows = build_windows(val_ids, labels, window_size=window_size, overlap=0)
    print(f"Training/validation windows: {len(train_windows)}/{len(val_windows)}")

    train_dataset = BinaryWindowDataset(train_windows, embedding_paths, labels, window_size)
    val_dataset = BinaryWindowDataset(val_windows, embedding_paths, labels, window_size)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model_config = {
        "per_resi_emb_dim": 1024,
        "hidden_dim": int(hidden_dim),
        "out_channel": int(out_channel),
        "num_block": int(num_block),
        "dropout": float(dropout),
        "kernel_size_conv1": int(kernel_size_conv1),
        "kernel_size_block": int(kernel_size_block),
        "bilstm": True,
        "device": str(device),
        "input_noise_std": float(input_noise_std),
        "multi": False,
        "num_types": 0,
    }
    model = DMDModel(**model_config).to(device)
    model.device = device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    best_epoch = 0
    best_val_metrics = None
    best_state = deepcopy(model.state_dict())
    no_improve = 0
    history = []
    checkpoint_path = outdir / "model.pt"

    training_config = {
        "task": "binary",
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
        "epochs": int(epochs),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "window_size": int(window_size),
        "train_overlap": int(train_overlap),
        "embedding_chunk_length": int(embedding_chunk_length),
        "embedding_overlap": int(embedding_overlap),
        "pos_weight": None if pos_weight is None else float(pos_weight),
        "label_scheme": str(label_scheme),
        "edge_ratio": float(edge_ratio),
        "edge_min": float(edge_min),
    }

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, device, optimizer=optimizer, pos_weight=pos_weight
        )
        val_metrics = _run_epoch(
            model, val_loader, device, optimizer=None, pos_weight=pos_weight
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(row)
        _write_history(outdir / "training_history.tsv", history)
        print(
            f"Epoch {epoch}/{epochs} | train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | val F1 {val_metrics['f1']:.4f}"
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_state = deepcopy(model.state_dict())
            no_improve = 0
            checkpoint = {
                "format_version": 1,
                "state_dict": best_state,
                "model_config": {**model_config, "device": "cpu"},
                "task": "binary",
                "type_names": [],
                "embedding_model": DEFAULT_PROTT5_MODEL,
                "command": command,
                "coordinate_system": "1-based-inclusive",
                "recommended_threshold": 0.8,
                "training_config": training_config,
                "tread_version": __version__,
                "best_epoch": best_epoch,
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {patience} epoch(s) without improvement.")
                break

    model.load_state_dict(best_state)
    epochs_completed = len(history)
    summary = {
        "format_version": 1,
        "task": "binary",
        "tread_version": __version__,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "embedding_model": DEFAULT_PROTT5_MODEL,
        "command": command,
        "annotation_coordinates": "1-based inclusive",
        "input_fasta": str(Path(fasta)),
        "input_annotations": str(Path(annotations)),
        "runtime_device": str(device),
        "proteins_total": len(records),
        "proteins_train": len(train_ids),
        "proteins_validation": len(val_ids),
        "positive_proteins": n_positive,
        "negative_proteins": n_negative,
        "training_windows": len(train_windows),
        "validation_windows": len(val_windows),
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "best_validation_loss": best_loss,
        "best_validation_metrics": best_val_metrics or {},
        "training_config": training_config,
        "model_config": {**model_config, "device": "cpu"},
        "train_ids_file": str(outdir / "train_ids.txt"),
        "validation_ids_file": str(outdir / "validation_ids.txt"),
        "checkpoint": str(checkpoint_path),
        "training_history": str(outdir / "training_history.tsv"),
        "embedding_cache": str(cache_dir),
    }
    with (outdir / "training_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nTraining complete. Best model: {checkpoint_path}")
    print(f"Training history: {outdir / 'training_history.tsv'}")
    print(f"Summary: {outdir / 'training_summary.json'}")
    print(f"Training split IDs: {outdir / 'train_ids.txt'}")
    print(f"Validation split IDs: {outdir / 'validation_ids.txt'}")
    print(f"Embedding cache: {cache_dir} (can be deleted after training)")
    return checkpoint_path


def _multitask_metrics(
    seg_logits: np.ndarray,
    seg_targets: np.ndarray,
    type_logits: np.ndarray,
    type_targets: np.ndarray,
    threshold: float = 0.5,
):
    seg = _binary_metrics(seg_logits, seg_targets, threshold=threshold)
    if type_logits.size:
        type_metrics = _binary_metrics(
            type_logits.reshape(-1), type_targets.reshape(-1), threshold=threshold
        )
    else:
        type_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "seg_f1": seg["f1"],
        "seg_precision": seg["precision"],
        "seg_recall": seg["recall"],
        "type_f1": type_metrics["f1"],
        "type_precision": type_metrics["precision"],
        "type_recall": type_metrics["recall"],
    }


def _run_multitask_epoch(
    model,
    loader,
    device,
    optimizer=None,
    seg_loss_weight=1.0,
    type_loss_weight=10.0,
    seg_pos_weight=None,
):
    training = optimizer is not None
    model.train(training)
    total_losses = []
    seg_losses = []
    type_losses = []
    all_seg_logits = []
    all_seg_targets = []
    all_type_logits = []
    all_type_targets = []

    with torch.set_grad_enabled(training):
        for x, seg_y, type_y, valid_mask, type_mask in loader:
            x = x.to(device=device, dtype=torch.float32)
            seg_y = seg_y.to(device=device, dtype=torch.float32)
            type_y = type_y.to(device=device, dtype=torch.float32)
            valid_mask = valid_mask.to(device=device)
            type_mask = type_mask.to(device=device)

            output = model(x)
            seg_logits = output["seg_logit"]
            type_logits = output["type_logit"]

            seg_loss = _masked_bce_loss(
                seg_logits, seg_y, valid_mask, pos_weight=seg_pos_weight
            )
            type_loss = _masked_bce_loss(
                type_logits, type_y, type_mask, pos_weight=None
            )
            total_loss = (
                float(seg_loss_weight) * seg_loss
                + float(type_loss_weight) * type_loss
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            total_losses.append(float(total_loss.detach().cpu()))
            seg_losses.append(float(seg_loss.detach().cpu()))
            type_losses.append(float(type_loss.detach().cpu()))

            seg_valid = valid_mask.detach().cpu().numpy().astype(bool)
            seg_logits_np = seg_logits.detach().cpu().numpy()
            seg_y_np = seg_y.detach().cpu().numpy()
            all_seg_logits.append(seg_logits_np[seg_valid])
            all_seg_targets.append(seg_y_np[seg_valid])

            type_valid = type_mask.detach().cpu().numpy().astype(bool)
            type_logits_np = type_logits.detach().cpu().numpy()
            type_y_np = type_y.detach().cpu().numpy()
            if type_valid.any():
                all_type_logits.append(type_logits_np[type_valid])
                all_type_targets.append(type_y_np[type_valid])

    if not total_losses:
        raise RuntimeError("No training windows were generated")

    seg_logits = np.concatenate(all_seg_logits) if all_seg_logits else np.empty(0)
    seg_targets = np.concatenate(all_seg_targets) if all_seg_targets else np.empty(0)
    type_logits = np.concatenate(all_type_logits) if all_type_logits else np.empty(0)
    type_targets = np.concatenate(all_type_targets) if all_type_targets else np.empty(0)
    metrics = _multitask_metrics(
        seg_logits, seg_targets, type_logits, type_targets, threshold=0.5
    )
    metrics.update(
        {
            "loss": float(np.mean(total_losses)),
            "seg_loss": float(np.mean(seg_losses)),
            "type_loss": float(np.mean(type_losses)),
        }
    )
    return metrics


def _write_multitask_history(path, history):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_seg_loss",
        "train_type_loss",
        "val_loss",
        "val_seg_loss",
        "val_type_loss",
        "val_seg_f1",
        "val_seg_precision",
        "val_seg_recall",
        "val_type_f1",
        "val_type_precision",
        "val_type_recall",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(history)


def train_multitask_model(
    fasta,
    annotations,
    outdir,
    device="auto",
    validation_fraction=0.1,
    seed=42,
    epochs=30,
    patience=5,
    batch_size=32,
    learning_rate=1e-5,
    window_size=64,
    train_overlap=32,
    embedding_chunk_length=1000,
    embedding_overlap=100,
    out_channel=64,
    hidden_dim=64,
    num_block=2,
    dropout=0.2,
    kernel_size_conv1=3,
    kernel_size_block=7,
    input_noise_std=0.0,
    seg_pos_weight=None,
    seg_loss_weight=1.0,
    type_loss_weight=10.0,
    label_scheme="linear",
    edge_ratio=0.1,
    edge_min=0.5,
    command=None,
):
    """Train a segmentation + repeat-type TREAD model.

    Segmentation loss is calculated for every non-padding residue. Repeat-type
    loss is calculated only at residues covered by at least one annotation.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    records = read_fasta(fasta)
    targets = read_multitask_annotations(
        annotations, records, label_scheme=label_scheme, edge_ratio=edge_ratio, edge_min=edge_min
    )
    train_ids, val_ids = split_multitask_sequence_ids(
        targets, validation_fraction, seed
    )
    _write_ids(outdir / "train_ids.txt", train_ids)
    _write_ids(outdir / "validation_ids.txt", val_ids)

    n_positive = sum(bool(y.any()) for y in targets.segmentation.values())
    n_negative = len(targets.segmentation) - n_positive
    type_protein_counts = {
        name: sum(
            bool(type_y[:, idx].any()) for type_y in targets.types.values()
        )
        for idx, name in enumerate(targets.type_names)
    }
    type_residue_counts = {
        name: int(sum((type_y[:, idx] > 0.0).sum() for type_y in targets.types.values()))
        for idx, name in enumerate(targets.type_names)
    }

    print(
        f"TREAD multitask training: {len(records)} proteins "
        f"({n_positive} annotated, {n_negative} unannotated negatives)"
    )
    print(f"Repeat types ({len(targets.type_names)}): {', '.join(targets.type_names)}")
    print(f"Train/validation proteins: {len(train_ids)}/{len(val_ids)}")
    print(
        f"Label scheme: {label_scheme} "
        f"(edge ratio={edge_ratio:g}, edge minimum={edge_min:g})"
    )
    print(f"Compute device: {device}")

    cache_dir = outdir / "embedding_cache"
    print("Loading ProtT5 and preparing residue embeddings...")
    embedder = ProtT5Embedder(device=device)
    embedding_paths = generate_embedding_cache(
        records,
        embedder,
        cache_dir,
        chunk_length=embedding_chunk_length,
        overlap=embedding_overlap,
    )
    del embedder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_windows = build_windows(
        train_ids,
        targets.segmentation,
        window_size=window_size,
        overlap=train_overlap,
    )
    val_windows = build_windows(
        val_ids, targets.segmentation, window_size=window_size, overlap=0
    )
    print(f"Training/validation windows: {len(train_windows)}/{len(val_windows)}")

    train_dataset = MultitaskWindowDataset(
        train_windows, embedding_paths, targets, window_size
    )
    val_dataset = MultitaskWindowDataset(
        val_windows, embedding_paths, targets, window_size
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model_config = {
        "per_resi_emb_dim": 1024,
        "hidden_dim": int(hidden_dim),
        "out_channel": int(out_channel),
        "num_block": int(num_block),
        "dropout": float(dropout),
        "kernel_size_conv1": int(kernel_size_conv1),
        "kernel_size_block": int(kernel_size_block),
        "bilstm": True,
        "device": str(device),
        "input_noise_std": float(input_noise_std),
        "multi": True,
        "num_types": len(targets.type_names),
    }
    model = DMDModel(**model_config).to(device)
    model.device = device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_config = {
        "task": "multitask",
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
        "epochs": int(epochs),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "window_size": int(window_size),
        "train_overlap": int(train_overlap),
        "embedding_chunk_length": int(embedding_chunk_length),
        "embedding_overlap": int(embedding_overlap),
        "seg_pos_weight": None if seg_pos_weight is None else float(seg_pos_weight),
        "seg_loss_weight": float(seg_loss_weight),
        "type_loss_weight": float(type_loss_weight),
        "type_loss_scope": "annotated-repeat-residues-only",
        "label_scheme": str(label_scheme),
        "edge_ratio": float(edge_ratio),
        "edge_min": float(edge_min),
    }

    best_loss = float("inf")
    best_epoch = 0
    best_val_metrics = None
    best_state = deepcopy(model.state_dict())
    no_improve = 0
    history = []
    checkpoint_path = outdir / "model.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = _run_multitask_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            seg_loss_weight=seg_loss_weight,
            type_loss_weight=type_loss_weight,
            seg_pos_weight=seg_pos_weight,
        )
        val_metrics = _run_multitask_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            seg_loss_weight=seg_loss_weight,
            type_loss_weight=type_loss_weight,
            seg_pos_weight=seg_pos_weight,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_seg_loss": train_metrics["seg_loss"],
            "train_type_loss": train_metrics["type_loss"],
            "val_loss": val_metrics["loss"],
            "val_seg_loss": val_metrics["seg_loss"],
            "val_type_loss": val_metrics["type_loss"],
            "val_seg_f1": val_metrics["seg_f1"],
            "val_seg_precision": val_metrics["seg_precision"],
            "val_seg_recall": val_metrics["seg_recall"],
            "val_type_f1": val_metrics["type_f1"],
            "val_type_precision": val_metrics["type_precision"],
            "val_type_recall": val_metrics["type_recall"],
        }
        history.append(row)
        _write_multitask_history(outdir / "training_history.tsv", history)
        print(
            f"Epoch {epoch}/{epochs} | train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"seg F1 {val_metrics['seg_f1']:.4f} | "
            f"type F1 {val_metrics['type_f1']:.4f}"
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_state = deepcopy(model.state_dict())
            no_improve = 0
            checkpoint = {
                "format_version": 1,
                "state_dict": best_state,
                "model_config": {**model_config, "device": "cpu"},
                "task": "multitask",
                "type_names": list(targets.type_names),
                "binary_class_name": "repeat",
                "embedding_model": DEFAULT_PROTT5_MODEL,
                "command": command,
                "coordinate_system": "1-based-inclusive",
                "recommended_threshold": 0.8,
                "training_config": training_config,
                "tread_version": __version__,
                "best_epoch": best_epoch,
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {patience} epoch(s) without improvement.")
                break

    model.load_state_dict(best_state)
    epochs_completed = len(history)
    summary = {
        "format_version": 1,
        "task": "multitask",
        "tread_version": __version__,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "embedding_model": DEFAULT_PROTT5_MODEL,
        "command": command,
        "annotation_coordinates": "1-based inclusive",
        "input_fasta": str(Path(fasta)),
        "input_annotations": str(Path(annotations)),
        "runtime_device": str(device),
        "proteins_total": len(records),
        "proteins_train": len(train_ids),
        "proteins_validation": len(val_ids),
        "positive_proteins": n_positive,
        "negative_proteins": n_negative,
        "repeat_types": list(targets.type_names),
        "repeat_type_protein_counts": type_protein_counts,
        "repeat_type_residue_counts": type_residue_counts,
        "training_windows": len(train_windows),
        "validation_windows": len(val_windows),
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "best_validation_loss": best_loss,
        "best_validation_metrics": best_val_metrics or {},
        "training_config": training_config,
        "model_config": {**model_config, "device": "cpu"},
        "train_ids_file": str(outdir / "train_ids.txt"),
        "validation_ids_file": str(outdir / "validation_ids.txt"),
        "checkpoint": str(checkpoint_path),
        "training_history": str(outdir / "training_history.tsv"),
        "embedding_cache": str(cache_dir),
    }
    with (outdir / "training_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nTraining complete. Best model: {checkpoint_path}")
    print(f"Training history: {outdir / 'training_history.tsv'}")
    print(f"Summary: {outdir / 'training_summary.json'}")
    print(f"Training split IDs: {outdir / 'train_ids.txt'}")
    print(f"Validation split IDs: {outdir / 'validation_ids.txt'}")
    print(f"Embedding cache: {cache_dir} (can be deleted after training)")
    return checkpoint_path
