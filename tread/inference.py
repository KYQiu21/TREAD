from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .model import DMDModel
from .utils import get_ranges


REPEAT_TYPES = (
    "all-alpha solenoid",
    "TIM barrel",
    "beta-propeller",
    "beta-barrel",
    "all-beta solenoid",
    "alpha-beta solenoid",
)

MODEL_SPECS = {
    "repeatsdb": {
        "filename": "linear-edge_model_repeatsdb.pt",
        "multi": True,
        "type_names": REPEAT_TYPES,
        "binary_class_name": "repeat",
    },
    "propeller-blade": {
        "filename": "linear-edge_model_propeller_blade.pt",
        "multi": False,
        "type_names": (),
        "binary_class_name": "beta-propeller blade",
    },
}


@dataclass
class Prediction:
    repeat_probability: np.ndarray
    type_probabilities: list | None


@dataclass
class LoadedTreadModel:
    model: DMDModel
    type_names: tuple
    binary_class_name: str
    metadata: dict


def _bundled_weight_path(filename):
    package_path = Path(__file__).resolve().parent / "trained_model" / filename
    if package_path.exists():
        return package_path
    repo_path = Path(__file__).resolve().parent.parent / "trained_model" / filename
    if repo_path.exists():
        return repo_path
    raise FileNotFoundError(
        f"Could not find bundled TREAD weight '{filename}'. Expected it under "
        "tread/trained_model/."
    )


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint format. Expected a PyTorch state_dict or a "
            "dictionary containing 'model_state_dict'/'state_dict'."
        )
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def _infer_model_config(state_dict, device):
    conv1_weight = state_dict["conv1.weight"]
    out_channel = conv1_weight.shape[0]
    per_resi_emb_dim = conv1_weight.shape[1]
    kernel_size_conv1 = conv1_weight.shape[2]
    kernel_size_block = state_dict["conv2.weight"].shape[2]

    block_indices = set()
    for key in state_dict:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))
    num_block = len(block_indices)
    hidden_dim = state_dict["fc2.weight"].shape[0]

    type_indices = set()
    for key in state_dict:
        if key.startswith("type_heads."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                type_indices.add(int(parts[1]))
    multi = len(type_indices) > 0
    num_types = len(type_indices) if multi else 0
    bilstm = any(key.startswith("bilstm.") for key in state_dict)

    return {
        "per_resi_emb_dim": per_resi_emb_dim,
        "hidden_dim": hidden_dim,
        "out_channel": out_channel,
        "num_block": num_block,
        "kernel_size_conv1": kernel_size_conv1,
        "kernel_size_block": kernel_size_block,
        "bilstm": bilstm,
        "input_noise_std": 0.0,
        "multi": multi,
        "num_types": num_types,
        "device": device,
    }


def _normalize_model_config(config, device):
    allowed = {
        "per_resi_emb_dim", "hidden_dim", "out_channel", "num_block", "dropout",
        "kernel_size_conv1", "kernel_size_block", "bilstm", "input_noise_std",
        "multi", "num_types",
    }
    result = {k: v for k, v in dict(config).items() if k in allowed}
    result["device"] = device
    return result


def load_tread_model(model_name, device):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{model_name}'")
    spec = MODEL_SPECS[model_name]
    checkpoint = torch.load(_bundled_weight_path(spec["filename"]), map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    config = _infer_model_config(state_dict, device)
    model = DMDModel(**config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.device = device
    model.eval()
    return LoadedTreadModel(
        model=model,
        type_names=tuple(spec["type_names"]),
        binary_class_name=spec["binary_class_name"],
        metadata={"source": "bundled", "model_name": model_name},
    )


def load_custom_tread_model(path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TREAD checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_config"), dict):
        config = _normalize_model_config(checkpoint["model_config"], device)
    else:
        config = _infer_model_config(state_dict, device)

    model = DMDModel(**config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.device = device
    model.eval()

    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    type_names = tuple(metadata.get("type_names", ()))
    if model.multi and len(type_names) != model.num_types:
        type_names = tuple(f"repeat_type_{i + 1}" for i in range(model.num_types))
    return LoadedTreadModel(
        model=model,
        type_names=type_names,
        binary_class_name=metadata.get("binary_class_name", "repeat"),
        metadata=metadata,
    )


def predict_embedding(model, embedding):
    device = next(model.parameters()).device
    embedding = embedding.to(device)
    with torch.inference_mode():
        output = model.predict_single(embedding)
    if model.multi:
        repeat_probability, type_probabilities = output
        return Prediction(repeat_probability, type_probabilities)
    return Prediction(output, None)


def segment_prediction(
    prediction,
    threshold=0.8,
    min_length=15,
    type_names=(),
    binary_class_name="repeat",
):
    ranges = get_ranges(
        prediction.repeat_probability,
        cutoff1=threshold,
        min_len=min_length,
        cutoff2=threshold,
        frac2=0.5,
    )
    rows = []
    for repeat_index, (start0, end0) in enumerate(ranges, start=1):
        mean_repeat_score = float(np.mean(prediction.repeat_probability[start0:end0]))
        repeat_class = binary_class_name
        class_score = None
        if prediction.type_probabilities is not None:
            type_means = [float(np.mean(p[start0:end0])) for p in prediction.type_probabilities]
            best = int(np.argmax(type_means))
            repeat_class = type_names[best] if best < len(type_names) else f"repeat_type_{best + 1}"
            class_score = type_means[best]
        rows.append({
            "repeat_index": repeat_index,
            "start": start0 + 1,
            "end": end0,
            "length": end0 - start0,
            "mean_repeat_score": mean_repeat_score,
            "repeat_class": repeat_class,
            "class_score": class_score,
        })
    return rows
