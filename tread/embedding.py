import re

import torch


DEFAULT_PROTT5_MODEL = "Rostlab/prot_t5_xl_uniref50"


def resolve_device(requested="auto"):
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


class ProtT5Embedder:
    """Generate residue-wise ProtT5 embeddings from amino-acid sequences."""

    def __init__(self, device="auto", model_name=DEFAULT_PROTT5_MODEL):
        # Lazy import keeps `tread --help` fast and avoids loading transformers
        # unless embedding generation is actually requested.
        from transformers import T5EncoderModel, T5Tokenizer

        self.device = resolve_device(device) if not isinstance(device, torch.device) else device
        self.model_name = model_name

        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name, do_lower_case=False, legacy=True
        )

        # On CUDA, fp16 substantially lowers memory usage. Embeddings are converted
        # back to float32 before they are passed to TREAD.
        if self.device.type == "cuda":
            self.model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float16)
        else:
            self.model = T5EncoderModel.from_pretrained(model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _prepare_sequence(sequence):
        # This is the standard ProtTrans preprocessing: rare amino acids are
        # represented as X and residues are separated by spaces for tokenization.
        sequence = re.sub(r"[UZOB]", "X", sequence.upper())
        return " ".join(sequence)

    def _embed_chunk(self, sequence):
        prepared = self._prepare_sequence(sequence)
        encoded = self.tokenizer(
            prepared,
            add_special_tokens=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.inference_mode():
            output = self.model(**encoded).last_hidden_state[0]

        # ProtT5 appends a special token. The first len(sequence) vectors map to
        # the protein residues.
        embedding = output[: len(sequence)].detach().float().cpu()
        if embedding.shape[0] != len(sequence):
            raise RuntimeError(
                f"ProtT5 returned {embedding.shape[0]} residue embeddings for a "
                f"sequence of length {len(sequence)}"
            )
        return embedding

    def embed(self, sequence, chunk_length=1000, overlap=100):
        """
        Embed a protein sequence.

        Long proteins are embedded as overlapping ProtT5 chunks and overlapping
        residue embeddings are averaged. This avoids a hard sequence-length limit
        imposed by a web service while retaining local context around chunk edges.
        """
        if chunk_length <= 0:
            raise ValueError("chunk_length must be > 0")
        if overlap < 0 or overlap >= chunk_length:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_length")

        if len(sequence) <= chunk_length:
            return self._embed_chunk(sequence)

        step = chunk_length - overlap
        embedding_sum = None
        counts = torch.zeros((len(sequence), 1), dtype=torch.float32)

        for start in range(0, len(sequence), step):
            end = min(start + chunk_length, len(sequence))
            chunk = sequence[start:end]
            chunk_embedding = self._embed_chunk(chunk)

            if embedding_sum is None:
                embedding_sum = torch.zeros(
                    (len(sequence), chunk_embedding.shape[1]), dtype=torch.float32
                )

            embedding_sum[start:end] += chunk_embedding
            counts[start:end] += 1.0

            if end == len(sequence):
                break

        if embedding_sum is None or torch.any(counts == 0):
            raise RuntimeError("Failed to generate embeddings for the full sequence")
        return embedding_sum / counts
