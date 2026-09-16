from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FastaRecord:
    identifier: str
    description: str
    sequence: str


def read_fasta(path):
    """Read one or more protein sequences from a FASTA file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    records = []
    header = None
    chunks = []

    def flush_record():
        nonlocal header, chunks
        if header is None:
            return
        sequence = "".join(chunks).replace(" ", "").replace("\t", "").upper()
        if sequence.endswith("*"):
            sequence = sequence[:-1]
        if not sequence:
            raise ValueError(f"FASTA record '{header}' has an empty sequence")
        if not sequence.isalpha():
            raise ValueError(
                f"FASTA record '{header}' contains non-letter characters in the sequence"
            )
        identifier = header.split()[0]
        records.append(FastaRecord(identifier, header, sequence))
        header = None
        chunks = []

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                header = line[1:].strip()
                if not header:
                    raise ValueError("Encountered a FASTA header without an identifier")
            else:
                if header is None:
                    raise ValueError("Sequence encountered before the first FASTA header")
                chunks.append(line)
        flush_record()

    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records
