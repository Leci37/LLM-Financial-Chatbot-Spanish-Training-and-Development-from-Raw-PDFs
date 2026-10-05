"""Where the pipeline reads and writes, anchored at the repository root so the scripts in
pipeline/ work from any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw"              # the five bank documents, .pdf and the .docx made from them
PROCESSED = ROOT / "data" / "processed"  # JSON written by steps 1-3 (kept in git: it is the dataset)
OUTPUTS = ROOT / "outputs"               # trained models, checkpoints, logs (not in git)

OUTPUTS.mkdir(parents=True, exist_ok=True)
