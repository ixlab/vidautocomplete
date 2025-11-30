"""Utility for (re)building MSVD video text embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_DESC_CSV = OUTPUTS_DIR / "msvd-pllava13b-descriptions-no-atmo.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "user-study" / "embeddings" / "msvd"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MSVD embeddings for the autocomplete demo")
    parser.add_argument("--descriptions", type=Path, default=DEFAULT_DESC_CSV, help="CSV file containing MSVD descriptions")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store .npy embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer checkpoint")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .npy files")
    return parser.parse_args()


def read_descriptions(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Description CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df[["Video ID", "Answer"]].rename(columns={"Video ID": "vid", "Answer": "text"})
    df["text"] = df["text"].fillna("").str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("No descriptions with text were found in the provided CSV")
    return df


def encode_texts(model: SentenceTransformer, texts: Iterable[str], batch_size: int) -> np.ndarray:
    return model.encode(list(texts), batch_size=batch_size, show_progress_bar=True)


def save_embeddings(df: pd.DataFrame, matrix: np.ndarray, output_dir: Path, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for vid, vector in zip(df["vid"], matrix):
        target = output_dir / f"{vid}.npy"
        if target.exists() and not force:
            continue
        np.save(target, vector)


def main() -> None:
    args = parse_args()
    print(f"Loading descriptions from {args.descriptions}")
    df = read_descriptions(args.descriptions)
    print(f"Loaded {len(df)} descriptions. Initializing model {args.model}")
    model = SentenceTransformer(args.model)
    vectors = encode_texts(model, df["text"], args.batch_size)
    print(f"Writing {len(vectors)} embeddings to {args.output_dir}")
    save_embeddings(df, vectors, args.output_dir, args.force)
    print("Done.")


if __name__ == "__main__":
    main()
