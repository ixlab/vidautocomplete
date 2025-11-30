"""FastAPI server that exposes an MSVD autocomplete demo."""
from __future__ import annotations

import argparse
import ast
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import duckdb
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_EMBEDDINGS_DIR = BASE_DIR / "embeddings" / "msvd"

MSVD_PHRASE_CSV = OUTPUTS_DIR / "msvd-pllava13b-llama3_3-70b-phrases-emojis-split-importance.csv"
MSVD_DESC_CSV = OUTPUTS_DIR / "msvd-pllava13b-descriptions-no-atmo.csv"
MSVD_EMBEDDINGS_DIR = Path(os.environ.get("MSVD_EMBEDDINGS_DIR", DEFAULT_EMBEDDINGS_DIR))

MIN_QUERY_WORDS = 1
TOP_K_RESULTS = 15
MAX_DESCRIPTION_CHARS = 200

duckdb_conn = duckdb.connect(database=":memory:")


def ensure_required_inputs() -> None:
    required = [MSVD_PHRASE_CSV, MSVD_DESC_CSV, MSVD_EMBEDDINGS_DIR]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required MSVD assets. Place the following files/directories before running:\n"
            + "\n".join(missing)
        )


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    records: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        phrase_payload = row.get("PhraseEmojis", "")
        try:
            parsed_items = ast.literal_eval(phrase_payload)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(parsed_items, list):
            continue
        for item in parsed_items:
            if not isinstance(item, dict):
                continue
            emojis = item.get("emojis", [])
            if any((isinstance(e, str) and not e.strip()) or e == "" for e in emojis):
                continue
            records.append(
                {
                    "vidID": row["Video ID"],
                    "phrase": item.get("phrase", ""),
                    "split": json.dumps(item.get("split", [])),
                    "emojis": json.dumps(emojis),
                    "importance": json.dumps(item.get("importance", [])),
                }
            )
    return pd.DataFrame.from_records(records, columns=["vidID", "phrase", "split", "emojis", "importance"])


def load_embeddings(folder: Path) -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}
    for file in folder.glob("*.npy"):
        embeddings[file.stem] = np.load(file)
    if not embeddings:
        raise FileNotFoundError(f"No embeddings (.npy) found inside {folder}")
    return embeddings


def load_descriptions(csv_path: Path) -> Dict[str, str]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return dict(zip(df["Video ID"], df["Answer"].fillna("")))


def search_phrases(term: str) -> pd.DataFrame:
    query = "SELECT phrase, split, emojis, importance FROM msvd WHERE phrase ILIKE ? LIMIT 40"
    return duckdb_conn.execute(query, [f"%{term}%"]).df()


def pick_emoji(row: pd.Series, term: str) -> str | None:
    try:
        splits = json.loads(row["split"])
        emojis = json.loads(row["emojis"])
        importance = json.loads(row.get("importance", "[]"))
    except json.JSONDecodeError:
        return None
    if not isinstance(emojis, list) or not emojis:
        return None
    term_clean = term.strip()
    term_lower = term_clean.lower()
    for idx, segment in enumerate(splits):
        if isinstance(segment, str) and term_lower in segment.lower():
            segment_len = max(len(segment.strip()), 1)
            ratio = len(term_clean) / segment_len
            if ratio < 0.6:
                return normalize_emoji(emojis, idx)
            fallback_idx = pick_by_importance(importance, exclude_idx=idx)
            if fallback_idx is not None:
                return normalize_emoji(emojis, fallback_idx)
            return normalize_emoji(emojis, idx)
    return normalize_emoji(emojis, 0)


def pick_by_importance(importance: List, exclude_idx: int) -> int | None:
    if not isinstance(importance, list):
        return None
    best_idx: int | None = None
    best_score = float("-inf")
    for idx, score in enumerate(importance):
        if idx == exclude_idx:
            continue
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if numeric_score > best_score:
            best_score = numeric_score
            best_idx = idx
    return best_idx


def normalize_emoji(emojis: List[str] | List[List[str]], idx: int) -> str | None:
    if not isinstance(emojis, list):
        return None
    if not emojis:
        return None
    if idx >= len(emojis):
        idx = 0
    candidate = emojis[idx]
    if isinstance(candidate, list):
        return "".join(candidate)
    if isinstance(candidate, str):
        return candidate
    return None


def knn_search(query: str, resources: "ResourceStore") -> List[Dict[str, str]]:
    assert resources.model is not None and resources.embeddings is not None and resources.descriptions is not None
    query_emb = resources.model.encode(query)
    q_norm = np.linalg.norm(query_emb) + 1e-8
    scored = []
    for vid, emb in resources.embeddings.items():
        score = float(np.dot(query_emb, emb) / (q_norm * (np.linalg.norm(emb) + 1e-8)))
        scored.append((vid, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    items = []
    for vid, score in scored[:TOP_K_RESULTS]:
        description = resources.descriptions.get(vid, "No description available")
        if len(description) > MAX_DESCRIPTION_CHARS:
            description = description[:MAX_DESCRIPTION_CHARS] + "..."
        items.append(
            {
                "video_url": f"/static/msvd-vids/{vid}.mp4",
                "thumbnail_url": f"/static/msvd-imgs/{vid}.jpg",
                "description": description,
                "score": score,
            }
        )
    return items


@dataclass
class AppConfig:
    mode: str = os.environ.get("AUTOCOMPLETE_MODE", "text")

    def set_mode(self, mode: str) -> None:
        if mode not in {"text", "emoji"}:
            raise ValueError("mode must be 'text' or 'emoji'")
        self.mode = mode


@dataclass
class ResourceStore:
    model: SentenceTransformer | None = None
    embeddings: Dict[str, np.ndarray] | None = None
    descriptions: Dict[str, str] | None = None
    initialized: bool = False

    def initialize(self) -> None:
        if self.initialized:
            return
        ensure_required_inputs()
        dataset = load_dataset(MSVD_PHRASE_CSV)
        duckdb_conn.register("msvd_df", dataset)
        duckdb_conn.execute("CREATE OR REPLACE TABLE msvd AS SELECT * FROM msvd_df")
        self.descriptions = load_descriptions(MSVD_DESC_CSV)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = load_embeddings(MSVD_EMBEDDINGS_DIR)
        self.initialized = True


config = AppConfig()
resources = ResourceStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    resources.initialize()
    yield


app = FastAPI(title="MSVD Autocomplete Demo", lifespan=lifespan)

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "autocomplete.html",
        {
            "request": request,
            "mode": config.mode,
            "min_words": MIN_QUERY_WORDS,
        },
    )


@app.get("/autocomplete")
async def autocomplete(term: str = Query("", min_length=1)):
    if not term:
        return []
    df = search_phrases(term)
    payload = []
    for _, row in df.iterrows():
        suggestion = {"label": row["phrase"], "value": row["phrase"]}
        if config.mode == "emoji":
            emoji = pick_emoji(row, term)
            suggestion["emoji"] = emoji
            suggestion["display_label"] = f"{emoji} {row['phrase']}" if emoji else row["phrase"]
        else:
            suggestion["display_label"] = row["phrase"]
        payload.append(suggestion)
    return payload[:20]


class SearchPayload(BaseModel):
    searchTxt: str


@app.post("/search")
async def search(payload: SearchPayload):
    query = payload.searchTxt.strip()
    if not query or len(query.split()) < MIN_QUERY_WORDS:
        return []
    return knn_search(query, resources)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MSVD autocomplete server")
    parser.add_argument("--mode", choices=["text", "emoji"], default=config.mode, help="Autocomplete suggestion style")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.set_mode(args.mode)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
