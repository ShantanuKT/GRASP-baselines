#!/usr/bin/env python3
"""
FastAPI semantic retrieval server backed directly by chunk_index.pkl.

This avoids sentence-level retrieval and any mapping back to external chunk JSON.
It keeps the same /retrieve request/response structure as the sentence-index servers:

POST /retrieve
{
  "queries": ["..."],
  "topk": 5,
  "return_scores": false
}
-> { "result": [ [ {title,text,contents}, ... ], ... ] }
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / denom


def _to_numpy_embeddings(emb: Any) -> np.ndarray:
    if isinstance(emb, np.ndarray):
        arr = emb
    elif torch.is_tensor(emb):
        arr = emb.detach().cpu().numpy()
    else:
        arr = np.asarray(emb)
    return arr.astype(np.float32, copy=False)


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


class _STEncoder:
    def __init__(self, model_name: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, queries: List[str]) -> np.ndarray:
        emb = self.model.encode(
            queries,
            batch_size=min(64, max(1, len(queries))),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return _to_numpy_embeddings(emb)


class _HFMeanPoolEncoder:
    def __init__(self, model_name: str, device: str, max_length: int = 256):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, queries: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs, return_dict=True)
        last_hidden = out.last_hidden_state  # [B, T, H]
        mask = inputs.get(
            "attention_mask", torch.ones(last_hidden.shape[:2], device=last_hidden.device)
        )
        mask_f = mask.unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask_f).sum(dim=1) / (mask_f.sum(dim=1).clamp_min(1e-6))
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


@dataclass
class AragChunkIndexRetriever:
    chunk_ids: List[Any]  # row i -> chunk id
    embeddings: np.ndarray  # [N, D] float32, L2-normalized
    chunks_by_id: Dict[str, Dict[str, Any]]  # chunk id -> {id,title,text,...}
    encoder: Any

    def search(self, query: str, topk: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        if topk <= 0:
            return [], []

        q_emb = self.encoder.encode([query])  # [1, D], normalized
        scores = (self.embeddings @ q_emb[0]).astype(np.float32)  # cosine if normalized

        n = scores.shape[0]
        k = min(topk, n)
        if k <= 0:
            return [], []

        idxs = np.argpartition(-scores, kth=k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]

        docs: List[Dict[str, Any]] = []
        out_scores: List[float] = []
        for idx in idxs.tolist():
            cid = str(self.chunk_ids[idx]) if idx < len(self.chunk_ids) else str(idx)
            ch = self.chunks_by_id.get(cid) or {}
            title = str(ch.get("title") or cid)
            text = str(ch.get("text") or ch.get("contents") or "")
            contents = f"{title}\n{text}".strip()
            docs.append({"id": cid, "title": title, "text": text, "contents": contents})
            out_scores.append(float(scores[idx]))

        return docs, out_scores

    def batch_search(
        self, queries: List[str], topk: int
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[float]]]:
        if not queries:
            return [], []
        if topk <= 0:
            return [[] for _ in queries], [[] for _ in queries]

        q_emb = self.encoder.encode(queries)  # [B, D], normalized
        all_docs: List[List[Dict[str, Any]]] = []
        all_scores: List[List[float]] = []

        # Compute per-query topk; keep it simple (B is usually small for API usage).
        for i in range(q_emb.shape[0]):
            scores = (self.embeddings @ q_emb[i]).astype(np.float32)  # [N]
            n = scores.shape[0]
            k = min(topk, n)
            if k <= 0:
                all_docs.append([])
                all_scores.append([])
                continue

            idxs = np.argpartition(-scores, kth=k - 1)[:k]
            idxs = idxs[np.argsort(-scores[idxs])]

            docs: List[Dict[str, Any]] = []
            out_scores: List[float] = []
            for idx in idxs.tolist():
                cid = str(self.chunk_ids[idx]) if idx < len(self.chunk_ids) else str(idx)
                ch = self.chunks_by_id.get(cid) or {}
                title = str(ch.get("title") or cid)
                text = str(ch.get("text") or ch.get("contents") or "")
                contents = f"{title}\n{text}".strip()
                docs.append({"id": cid, "title": title, "text": text, "contents": contents})
                out_scores.append(float(scores[idx]))

            all_docs.append(docs)
            all_scores.append(out_scores)

        return all_docs, all_scores


def _coerce_chunks_by_id(chunks_obj: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(chunks_obj, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in chunks_obj.items():
            out[str(k)] = v if isinstance(v, dict) else {"id": k, "text": str(v)}
        return out
    if isinstance(chunks_obj, list):
        out = {}
        for i, item in enumerate(chunks_obj):
            if isinstance(item, dict):
                cid = item.get("id") or item.get("chunk_id") or i
                out[str(cid)] = item
            else:
                out[str(i)] = {"id": i, "text": str(item)}
        return out
    return {}


def build_retriever(
    chunk_index_pkl: Path,
    device: str,
    encoder_model: Optional[str],
) -> AragChunkIndexRetriever:
    with chunk_index_pkl.open("rb") as f:
        index_data = pickle.load(f)

    if not isinstance(index_data, dict):
        raise ValueError(f"chunk_index.pkl must be a dict; got {type(index_data)}")

    chunk_ids = index_data.get("chunk_ids")
    embeddings = index_data.get("embeddings")
    chunks_obj = index_data.get("chunks")
    model_name = index_data.get("model_name") or encoder_model

    if chunk_ids is None or embeddings is None:
        raise ValueError("chunk_index.pkl must contain at least 'chunk_ids' and 'embeddings'")
    if not isinstance(chunk_ids, list):
        raise ValueError(f"Expected 'chunk_ids' to be a list, got {type(chunk_ids)}")
    if not model_name:
        raise ValueError(
            "Could not determine encoder model. Provide --encoder_model or ensure pickle has 'model_name'."
        )

    emb_np = _to_numpy_embeddings(embeddings)
    emb_np = _l2_normalize(emb_np)
    chunks_by_id = _coerce_chunks_by_id(chunks_obj)

    # Prefer SentenceTransformer for compatibility with how indexes are usually built.
    try:
        encoder = _STEncoder(model_name, device=device)
        encoder_kind = "sentence-transformers"
    except Exception:
        encoder = _HFMeanPoolEncoder(model_name, device=device)
        encoder_kind = "transformers-meanpool"

    print("[Chunk Index Retriever]")
    print(f"  index: {chunk_index_pkl}")
    if isinstance(index_data.get('input_chunks_json'), str):
        print(f"  input_chunks_json: {index_data.get('input_chunks_json')}")
    if isinstance(index_data.get('text_format'), str):
        print(f"  text_format: {index_data.get('text_format')}")
    print(f"  chunks: {len(chunks_by_id)}")
    print(f"  embeddings: {emb_np.shape} (normalized)")
    print(f"  encoder: {model_name} ({encoder_kind}) on {device}")

    return AragChunkIndexRetriever(
        chunk_ids=chunk_ids,
        embeddings=emb_np,
        chunks_by_id=chunks_by_id,
        encoder=encoder,
    )


app = FastAPI()
retriever: Optional[AragChunkIndexRetriever] = None
default_topk: int = 3


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    if retriever is None:
        raise RuntimeError("Retriever not initialized")

    topk = request.topk or default_topk
    docs, scores = retriever.batch_search(request.queries, topk)

    resp = []
    for i, single_docs in enumerate(docs):
        if request.return_scores:
            resp.append([{"document": d, "score": s} for d, s in zip(single_docs, scores[i])])
        else:
            resp.append(single_docs)
    return {"result": resp}


def main() -> None:
    global retriever, default_topk

    p = argparse.ArgumentParser(description="Launch chunk-level retrieval server from chunk_index.pkl")
    p.add_argument(
        "--chunk_index_pkl",
        required=True,
        help="Path to chunk_index.pkl",
    )
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--encoder_model",
        default=None,
        help="Override encoder model (if pickle lacks model_name)",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()

    default_topk = args.topk
    retriever = build_retriever(
        chunk_index_pkl=Path(args.chunk_index_pkl),
        device=args.device,
        encoder_model=args.encoder_model,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

