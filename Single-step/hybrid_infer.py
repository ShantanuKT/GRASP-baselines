#!/usr/bin/env python3
"""
Run vanilla single-shot RAG using hybrid retrieval:
  1) Retrieve N BM25 candidates from chunks.json
  2) Retrieve N semantic candidates from chunk_index.pkl
  3) Merge + deduplicate candidates
  4) Rerank candidate pool with Qwen3-Reranker-0.6B
  5) Keep final top-k docs for prompt, then call vLLM chat endpoint

Example:
  python hybrid_infer.py \
    --questions data_baselines/2wiki_questions_500.json \
    --chunks data_baselines/2wiki_chunks.json \
    --chunk_index_pkl data_baselines/2wiki_index.pkl \
    --output results_2wiki/hybrid.jsonl \
    --vllm_model Qwen/Qwen2.5-3B-Instruct \
    --vllm_base_url http://127.0.0.1:8000/v1 \
    --topk 5 \
    --candidate_k 50 \
    --reranker_model Qwen/Qwen3-Reranker-0.6B
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Avoid stray Hugging Face Hub HTTP calls during AutoTokenizer/AutoModel loading
# (e.g. transformers 4.57.x's ``is_base_mistral`` check) on compute nodes that
# can't reach huggingface.co. Models must already be present in the local HF
# cache. Set ``HF_HUB_OFFLINE=0`` in the env to disable this.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import requests
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# -----------------------------
# Shared utilities
# -----------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _safe_pickle_load(f) -> Any:
    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    return _CompatUnpickler(f).load()


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


def load_json_list(path: Path, name: str) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {name}, got {type(data)}")
    return data


def item_qid_key(item: Dict[str, Any]) -> Optional[str]:
    q = item.get("qid") or item.get("id")
    return str(q) if q is not None else None


def load_completed_qids_from_jsonl(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = obj.get("qid")
            if q is not None:
                done.add(str(q))
    return done


def ensure_question_mark(q: str) -> str:
    q = (q or "").strip()
    if q and not q.endswith("?"):
        q += "?"
    return q


def canonical_doc_key(doc: Dict[str, Any]) -> str:
    """Dedup by id when available; otherwise use normalized title+text."""
    doc_id = doc.get("id")
    if doc_id is not None and str(doc_id).strip():
        return f"id::{str(doc_id)}"
    title = str(doc.get("title") or "").strip().lower()
    text = re.sub(r"\s+", " ", str(doc.get("text") or "").strip().lower())
    return f"text::{title}::{text[:500]}"


def format_context_docs(docs: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, d in enumerate(docs, start=1):
        lines.append(f"[Doc {i}] Title: {d.get('title', '')}")
        lines.append(str(d.get("text", "")).strip())
        lines.append("")
    return "\n".join(lines).strip()


def build_messages(question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context = format_context_docs(docs)
    system_prompt = (
        "Answer the given question with some potentially useful context. "
        "You should analyze the question carefully, evaluate the given context "
        "(which may or may not be useful), and then generate an accurate and well-reasoned response. "
        "You should first have a reasoning process in mind and then provides the answer. "
        "Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, "
        "for example <answer> Beijing </answer>."
    )
    user_prompt = f"Question: {question}\n\nContext:\n{context}\n\n"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def call_vllm_chat(
    *,
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "").strip()


# -----------------------------
# BM25 retriever
# -----------------------------

class BM25Retriever:
    def __init__(self, chunks: List[Dict[str, Any]], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.doc_count = len(chunks)
        self.tokenized_docs: List[List[str]] = [
            _tokenize(str(chunk.get("text") or "")) for chunk in chunks
        ]
        self.doc_lens: List[int] = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_len = (sum(self.doc_lens) / self.doc_count) if self.doc_count > 0 else 0.0
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)

    def search(self, query: str, topk: int) -> List[Dict[str, Any]]:
        if topk <= 0 or self.doc_count == 0:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []

        scores = self.bm25.get_scores(q_terms)
        ranked = sorted(range(self.doc_count), key=lambda i: float(scores[i]), reverse=True)[
            : min(topk, self.doc_count)
        ]

        docs: List[Dict[str, Any]] = []
        for idx in ranked:
            c = self.chunks[idx]
            cid = c.get("id", idx)
            docs.append(
                {
                    "id": str(cid),
                    "title": str(c.get("title") or f"chunk_{cid}"),
                    "text": str(c.get("text") or ""),
                    "score": float(scores[idx]),
                    "retriever": "bm25",
                    "bm25_rank": len(docs) + 1,
                    "bm25_score": float(scores[idx]),
                }
            )
        return docs


# -----------------------------
# Semantic retriever
# -----------------------------

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

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True, trust_remote_code=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False, trust_remote_code=True
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
        last_hidden = out.last_hidden_state
        mask = inputs.get(
            "attention_mask",
            torch.ones(last_hidden.shape[:2], device=last_hidden.device),
        )
        mask_f = mask.unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask_f).sum(dim=1) / (mask_f.sum(dim=1).clamp_min(1e-6))
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


def _get_chunk_obj_by_id(chunks_map: Dict[Any, Any], chunk_id: Any) -> Dict[str, Any]:
    if chunk_id in chunks_map and isinstance(chunks_map[chunk_id], dict):
        return chunks_map[chunk_id]
    sid = str(chunk_id)
    if sid in chunks_map and isinstance(chunks_map[sid], dict):
        return chunks_map[sid]
    try:
        iid = int(chunk_id)
    except Exception:
        iid = None
    if iid is not None and iid in chunks_map and isinstance(chunks_map[iid], dict):
        return chunks_map[iid]
    return {}


class ChunkIndexRetriever:
    def __init__(
        self,
        *,
        chunk_ids: List[Any],
        chunk_texts: List[str],
        chunks_map: Dict[Any, Any],
        embeddings: np.ndarray,
        encoder: Any,
    ):
        self.chunk_ids = chunk_ids
        self.chunk_texts = chunk_texts
        self.chunks_map = chunks_map
        self.embeddings = embeddings
        self.encoder = encoder

    def search(self, query: str, topk: int) -> List[Dict[str, Any]]:
        if topk <= 0 or self.embeddings.shape[0] == 0:
            return []

        q_emb = self.encoder.encode([query])
        scores = (self.embeddings @ q_emb[0]).astype(np.float32)

        n = scores.shape[0]
        k = min(topk, n)
        idxs = np.argpartition(-scores, kth=k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]

        docs: List[Dict[str, Any]] = []
        for rank, idx in enumerate(idxs.tolist(), start=1):
            chunk_id = self.chunk_ids[idx] if idx < len(self.chunk_ids) else idx
            chunk_obj = _get_chunk_obj_by_id(self.chunks_map, chunk_id)
            title = str(chunk_obj.get("title") or f"chunk_{chunk_id}")
            text = str(chunk_obj.get("text") or self.chunk_texts[idx])
            docs.append(
                {
                    "id": str(chunk_id),
                    "title": title,
                    "text": text,
                    "score": float(scores[idx]),
                    "retriever": "semantic",
                    "semantic_rank": rank,
                    "semantic_score": float(scores[idx]),
                }
            )
        return docs


def build_semantic_retriever(
    chunk_index_pkl: Path,
    device: str,
    encoder_model: Optional[str],
) -> ChunkIndexRetriever:
    with chunk_index_pkl.open("rb") as f:
        index_data = _safe_pickle_load(f)

    if not isinstance(index_data, dict):
        raise ValueError(f"chunk_index.pkl must be a dict; got {type(index_data)}")

    chunk_texts = index_data.get("chunk_texts")
    embeddings = index_data.get("embeddings")
    chunk_ids = index_data.get("chunk_ids")
    chunks_map = index_data.get("chunks") or {}
    model_name = index_data.get("model_name") or encoder_model

    if chunk_texts is None or embeddings is None or chunk_ids is None:
        raise ValueError("chunk_index.pkl must contain 'chunk_texts', 'chunk_ids', and 'embeddings'.")
    if not isinstance(chunk_texts, list):
        raise ValueError(f"Expected 'chunk_texts' to be a list, got {type(chunk_texts)}")
    if not isinstance(chunk_ids, list):
        raise ValueError(f"Expected 'chunk_ids' to be a list, got {type(chunk_ids)}")
    if not model_name:
        raise ValueError("Could not determine encoder model. Pass --encoder_model or include model_name in pickle.")

    emb_np = _l2_normalize(_to_numpy_embeddings(embeddings))

    try:
        encoder = _STEncoder(model_name, device=device)
        encoder_kind = "sentence-transformers"
    except Exception:
        encoder = _HFMeanPoolEncoder(model_name, device=device)
        encoder_kind = "transformers-meanpool"

    print("[Semantic chunk index loaded]")
    print(f"  index: {chunk_index_pkl}")
    print(f"  chunks: {len(chunk_ids)}")
    print(f"  embeddings: {emb_np.shape} (normalized)")
    print(f"  encoder: {model_name} ({encoder_kind}) on {device}")

    return ChunkIndexRetriever(
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        chunks_map=chunks_map if isinstance(chunks_map, dict) else {},
        embeddings=emb_np,
        encoder=encoder,
    )


# -----------------------------
# Qwen3 reranker
# -----------------------------

class Qwen3Reranker:
    """
    Generative reranker for Qwen3-Reranker models, following the official
    "Using Transformers" recipe on the model card.

    NOTE: we deliberately do NOT use ``sentence_transformers.CrossEncoder`` here,
    because in many sentence-transformers versions it loads Qwen3-Reranker as a
    generic ``Qwen3ForSequenceClassification`` model with a randomly initialized
    score head, which produces meaningless rerank scores (and warns:
    ``Some weights ... ['score.weight'] ... newly initialized``).

    Instead we load Qwen3-Reranker as the AutoModelForCausalLM it actually is,
    format each (query, document) pair through the official chat-template style
    prompt, take the last-token logits, and score with ``softmax([no, yes])[..., 1]``.
    """

    DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    PREFIX = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and "
        "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
        "<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int = 8192,
        batch_size: int = 8,
        instruction: Optional[str] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction or self.DEFAULT_INSTRUCTION

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and "cuda" in str(device) else None
        load_kwargs: Dict[str, Any] = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self.model.to(self.device)

        self.token_yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no_id = self.tokenizer.convert_tokens_to_ids("no")
        if self.token_yes_id is None or self.token_no_id is None:
            raise RuntimeError(
                "Qwen3Reranker: could not resolve 'yes'/'no' token ids from tokenizer."
            )

        self.prefix_ids: List[int] = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_ids: List[int] = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)

        self.kind = "qwen3-causal-yes-no"

    @classmethod
    def _format(cls, instruction: str, query: str, doc: str) -> str:
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    @torch.no_grad()
    def _score_batch(self, batch_texts: List[str]) -> List[float]:
        inner_max = max(8, self.max_length - len(self.prefix_ids) - len(self.suffix_ids))
        inputs = self.tokenizer(
            batch_texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=inner_max,
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_ids + list(ids) + self.suffix_ids

        padded = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_length
        )
        padded = {k: v.to(self.device) for k, v in padded.items()}

        last_logits = self.model(**padded).logits[:, -1, :]
        yes_logits = last_logits[:, self.token_yes_id]
        no_logits = last_logits[:, self.token_no_id]
        stacked = torch.stack([no_logits, yes_logits], dim=1).float()
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        scores = log_probs[:, 1].exp()
        return [float(x) for x in scores.detach().cpu().tolist()]

    def score(self, query: str, docs: List[Dict[str, Any]]) -> List[float]:
        if not docs:
            return []

        texts: List[str] = [
            self._format(
                self.instruction,
                query,
                (str(d.get("title") or "") + "\n" + str(d.get("text") or "")).strip(),
            )
            for d in docs
        ]

        all_scores: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            all_scores.extend(self._score_batch(texts[start : start + self.batch_size]))
        return all_scores


def merge_and_dedup_candidates(
    bm25_docs: List[Dict[str, Any]],
    semantic_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for doc in bm25_docs + semantic_docs:
        key = canonical_doc_key(doc)
        if key not in merged:
            merged[key] = dict(doc)
            merged[key]["retrievers"] = [doc.get("retriever", "unknown")]
        else:
            existing = merged[key]
            r = doc.get("retriever", "unknown")
            if r not in existing.setdefault("retrievers", []):
                existing["retrievers"].append(r)
            for field in ["bm25_rank", "bm25_score", "semantic_rank", "semantic_score"]:
                if field in doc and field not in existing:
                    existing[field] = doc[field]

    return list(merged.values())


class HybridRetriever:
    def __init__(
        self,
        *,
        bm25_retriever: BM25Retriever,
        semantic_retriever: ChunkIndexRetriever,
        reranker: Qwen3Reranker,
        candidate_k: int,
    ):
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(self, query: str, topk: int) -> List[Dict[str, Any]]:
        bm25_docs = self.bm25_retriever.search(query, topk=self.candidate_k)
        semantic_docs = self.semantic_retriever.search(query, topk=self.candidate_k)
        candidates = merge_and_dedup_candidates(bm25_docs, semantic_docs)

        rerank_scores = self.reranker.score(query, candidates)
        for doc, score in zip(candidates, rerank_scores):
            doc["rerank_score"] = float(score)

        candidates.sort(key=lambda d: float(d.get("rerank_score", float("-inf"))), reverse=True)
        final_docs = candidates[:topk]

        for rank, doc in enumerate(final_docs, start=1):
            doc["hybrid_rank"] = rank
        return final_docs


# -----------------------------
# Inference loop
# -----------------------------

def run_one_question(
    item: Dict[str, Any],
    *,
    retriever: HybridRetriever,
    topk: int,
    vllm_base_url: str,
    vllm_api_key: Optional[str],
    vllm_model: str,
    temperature: float,
    max_tokens: int,
    request_timeout: int,
) -> Dict[str, Any]:
    qid = item.get("qid") or item.get("id")
    question = ensure_question_mark(item.get("question", ""))
    gold = item.get("answer") or item.get("gold_answer") or ""

    retrieved_docs = retriever.search(question, topk=topk)
    messages = build_messages(question, retrieved_docs)

    pred = call_vllm_chat(
        base_url=vllm_base_url,
        api_key=vllm_api_key,
        model=vllm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=request_timeout,
    )

    return {
        "qid": qid,
        "question": question,
        "gold_answer": gold,
        "pred_answer_raw": pred,
        "top5_retrieved_chunks": retrieved_docs[:5],
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Vanilla RAG inference with hybrid BM25 + semantic retrieval + Qwen3 reranking."
    )
    p.add_argument("--questions", required=True, help="Path to question JSON file (list of objects)")
    p.add_argument("--chunks", required=True, help="Path to chunks JSON file for BM25")
    p.add_argument("--chunk_index_pkl", required=True, help="Path to semantic chunk_index.pkl")
    p.add_argument("--output", required=True, help="Output predictions JSONL path")
    p.add_argument("--topk", type=int, default=5, help="Final reranked docs kept for prompt")
    p.add_argument("--candidate_k", type=int, default=50, help="Candidates retrieved from each retriever before merge")
    p.add_argument("--limit", type=int, default=None, help="Optional number of questions to run")
    p.add_argument("--resume", action="store_true", help="Append mode; skip qids already in output")

    p.add_argument("--bm25_k1", type=float, default=1.5, help="BM25 k1 parameter")
    p.add_argument("--bm25_b", type=float, default=0.75, help="BM25 b parameter")

    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for semantic encoder and reranker",
    )
    p.add_argument(
        "--encoder_model",
        default=None,
        help="Override query encoder model if chunk_index.pkl lacks model_name",
    )
    p.add_argument(
        "--reranker_model",
        default="Qwen/Qwen3-Reranker-0.6B",
        help="Qwen reranker model name/path",
    )
    p.add_argument(
        "--reranker_max_length",
        type=int,
        default=2048,
        help="Max tokens for the full reranker prompt (system + user + query + doc).",
    )
    p.add_argument(
        "--reranker_batch_size",
        type=int,
        default=8,
        help="Reranker batch size (Qwen3-Reranker is a 0.6B CausalLM run forward-only).",
    )
    p.add_argument(
        "--reranker_instruction",
        type=str,
        default=None,
        help="Optional task instruction injected into Qwen3-Reranker's <Instruct> field.",
    )

    p.add_argument(
        "--vllm_base_url",
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible vLLM base URL ending with /v1",
    )
    p.add_argument(
        "--vllm_api_key",
        default=os.getenv("VLLM_API_KEY", ""),
        help="Optional API key for vLLM endpoint",
    )
    p.add_argument("--vllm_model", required=True, help="Model name exposed by vLLM /v1/chat/completions")
    p.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature for vLLM generation")
    p.add_argument("--max_tokens", type=int, default=1024, help="Max generated tokens")
    p.add_argument("--request_timeout", type=int, default=120, help="HTTP timeout in seconds for vLLM requests")

    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Concurrent vLLM requests. Keep 1 if using one GPU for reranker to avoid contention.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Questions per scheduling batch when num_workers > 1",
    )
    args = p.parse_args()

    if args.topk <= 0:
        raise ValueError("--topk must be > 0")
    if args.candidate_k <= 0:
        raise ValueError("--candidate_k must be > 0")

    questions_path = Path(args.questions)
    chunks_path = Path(args.chunks)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_json_list(questions_path, "questions file")
    if args.limit is not None:
        items = items[: args.limit]

    file_mode = "w"
    if args.resume:
        completed = load_completed_qids_from_jsonl(out_path)
        n_before = len(items)
        items = [it for it in items if (k := item_qid_key(it)) is None or k not in completed]
        skipped = n_before - len(items)
        if skipped:
            print(f"Resume: skipping {skipped} questions already in {out_path}, {len(items)} remaining.")
        file_mode = "a" if out_path.is_file() else "w"

    if not items:
        print("No questions to run (empty list or all already completed).")
        return

    chunks = load_json_list(chunks_path, "chunks file")
    bm25_retriever = BM25Retriever(chunks, k1=args.bm25_k1, b=args.bm25_b)
    print("[BM25 index loaded]")
    print(f"  chunks: {len(chunks)}")
    print(f"  avg_doc_len: {bm25_retriever.avg_doc_len:.2f}")
    print(f"  k1={args.bm25_k1}, b={args.bm25_b}")

    semantic_retriever = build_semantic_retriever(
        chunk_index_pkl=Path(args.chunk_index_pkl),
        device=args.device,
        encoder_model=args.encoder_model,
    )

    reranker = Qwen3Reranker(
        model_name=args.reranker_model,
        device=args.device,
        max_length=args.reranker_max_length,
        batch_size=args.reranker_batch_size,
        instruction=args.reranker_instruction,
    )
    print("[Reranker loaded]")
    print(f"  model: {args.reranker_model}")
    print(f"  kind: {reranker.kind}")
    print(f"  device: {args.device}")
    print(f"  candidate_k: {args.candidate_k} from BM25 + {args.candidate_k} from semantic")
    print(f"  final topk: {args.topk}")

    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        semantic_retriever=semantic_retriever,
        reranker=reranker,
        candidate_k=args.candidate_k,
    )

    with out_path.open(file_mode, encoding="utf-8") as f:
        num_workers = max(1, args.num_workers)
        batch_size = max(1, args.batch_size)

        if num_workers == 1:
            for item in tqdm(items, desc="Vanilla-RAG-Hybrid"):
                qid = item.get("qid") or item.get("id")
                question = ensure_question_mark(item.get("question", ""))
                gold = item.get("answer") or item.get("gold_answer") or ""
                try:
                    result = run_one_question(
                        item,
                        retriever=retriever,
                        topk=args.topk,
                        vllm_base_url=args.vllm_base_url,
                        vllm_api_key=args.vllm_api_key or None,
                        vllm_model=args.vllm_model,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        request_timeout=args.request_timeout,
                    )
                except Exception as e:
                    result = {
                        "qid": qid,
                        "question": question,
                        "gold_answer": gold,
                        "pred_answer_raw": "",
                        "top5_retrieved_chunks": [],
                        "error": f"{type(e).__name__}: {e}",
                    }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        else:
            for start in tqdm(range(0, len(items), batch_size), desc="Vanilla-RAG-Hybrid batches"):
                batch = items[start : start + batch_size]
                ordered_results: List[Optional[Dict[str, Any]]] = [None] * len(batch)
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    future_to_idx = {
                        executor.submit(
                            run_one_question,
                            item,
                            retriever=retriever,
                            topk=args.topk,
                            vllm_base_url=args.vllm_base_url,
                            vllm_api_key=args.vllm_api_key or None,
                            vllm_model=args.vllm_model,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            request_timeout=args.request_timeout,
                        ): i
                        for i, item in enumerate(batch)
                    }
                    for future in concurrent.futures.as_completed(future_to_idx):
                        i = future_to_idx[future]
                        item = batch[i]
                        qid = item.get("qid") or item.get("id")
                        question = ensure_question_mark(item.get("question", ""))
                        gold = item.get("answer") or item.get("gold_answer") or ""
                        try:
                            ordered_results[i] = future.result()
                        except Exception as e:
                            ordered_results[i] = {
                                "qid": qid,
                                "question": question,
                                "gold_answer": gold,
                                "pred_answer_raw": "",
                                "top5_retrieved_chunks": [],
                                "error": f"{type(e).__name__}: {e}",
                            }

                for result in ordered_results:
                    if result is not None:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())


if __name__ == "__main__":
    main()
