#!/usr/bin/env python3
"""
Run vanilla single-shot RAG over a local chunk_index.pkl and write predictions JSONL.


Example:
  python semantic_infer.py \
    --questions data_baselines/2wiki_questions_500.json \
    --chunk_index_pkl data_baselines/2wiki_index.pkl \
    --output results_2wiki/semantic.jsonl \
    --vllm_model Qwen/Qwen2.5-3B-Instruct \
    --vllm_base_url http://127.0.0.1:8000/v1 \
    --topk 5
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import torch
from tqdm import tqdm


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

        q_emb = self.encoder.encode([query])  # [1, D], normalized
        scores = (self.embeddings @ q_emb[0]).astype(np.float32)

        n = scores.shape[0]
        k = min(topk, n)
        idxs = np.argpartition(-scores, kth=k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]

        docs: List[Dict[str, Any]] = []
        for idx in idxs.tolist():
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
                }
            )
        return docs


def build_retriever(chunk_index_pkl: Path, device: str, encoder_model: Optional[str]) -> ChunkIndexRetriever:
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

    print("[Chunk index loaded]")
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


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in questions file, got {type(data)}")
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


def format_context_docs(docs: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, d in enumerate(docs, start=1):
        title = d.get("title", "")
        text = d.get("text", "")
        lines.append(f"[Doc {i}] Title: {title}")
        lines.append(str(text).strip())
        lines.append("")
    return "\n".join(lines).strip()


def build_messages(question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context = format_context_docs(docs)
    system_prompt = (
        "Answer the given question with some potentially useful context. \
You should analyze the question carefully, evaluate the given context (which may or may not be useful), and then generate an accurate and well-reasoned response. \
You should first have a reasoning process in mind and then provides the answer. \
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>."
        
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"   
    )
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


def main() -> None:
    p = argparse.ArgumentParser(description="Vanilla RAG inference over chunk_index.pkl + vLLM chat endpoint")
    p.add_argument("--questions", required=True, help="Path to question JSON file (list of objects)")
    p.add_argument("--chunk_index_pkl", required=True, help="Path to chunk_index.pkl")
    p.add_argument("--output", required=True, help="Output predictions JSONL path")
    p.add_argument("--topk", type=int, default=5, help="Retrieved chunks per question")
    p.add_argument("--limit", type=int, default=None, help="Optional number of questions to run")
    p.add_argument("--resume", action="store_true", help="Append mode; skip qids already in output")
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for embedding query encoder",
    )
    p.add_argument(
        "--encoder_model",
        default=None,
        help="Override query encoder model (if pickle lacks model_name)",
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
    p.add_argument(
        "--vllm_model",
        required=True,
        help="Model name exposed by vLLM /v1/chat/completions",
    )
    p.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature for vLLM generation")
    p.add_argument("--max_tokens", type=int, default=1024, help="Max generated tokens")
    p.add_argument("--request_timeout", type=int, default=120, help="HTTP timeout in seconds for vLLM requests")
    args = p.parse_args()

    questions_path = Path(args.questions)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_questions(questions_path)
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

    retriever = build_retriever(
        chunk_index_pkl=Path(args.chunk_index_pkl),
        device=args.device,
        encoder_model=args.encoder_model,
    )

    with out_path.open(file_mode, encoding="utf-8") as f:
        for item in tqdm(items, desc="Vanilla-RAG"):
            qid = item.get("qid") or item.get("id")
            question = ensure_question_mark(item.get("question", ""))
            gold = item.get("answer") or item.get("gold_answer") or ""

            retrieved_docs = retriever.search(question, topk=max(args.topk, 5))
            prompt_docs = retrieved_docs[: args.topk]
            top5_retrieved_chunks = retrieved_docs[:5]
            retrieved_chunk_texts = [str(doc.get("text", "")).strip() for doc in top5_retrieved_chunks]
            messages = build_messages(question, prompt_docs)
            pred = call_vllm_chat(
                base_url=args.vllm_base_url,
                api_key=args.vllm_api_key or None,
                model=args.vllm_model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
            )

            f.write(
                json.dumps(
                    {
                        "qid": qid,
                        "question": question,
                        "gold_answer": gold,
                        "pred_answer_raw": pred,
                        "top5_retrieved_chunks": top5_retrieved_chunks,
                        # "retrieved_chunk_texts": retrieved_chunk_texts,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            os.fsync(f.fileno())


if __name__ == "__main__":
    main()
