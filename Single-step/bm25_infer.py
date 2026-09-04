#!/usr/bin/env python3
"""
Run vanilla single-shot RAG using BM25 retrieval over chunks.json and write predictions JSONL.

Example:
  python vanilla_rag_bm25_vllm_infer.py \
    --questions /project/pi_dagarwal_umass_edu/project_22/stodmal/VRAG/data_new/questions.json \
    --chunks /project/pi_dagarwal_umass_edu/project_22/stodmal/VRAG/data_new/chunks.json \
    --output results/hotpotqa_v2/vanilla_rag_bm25/predictions_test_1000.jsonl \
    --vllm_model Qwen/Qwen2.5-7B-Instruct \
    --vllm_base_url http://127.0.0.1:8000/v1 \
    --topk 5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class BM25Retriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
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
                }
            )
        return docs


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in questions file, got {type(data)}")
    return data


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in chunks file, got {type(data)}")
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
        lines.append(f"[Doc {i}] Title: {d.get('title', '')}")
        lines.append(str(d.get("text", "")).strip())
        lines.append("")
    return "\n".join(lines).strip()


def build_messages(question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context = format_context_docs(docs)
    system_prompt = (
        "Answer the given question with some potentially useful context. "
        "You should analyze the question carefully, evaluate the given context (which may or may not be useful), and then generate an accurate and well-reasoned response. "
        "You should first have a reasoning process in mind and then provides the answer. "
        "Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>."
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


def run_one_question(
    item: Dict[str, Any],
    *,
    retriever: BM25Retriever,
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

    retrieved_docs = retriever.search(question, topk=max(topk, 5))
    prompt_docs = retrieved_docs[:topk]
    top5_retrieved_chunks = retrieved_docs[:5]
    messages = build_messages(question, prompt_docs)

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
        "top5_retrieved_chunks": top5_retrieved_chunks,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Vanilla RAG inference over chunks.json with BM25 + vLLM chat endpoint")
    p.add_argument("--questions", required=True, help="Path to question JSON file (list of objects)")
    p.add_argument("--chunks", required=True, help="Path to chunks JSON file (list of objects)")
    p.add_argument("--output", required=True, help="Output predictions JSONL path")
    p.add_argument("--topk", type=int, default=5, help="Retrieved chunks per question")
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Questions per scheduling batch (for progress + ordered writes)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Concurrent vLLM requests; set 1 for sequential mode",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional number of questions to run")
    p.add_argument("--resume", action="store_true", help="Append mode; skip qids already in output")
    p.add_argument("--bm25_k1", type=float, default=1.5, help="BM25 k1 parameter")
    p.add_argument("--bm25_b", type=float, default=0.75, help="BM25 b parameter")
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
    chunks_path = Path(args.chunks)
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

    chunks = load_chunks(chunks_path)
    retriever = BM25Retriever(chunks, k1=args.bm25_k1, b=args.bm25_b)
    print("[BM25 index loaded]")
    print(f"  chunks: {len(chunks)}")
    print(f"  avg_doc_len: {retriever.avg_doc_len:.2f}")
    print(f"  k1={args.bm25_k1}, b={args.bm25_b}")

    with out_path.open(file_mode, encoding="utf-8") as f:
        batch_size = max(1, args.batch_size)
        num_workers = max(1, args.num_workers)
        for start in tqdm(range(0, len(items), batch_size), desc="Vanilla-RAG-BM25 batches"):
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
                if result is None:
                    continue
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


if __name__ == "__main__":
    main()
