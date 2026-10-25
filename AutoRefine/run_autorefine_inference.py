#!/usr/bin/env python3
"""
AutoRefine inference on a QA test set with an in-process dense retriever.

This script reproduces the AutoRefine "search-and-refine-during-think" loop
(see AutoRefine/infer.py) but, instead of talking to an HTTP retrieval server,
it performs retrieval directly in-process against a precomputed chunk index
(``data_baselines/2wiki_index.pkl``). This keeps everything in a single job.

The chunk index pickle is a dict with keys:
    - ``embeddings``    : np.ndarray [N, D], one row per chunk (L2-normalized)
    - ``chunk_ids``     : list[int/str], row i -> chunk id
    - ``chunks``        : dict[id -> {"id", "title", "text"}]
    - ``model_name``    : the sentence-embedding model used to build the index
    - ``chunk_texts``   : list[str], the raw text encoded for each row

Queries are encoded with the same embedding model (``model_name``) and scored by
cosine similarity (dot product on normalized vectors), matching the retrieval
recipe in ``search_r1/search/arag_chunk_index_server.py``.

Output: one JSON object per line (JSONL) with the fields:
    id, question, ground_truth, predicted, retrieved_chunks, trajectory
plus a few convenience fields (correct, f1, num_searches).

Example
-------
    python run_autorefine_inference.py \
        --questions ../data_baselines/2wiki_questions_500.json \
        --index     ../data_baselines/2wiki_index.pkl \
        --model_id  yrshi/AutoRefine-Qwen2.5-3B-Base \
        --output    results/2wiki/autorefine.jsonl \
        --topk 5
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import string
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers


# ── Prompt / templates (from AutoRefine/infer.py) ───────────────────────────────

PROMPT_TEMPLATE = (
    "You are a helpful assistant excel at answering questions with multi-turn search engine calling. "
    "To answer questions, you must first reason through the available information using <think> and </think>. "
    "If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. "
    "The retrieval system will provide you with the three most relevant documents enclosed in <documents> and </documents>. "
    "After each search, you need to summarize and refine the existing documents in <refine> and </refine>. "
    "You may send multiple search requests if needed. "
    "Once you have sufficient information, provide a concise final answer using <answer> and </answer>. "
    "For example, <answer> Donald Trump </answer>. Question: {question}\n"
)

# `\n\n{output_text}<documents>{search_results}</documents>\n\n`
SEARCH_TEMPLATE = "\n\n{output_text}<documents>{search_results}</documents>\n\n"

# Qwen2.5 EOS ids.
CURR_EOS = [151645, 151643]

# Stop generation as soon as a </search> is emitted so we can run retrieval.
TARGET_SEQUENCES = [
    "</search>", " </search>", "</search>\n", " </search>\n",
    "</search>\n\n", " </search>\n\n",
]


# ── In-process dense retriever ──────────────────────────────────────────────────

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
                cid = item.get("id", item.get("chunk_id", i))
                out[str(cid)] = item
            else:
                out[str(i)] = {"id": i, "text": str(item)}
        return out
    return {}


@dataclass
class ChunkIndexRetriever:
    chunk_ids: List[Any]
    embeddings: np.ndarray  # [N, D] float32, L2-normalized
    chunks_by_id: Dict[str, Dict[str, Any]]
    model: Any  # SentenceTransformer
    query_prompt_name: Optional[str] = None

    def _encode(self, queries: List[str]) -> np.ndarray:
        kwargs = dict(
            batch_size=min(64, max(1, len(queries))),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # Qwen3-Embedding is instruction-tuned for queries; use the "query"
        # prompt if the model exposes one, otherwise fall back to plain text
        # (matching arag_chunk_index_server.py).
        if self.query_prompt_name:
            try:
                emb = self.model.encode(queries, prompt_name=self.query_prompt_name, **kwargs)
            except Exception:
                emb = self.model.encode(queries, **kwargs)
        else:
            emb = self.model.encode(queries, **kwargs)
        return _to_numpy_embeddings(emb)

    def search(self, query: str, topk: int) -> List[Dict[str, Any]]:
        if topk <= 0:
            return []
        q_emb = self._encode([query])[0]
        scores = (self.embeddings @ q_emb).astype(np.float32)
        n = scores.shape[0]
        k = min(topk, n)
        if k <= 0:
            return []
        idxs = np.argpartition(-scores, kth=k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]

        docs: List[Dict[str, Any]] = []
        for rank, idx in enumerate(idxs.tolist()):
            cid = str(self.chunk_ids[idx]) if idx < len(self.chunk_ids) else str(idx)
            ch = self.chunks_by_id.get(cid) or {}
            title = str(ch.get("title") or cid)
            text = str(ch.get("text") or ch.get("contents") or "")
            docs.append({
                "rank": rank + 1,
                "id": cid,
                "title": title,
                "text": text,
                "contents": f"{title}\n{text}".strip(),
                "score": float(scores[idx]),
            })
        return docs


def build_retriever(index_path: str, device: str, encoder_model: Optional[str],
                    query_prompt_name: Optional[str]) -> ChunkIndexRetriever:
    from sentence_transformers import SentenceTransformer

    with open(index_path, "rb") as f:
        index_data = pickle.load(f)
    if not isinstance(index_data, dict):
        raise ValueError(f"index pickle must be a dict; got {type(index_data)}")

    chunk_ids = index_data.get("chunk_ids")
    embeddings = index_data.get("embeddings")
    chunks_obj = index_data.get("chunks")
    model_name = encoder_model or index_data.get("model_name")

    if chunk_ids is None or embeddings is None:
        raise ValueError("index pickle must contain 'chunk_ids' and 'embeddings'")
    if not model_name:
        raise ValueError("Could not determine encoder model; pass --encoder_model.")

    emb_np = _l2_normalize(_to_numpy_embeddings(embeddings))
    chunks_by_id = _coerce_chunks_by_id(chunks_obj)

    print(f"[retriever] index: {index_path}", flush=True)
    print(f"[retriever] embeddings: {emb_np.shape} (normalized)", flush=True)
    print(f"[retriever] chunks: {len(chunks_by_id)}", flush=True)
    print(f"[retriever] encoder: {model_name} on {device}", flush=True)

    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

    return ChunkIndexRetriever(
        chunk_ids=chunk_ids,
        embeddings=emb_np,
        chunks_by_id=chunks_by_id,
        model=model,
        query_prompt_name=query_prompt_name,
    )


def passages_to_string(docs: List[Dict[str, Any]]) -> str:
    """Format retrieved docs exactly like AutoRefine training/inference."""
    out = ""
    for i, d in enumerate(docs):
        out += f"Doc {i+1}(Title: {d['title']}) {d['text']}\n"
    return out


# ── AutoRefine generation loop ───────────────────────────────────────────────────

class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [
            tokenizer.encode(t, add_special_tokens=False) for t in target_sequences
        ]
        self.target_lengths = [len(t) for t in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(t, device=input_ids.device) for t in self.target_ids]
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False


def get_last_query(text: str) -> Optional[str]:
    matches = re.findall(r"<search>(.*?)</search>", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def run_autorefine(model, tokenizer, retriever, question: str, topk: int,
                   max_new_tokens: int, max_rounds: int, do_sample: bool,
                   temperature: float, device) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
    question = question.strip()
    prompt = PROMPT_TEMPLATE.format(question=question)
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(TARGET_SEQUENCES, tokenizer)]
    )

    trajectory_parts: List[str] = []
    retrieved: List[Dict[str, Any]] = []
    rounds = 0

    while rounds <= max_rounds:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

        # Finished: model emitted an EOS token -> final answer turn.
        if outputs[0][-1].item() in CURR_EOS:
            gen = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(gen, skip_special_tokens=True)
            trajectory_parts.append(output_text)
            break

        gen = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(gen, skip_special_tokens=True)

        query_text = get_last_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if query_text:
            docs = retriever.search(query_text, topk)
            search_results = passages_to_string(docs)
            retrieved.append({
                "turn": rounds + 1,
                "query": query_text,
                "chunks": [
                    {"rank": d["rank"], "id": d["id"], "title": d["title"],
                     "text": d["text"], "score": d["score"]}
                    for d in docs
                ],
            })
        else:
            # Stopped without a search and without EOS -> nothing more to do.
            trajectory_parts.append(output_text)
            break

        search_text = SEARCH_TEMPLATE.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        trajectory_parts.append(search_text)
        rounds += 1

    trajectory = "".join(part.strip() for part in trajectory_parts)

    answer_matches = re.findall(r"<answer>(.*?)</answer>", trajectory, re.DOTALL)
    predicted = answer_matches[-1].strip() if answer_matches else None
    return trajectory, predicted, retrieved


# ── Scoring (EM / F1) ────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def em_score(prediction: Optional[str], targets) -> int:
    if prediction is None:
        return 0
    if isinstance(targets, str):
        targets = [targets]
    pred_norm = normalize(prediction)
    return int(any(normalize(str(t)) == pred_norm for t in targets))


def f1_score(prediction: Optional[str], targets) -> float:
    if prediction is None:
        return 0.0
    if isinstance(targets, str):
        targets = [targets]
    pred_tokens = normalize(prediction).split()
    best = 0.0
    for t in targets:
        gold_tokens = normalize(str(t)).split()
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        common = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in set(pred_tokens))
        if common == 0:
            continue
        p = common / len(pred_tokens)
        r = common / len(gold_tokens)
        best = max(best, (2 * p * r) / (p + r))
    return best


# ── Data loading ─────────────────────────────────────────────────────────────────

def load_questions(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"--questions must be a JSON list, got {type(data)}")

    samples: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        question = item.get("question") or item.get("query")
        if not isinstance(question, str) or not question.strip():
            continue
        gt = None
        for key in ("answer", "answers", "ground_truth", "gold", "target"):
            if item.get(key) is not None:
                gt = item[key]
                break
        samples.append({
            "id": item.get("id", item.get("qid", idx)),
            "question": question.strip(),
            "ground_truth": gt,
        })
        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_completed_ids(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("id") is not None:
                done.add(str(obj["id"]))
    return done


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AutoRefine inference with in-process retrieval.")
    parser.add_argument("--questions", required=True, help="Path to questions JSON list.")
    parser.add_argument("--index", required=True, help="Path to chunk index .pkl.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--model_id", default="yrshi/AutoRefine-Qwen2.5-3B-Base",
                        help="HuggingFace model id for the AutoRefine checkpoint.")
    parser.add_argument("--topk", type=int, default=5, help="Number of chunks per search.")
    parser.add_argument("--encoder_model", default=None,
                        help="Override the embedding model (defaults to index's model_name).")
    parser.add_argument("--query_prompt_name", default=None,
                        help="SentenceTransformer prompt name for queries (e.g. 'query' for Qwen3-Embedding).")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_rounds", type=int, default=10,
                        help="Max number of search rounds before forcing a stop.")
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--greedy", dest="do_sample", action="store_false",
                        help="Use greedy decoding instead of sampling.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--retriever_device", default=None,
                        help="Device for the embedding model (defaults to --device).")
    parser.add_argument("--cache_dir", default=None, help="HF cache dir for the LLM.")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true",
                        help="Skip questions whose id already appears in --output and append.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    samples = load_questions(args.questions, args.limit)
    print(f"Loaded {len(samples)} questions from {args.questions}", flush=True)

    file_mode = "w"
    if args.resume:
        done = load_completed_ids(args.output)
        n_before = len(samples)
        samples = [s for s in samples if str(s["id"]) not in done]
        print(f"Resume: skipping {n_before - len(samples)} completed; {len(samples)} remaining.",
              flush=True)
        file_mode = "a"

    if not samples:
        print("Nothing to do.", flush=True)
        return

    retriever_device = args.retriever_device or args.device
    retriever = build_retriever(args.index, retriever_device, args.encoder_model,
                                args.query_prompt_name)

    print(f"\nLoading AutoRefine model: {args.model_id}", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=args.trust_remote_code, cache_dir=args.cache_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device.startswith("cuda") else None,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )
    model.eval()
    print("Model ready.\n", flush=True)

    n_correct = 0
    f1_sum = 0.0
    n_scored = 0
    t0 = time.time()

    with open(args.output, file_mode, encoding="utf-8") as out_f:
        for i, sample in enumerate(samples):
            qid = sample["id"]
            question = sample["question"]
            gt = sample["ground_truth"]

            print(f"[{i+1}/{len(samples)}] (id={qid}) {question[:80]}", flush=True)

            try:
                trajectory, predicted, retrieved = run_autorefine(
                    model, tokenizer, retriever, question,
                    topk=args.topk,
                    max_new_tokens=args.max_new_tokens,
                    max_rounds=args.max_rounds,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    device=model.device if hasattr(model, "device") else args.device,
                )
            except Exception as e:
                print(f"  ERROR on id={qid}: {e}", flush=True)
                trajectory, predicted, retrieved = f"(error: {e})", None, []

            has_gt = gt is not None and (not isinstance(gt, str) or bool(gt.strip()))
            correct = em_score(predicted, gt) if has_gt else None
            f1 = f1_score(predicted, gt) if has_gt else None

            result = {
                "id": qid,
                "question": question,
                "ground_truth": gt,
                "predicted": predicted,
                "retrieved_chunks": retrieved,
                "trajectory": trajectory,
                "num_searches": len(retrieved),
                "correct": correct,
                "f1": f1,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if has_gt:
                n_scored += 1
                n_correct += correct
                f1_sum += f1
                print(f"  GT: {gt}  |  Pred: {predicted}  |  "
                      f"{'CORRECT' if correct else 'WRONG'}  F1: {f1:.2f}  "
                      f"searches: {len(retrieved)}", flush=True)
            else:
                print(f"  Pred: {predicted}  |  not scored  searches: {len(retrieved)}", flush=True)

    dt = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"  Model    : {args.model_id}", flush=True)
    print(f"  Samples  : {len(samples)}  ({dt/60:.1f} min)", flush=True)
    if n_scored:
        print(f"  EM       : {n_correct}/{n_scored} = {n_correct / n_scored:.3f}", flush=True)
        print(f"  F1       : {f1_sum / n_scored:.3f}", flush=True)
    else:
        print("  EM/F1    : N/A (no ground-truth answers found)", flush=True)
    print(f"  Output   : {args.output}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
