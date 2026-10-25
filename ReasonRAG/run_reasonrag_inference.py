#!/usr/bin/env python3
"""
Standalone ReasonRAG inference over the `data_baselines` 500-question test sets.

This reproduces the ReasonRAG agentic RAG loop (see
`pipeline/reasonrag_pipeline.py::ReasonRAGPipeline.run_batch`) *without* the
FlashRAG runtime. By default retrieval runs over the *entire* per-dataset chunk
corpus in `data_baselines/` (e.g. `2wiki_chunks.json`, ~56k chunks) with
`--topk 5`, mirroring the original open-domain-over-a-corpus setup. Gold labels
for analysis are recovered per question (by corpus id for MuSiQue, by
supporting-fact title for 2Wiki/HotpotQA). Pass `--retrieval-scope per_question`
to instead retrieve only within each question's own gold+distractor pool. The
generator model is loaded straight from HuggingFace.

ReasonRAG loop (per question):
    begin_reasoning
        -> if <answer>: done
        -> if <query>: document_analysis (retrieve top-k, extract <evidence>)
                          -> reasoning
                                -> <answer> done, or <query> -> document_analysis ...
    (falls back to answer_generation on the last iteration)

Output: a JSON list, one object per question, with
    id, question, ground_truth, predicted_answer, retrieved_chunks, trajectory

Examples
--------
# MuSiQue, global top-5 retrieval over the whole Musique chunk corpus
python run_reasonrag_inference.py \
    --questions ../data_baselines/musique_questions_500.json \
    --chunks ../data_baselines/Musique_chunks.json \
    --chunk-ids ../data_baselines/musique_questions_500_chunk_ids_from_Musique_chunks.json \
    --model reasonrag/Qwen2.5-7B-Instruct-ReasonRAG \
    --topk 5 \
    --output output/reasonrag_musique_500.json

# 2Wiki, global top-5 retrieval over the whole 2wiki chunk corpus
python run_reasonrag_inference.py \
    --questions ../data_baselines/2wiki_questions_500.json \
    --chunks ../data_baselines/2wiki_chunks.json \
    --model reasonrag/Qwen2.5-7B-Instruct-ReasonRAG \
    --topk 5 \
    --output output/reasonrag_2wiki_500.json

# HotpotQA, global top-5 retrieval over the whole hqa chunk corpus
python run_reasonrag_inference.py \
    --questions ../data_baselines/hqa_questions_500.json \
    --chunks ../data_baselines/hqa_chunks.json \
    --model reasonrag/Qwen2.5-7B-Instruct-ReasonRAG \
    --topk 5 \
    --output output/reasonrag_hqa_500.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # optional dependency; degrade gracefully if unavailable.
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


# --------------------------------------------------------------------------- #
# Prompts (copied verbatim from ReasonRAGPipeline so behaviour matches).
# --------------------------------------------------------------------------- #
BEGIN_REASONING_PROMPT = """You are an assistant for question answering with access to a retrieval tool. Upon receiving a question, your task is to:
* Analyze and Decompose the Question: Break the question into smaller, manageable sub-questions to ensure all aspects are addressed.
* Evaluate Your Knowledge: Assess each sub-question or component:
- Identify parts you can confidently answer based on your existing knowledge.
- Pinpoint parts that require additional information or verification through retrieval tools.
* Conciseness: Ensure both queries and answers are concise, using nouns or short phrases whenever possible.
* Respond Format:
If your knowledge is sufficient to answer the question, conclude with:
"So the answer is <answer>{answer_format}</answer>"
If retrieval is necessary to provide a complete answer, conclude with:
"So the next query is <query>query</query>"
"""

DOCUMENT_ANALYSIS_PROMPT = """You are an information retrieval assistant. Your task is to extract relevant evidence from the provided Wikipedia documents based on the latest query.

    Instructions:

    * Identify key terms or concepts in the query.
    * Search the documents for evidence that supports the query.
    * Response format:
    If relevant evidence is found, output:
       Based on the query, the relevant evidence is <evidence>evidence</evidence>.
    If no relevant evidence is found, output:
       <evidence>None</evidence>.
"""

REASONING_PROMPT = """You are a question-answering assistant with access to a retrieval tool. Your goal is to provide a concise and accurate reasoning process.
Instructions:
* Error Reflection: If errors exist in previous thoughts, identify and correct them. Skip this step if no errors are present.
* Information Sufficiency: Evaluate whether the current information is sufficient to fully and accurately answer the question. If additional retrieval is needed, deconstruct the question and generate the next query. Avoid repeating previous queries. If no meaningful new query can be generated, explain why and provide an answer based on the current information.
* Conciseness: Ensure both queries and answers are concise, using nouns or short phrases whenever possible.
* Conclusion:
If generating an answer:
"So the answer is <answer>{answer_format}</answer>".
If more retrieval is needed:
"So the next query is <query>query</query>".
    """

ANSWER_GENERATION_PROMPT = """You are a reasoning assistant with retrieval. Give a precise and very concise final answer for the given question, conclude with 'So the answer is <answer>{answer_format}</answer>'. Keep your final answer brief and to the point, followed without any explanation.
"""

REFERENCE_TEMPLATE = "Wikipedia Title: {title}\n{text}\n\n"

STOP_TOKENS = ["<|im_end|>", "<|endoftext|>", "</answer>", "</query>", "</evidence>"]


# --------------------------------------------------------------------------- #
# Parsing helpers (mirror ReasonRAGPipeline).
# --------------------------------------------------------------------------- #
def get_flag(response: str) -> str:
    if "So the next query is" in response:
        return "query"
    if "So the answer is" in response:
        return "answer"
    if "<evidence>" in response:
        return "evidence"
    return "None"


def _strip_special_tags(text: str) -> str:
    return re.sub(r"</?(answer|query|evidence)>", "", text).strip()


def extract_query(response: str) -> str:
    match = re.search(r"So the next query is\s*(.*?)(?=\n|$)", response, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _strip_special_tags(match.group(1).strip())


def extract_answer(response: str) -> str:
    match = re.search(r"So the answer is\s*(.*?)(?=\n|$)", response, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _strip_special_tags(match.group(1).strip())


# --------------------------------------------------------------------------- #
# Dataset loading. Produces a uniform per-question record with a candidate
# chunk pool, working for MuSiQue / 2Wiki / HotpotQA layouts.
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_corpus(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Chunk corpus keyed by string id -> {id, title, text}."""
    if path is None:
        return {}
    corpus: Dict[str, Dict[str, Any]] = {}
    for row in _load_json(path):
        cid = str(row.get("id"))
        corpus[cid] = {
            "id": row.get("id"),
            "title": row.get("title", ""),
            "text": row.get("text", row.get("contents", "")),
        }
    return corpus


def golden_answers(item: Dict[str, Any]) -> List[str]:
    answers: List[str] = []
    for key in ("answer", "gold_answer", "golden_answers"):
        val = item.get(key)
        if isinstance(val, list):
            answers.extend(str(v) for v in val)
        elif val not in (None, ""):
            answers.append(str(val))
    for alias in item.get("answer_aliases", []) or []:
        answers.append(str(alias))
    seen, unique = set(), []
    for a in answers:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique or [""]


def pool_from_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a per-question candidate chunk pool from the question's own context.

    Handles MuSiQue (`paragraphs`) and 2Wiki/HotpotQA (`context`) layouts.
    """
    pool: List[Dict[str, Any]] = []

    # MuSiQue-style paragraphs.
    if isinstance(item.get("paragraphs"), list):
        for para in item["paragraphs"]:
            pool.append(
                {
                    "id": para.get("idx"),
                    "title": para.get("title", ""),
                    "text": para.get("paragraph_text", para.get("text", "")),
                    "is_gold": bool(para.get("is_supporting", False)),
                }
            )
        return pool

    # 2Wiki / HotpotQA-style context.
    context = item.get("context")
    gold_titles = set()
    sf = item.get("supporting_facts") or {}
    if isinstance(sf, dict):
        gold_titles = set(sf.get("title", []) or [])
    if isinstance(context, dict) and "title" in context and "sentences" in context:
        for i, (title, sents) in enumerate(zip(context["title"], context["sentences"])):
            text = "".join(sents) if isinstance(sents, list) else str(sents)
            pool.append(
                {"id": i, "title": title, "text": text, "is_gold": title in gold_titles}
            )
    elif isinstance(context, list):
        # HotpotQA raw layout: [[title, [sentences...]], ...]
        for i, entry in enumerate(context):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                title, sents = entry
                text = "".join(sents) if isinstance(sents, list) else str(sents)
                pool.append(
                    {"id": i, "title": title, "text": text, "is_gold": title in gold_titles}
                )
    return pool


def build_records(
    questions_path: Path,
    corpus: Dict[str, Dict[str, Any]],
    chunk_ids_path: Optional[Path],
) -> List[Dict[str, Any]]:
    """Return records: {id, question, golden_answers, pool, gold_ids, gold_titles}.

    ``gold_ids``/``gold_titles`` let us recover the per-question gold labels even
    when retrieval runs over a shared (global) corpus, where the pool itself is
    not question-specific. Gold is identified by corpus id (MuSiQue, via the
    chunk-id index) and/or by supporting-fact title (2Wiki / HotpotQA).
    """
    items = _load_json(questions_path)

    chunk_id_map: Dict[str, Dict[str, List[Any]]] = {}
    if chunk_ids_path is not None:
        for row in _load_json(chunk_ids_path):
            chunk_id_map[str(row.get("id"))] = {
                "gold": row.get("gold_chunk_ids", []) or [],
                "distractor": row.get("distractor_chunk_ids", []) or [],
            }

    records: List[Dict[str, Any]] = []
    for item in items:
        qid = str(item.get("id", item.get("qid")))
        question = item.get("question", "")

        # Per-question gold identifiers (used to tag retrieved chunks as gold).
        gold_ids: set = set()
        gold_titles: set = set()
        sf = item.get("supporting_facts") or {}
        if isinstance(sf, dict):
            gold_titles = set(sf.get("title", []) or [])
        if qid in chunk_id_map:
            gold_ids = {str(i) for i in chunk_id_map[qid]["gold"]}

        pool: List[Dict[str, Any]] = []
        # Prefer the provided chunk-id index mapped against the chunk corpus.
        if qid in chunk_id_map and corpus:
            ids = chunk_id_map[qid]
            for cid in list(ids["gold"]) + list(ids["distractor"]):
                chunk = corpus.get(str(cid))
                if chunk is not None:
                    pool.append({**chunk, "is_gold": str(cid) in gold_ids})
        # Otherwise build the pool from the question's own context.
        if not pool:
            pool = pool_from_item(item)

        records.append(
            {
                "id": qid,
                "question": question,
                "golden_answers": golden_answers(item),
                "pool": pool,
                "gold_ids": gold_ids,
                "gold_titles": gold_titles,
            }
        )
    return records


# --------------------------------------------------------------------------- #
# Dense retriever over per-question (or global) chunk pools using BGE.
# --------------------------------------------------------------------------- #
class DenseRetriever:
    def __init__(self, model_name: str, device: Optional[str], query_instruction: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.query_instruction = query_instruction

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=128,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32, copy=False)

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        prefixed = [f"{self.query_instruction}{q}" for q in queries]
        return self.encode_passages(prefixed)


def chunk_text_for_embedding(chunk: Dict[str, Any]) -> str:
    title = chunk.get("title", "") or ""
    text = chunk.get("text", "") or ""
    return f"{title} {text}".strip()


def precompute_pool_embeddings(
    records: List[Dict[str, Any]], retriever: DenseRetriever, global_pool: Optional[List[Dict[str, Any]]]
) -> None:
    """Attach `pool_emb` (np.ndarray) to each record's pool."""
    if global_pool is not None:
        emb = retriever.encode_passages([chunk_text_for_embedding(c) for c in global_pool])
        for rec in records:
            rec["pool"] = global_pool
            rec["pool_emb"] = emb
        return

    # Per-question: flatten, encode once, split back.
    flat_texts: List[str] = []
    spans: List[Tuple[int, int]] = []
    for rec in records:
        start = len(flat_texts)
        flat_texts.extend(chunk_text_for_embedding(c) for c in rec["pool"])
        spans.append((start, len(flat_texts)))
    all_emb = retriever.encode_passages(flat_texts)
    for rec, (s, e) in zip(records, spans):
        rec["pool_emb"] = all_emb[s:e]


def retrieve(rec: Dict[str, Any], query: str, retriever: DenseRetriever, topk: int) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = rec["pool"]
    pool_emb: np.ndarray = rec["pool_emb"]
    if not query or len(pool) == 0 or pool_emb.shape[0] == 0:
        return []
    q_emb = retriever.encode_queries([query])[0]
    scores = pool_emb @ q_emb
    k = min(topk, len(pool))
    idxs = np.argsort(-scores)[:k]
    gold_ids: set = rec.get("gold_ids", set())
    gold_titles: set = rec.get("gold_titles", set())
    docs = []
    for idx in idxs.tolist():
        chunk = pool[idx]
        title = chunk.get("title", "")
        # In global scope the pool is shared, so recover gold status from the
        # question's own gold ids/titles; fall back to the chunk's own flag.
        is_gold = (
            str(chunk.get("id")) in gold_ids
            or (title != "" and title in gold_titles)
            or bool(chunk.get("is_gold", False))
        )
        docs.append(
            {
                "id": chunk.get("id"),
                "title": title,
                "text": chunk.get("text", ""),
                "is_gold": is_gold,
                "score": float(scores[idx]),
            }
        )
    return docs


def format_reference(docs: List[Dict[str, Any]]) -> str:
    return "".join(
        REFERENCE_TEMPLATE.format(title=d.get("title", ""), text=d.get("text", "")) for d in docs
    )


# --------------------------------------------------------------------------- #
# Generator backends (vLLM preferred, HF transformers fallback).
# --------------------------------------------------------------------------- #
class Generator:
    def generate(self, prompts: List[str], max_tokens: int) -> List[str]:
        raise NotImplementedError


class VLLMGenerator(Generator):
    def __init__(self, model_name: str, max_model_len: int, gpu_memory_utilization: float,
                 tensor_parallel_size: int, cache_dir: Optional[str], temperature: float):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            download_dir=cache_dir,
        )
        self.temperature = temperature
        self._SamplingParams = SamplingParams

    def generate(self, prompts: List[str], max_tokens: int) -> List[str]:
        if not prompts:
            return []
        sampling = self._SamplingParams(
            temperature=self.temperature, top_p=1.0, max_tokens=max_tokens, stop=STOP_TOKENS
        )
        outputs = self.llm.generate(prompts, sampling, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def apply_chat(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


class HFGenerator(Generator):
    def __init__(self, model_name: str, device: str, cache_dir: Optional[str], temperature: float):
        import torch
        import transformers

        self.torch = torch
        self.temperature = temperature
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.model.eval()

    def apply_chat(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _truncate_at_stop(self, text: str) -> str:
        cut = len(text)
        for stop in STOP_TOKENS:
            pos = text.find(stop)
            if pos != -1:
                cut = min(cut, pos)
        return text[:cut]

    def generate(self, prompts: List[str], max_tokens: int) -> List[str]:
        results: List[str] = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            do_sample = self.temperature > 0.0
            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = self.temperature
            with self.torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)
            gen = out[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
            results.append(self._truncate_at_stop(text))
        return results


def build_generator(args) -> Generator:
    engine = args.engine
    if engine in ("auto", "vllm"):
        try:
            gen = VLLMGenerator(
                model_name=args.model,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=args.tensor_parallel_size,
                cache_dir=args.cache_dir,
                temperature=args.temperature,
            )
            print(f"[generator] using vLLM backend for {args.model}")
            return gen
        except Exception as exc:  # noqa: BLE001
            if engine == "vllm":
                raise
            print(f"[generator] vLLM unavailable ({exc}); falling back to transformers.")
    gen = HFGenerator(model_name=args.model, device=args.device, cache_dir=args.cache_dir,
                      temperature=args.temperature)
    print(f"[generator] using transformers backend for {args.model}")
    return gen


# --------------------------------------------------------------------------- #
# The ReasonRAG agentic loop, batched over all active questions.
# --------------------------------------------------------------------------- #
def run_inference(
    records: List[Dict[str, Any]],
    generator: Generator,
    retriever: DenseRetriever,
    answer_format: str,
    max_iter: int,
    topk: int,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    begin_system = BEGIN_REASONING_PROMPT.format(answer_format=answer_format)
    reasoning_system = REASONING_PROMPT.format(answer_format=answer_format)
    answer_system = ANSWER_GENERATION_PROMPT.format(answer_format=answer_format)
    doc_system = DOCUMENT_ANALYSIS_PROMPT

    # Per-question mutable state.
    states = []
    for rec in records:
        states.append(
            {
                "rec": rec,
                "thoughts": [],
                "iteration_count": 0,
                "flag": None,
                "query": None,
                "answer": None,
                "finish": False,
                "trajectory": [],
                "retrieved_chunks": [],
            }
        )

    def question_with_thoughts(st) -> str:
        return st["rec"]["question"] + "\nPrevious Thoughts: " + " ".join(st["thoughts"])

    total = len(states)
    # Question-wise progress bar: advances as each question reaches `finish`.
    if tqdm is not None:
        progress = tqdm(total=total, desc="questions", unit="q")
    else:
        progress = None
    finished_count = 0

    def _report(step_label: str) -> None:
        """Advance the progress bar by the number of newly-finished questions."""
        nonlocal finished_count
        done = sum(1 for st in states if st["finish"])
        delta = done - finished_count
        finished_count = done
        active_n = total - done
        if progress is not None:
            if delta > 0:
                progress.update(delta)
            progress.set_postfix(step=step_label, active=active_n, done=done)
        else:
            print(f"[inference] {step_label}: {done}/{total} finished, {active_n} active",
                  flush=True)

    # ---- Step 0: begin_reasoning for every question ----
    prompts = [
        generator.apply_chat(begin_system, f"Question: {st['rec']['question']}") for st in states
    ]
    responses = generator.generate(prompts, max_tokens)
    for st, resp in zip(states, responses):
        st["thoughts"].append(resp)
        st["flag"] = get_flag(resp)
        st["query"] = extract_query(resp)
        st["answer"] = extract_answer(resp)
        st["trajectory"].append(
            {
                "step": 0,
                "action": "begin_reasoning",
                "response": resp,
                "query": st["query"],
                "answer": st["answer"],
                "retrieved_chunk_ids": [],
            }
        )
        if st["flag"] in ("finish", "answer"):
            st["finish"] = True
    _report("begin_reasoning")

    # ---- Iterative refinement ----
    for step in range(max_iter):
        active = [st for st in states if not st["finish"]]
        if not active:
            break

        # Retrieve for questions whose last action asked a query.
        step_docs: Dict[int, List[Dict[str, Any]]] = {}
        for i, st in enumerate(active):
            if st["iteration_count"] < max_iter - 1 and "query" in (st["flag"] or ""):
                docs = retrieve(st["rec"], st["query"] or "", retriever, topk)
                step_docs[i] = docs
                for d in docs:
                    st["retrieved_chunks"].append({**d, "step": step + 1, "query": st["query"]})

        # Build the next prompt for each active question.
        prompts = []
        actions = []
        for i, st in enumerate(active):
            qt = question_with_thoughts(st)
            if st["iteration_count"] >= max_iter - 1:
                prompts.append(generator.apply_chat(answer_system, f"Question: {qt}"))
                actions.append("answer_generation")
            elif "query" in (st["flag"] or ""):
                reference = format_reference(step_docs.get(i, []))
                user = f"Question: {qt}. Reference: <reference>{reference}</reference>"
                prompts.append(generator.apply_chat(doc_system, user))
                actions.append("document_analysis")
            elif "evidence" in (st["flag"] or ""):
                prompts.append(generator.apply_chat(reasoning_system, f"Question: {qt}"))
                actions.append("reasoning")
            else:
                prompts.append(generator.apply_chat(answer_system, f"Question: {qt}"))
                actions.append("answer_generation")

        responses = generator.generate(prompts, max_tokens)

        for i, (st, resp, action) in enumerate(zip(active, responses, actions)):
            st["thoughts"].append(resp)
            st["iteration_count"] += 1
            st["flag"] = get_flag(resp)
            st["query"] = extract_query(resp)
            st["answer"] = extract_answer(resp)
            st["trajectory"].append(
                {
                    "step": st["iteration_count"],
                    "action": action,
                    "response": resp,
                    "query": st["query"],
                    "answer": st["answer"],
                    "retrieved_chunk_ids": [d.get("id") for d in step_docs.get(i, [])],
                }
            )
            if st["flag"] in ("finish", "answer") or st["iteration_count"] >= max_iter:
                st["finish"] = True
        _report(f"step {step + 1}/{max_iter}")

    # Any question still unfinished after the loop is treated as done now.
    for st in states:
        st["finish"] = True
    _report("done")
    if progress is not None:
        progress.close()

    # ---- Collect outputs ----
    outputs = []
    for st in states:
        rec = st["rec"]
        pred = st["answer"] or ""
        if not pred:
            # Fallback: scan thoughts for a final answer.
            for resp in reversed(st["thoughts"]):
                cand = extract_answer(resp)
                if cand:
                    pred = cand
                    break
        golden = rec["golden_answers"]
        outputs.append(
            {
                "id": rec["id"],
                "question": rec["question"],
                "ground_truth": golden[0] if len(golden) == 1 else golden,
                "predicted_answer": pred,
                "retrieved_chunks": st["retrieved_chunks"],
                "trajectory": st["trajectory"],
            }
        )
    return outputs


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_data = (here.parent / "data_baselines")

    p = argparse.ArgumentParser(description="ReasonRAG inference over data_baselines test sets.")
    p.add_argument("--questions", type=Path,
                   default=default_data / "musique_questions_500.json",
                   help="Questions JSON (musique / 2wiki / hqa).")
    p.add_argument("--chunks", type=Path,
                   default=default_data / "Musique_chunks.json",
                   help="Chunk corpus JSON [{id,title,text}]. Optional.")
    p.add_argument("--chunk-ids", type=Path, dest="chunk_ids",
                   default=default_data / "musique_questions_500_chunk_ids_from_Musique_chunks.json",
                   help="Per-question gold/distractor chunk-id index. Optional.")
    p.add_argument("--model", type=str, default="reasonrag/Qwen2.5-7B-Instruct-ReasonRAG",
                   help="HuggingFace id/path of the ReasonRAG generator model.")
    p.add_argument("--retriever-model", type=str, default="BAAI/bge-base-en-v1.5",
                   help="HuggingFace id/path of the dense retriever (sentence-transformers).")
    p.add_argument("--query-instruction", type=str,
                   default="Represent this sentence for searching relevant passages: ",
                   help="Prefix added to queries before encoding (BGE convention).")
    p.add_argument("--retrieval-scope", choices=["per_question", "global"], default="global",
                   help="Retrieve across the whole chunk corpus (global, default) or within "
                        "each question's own gold+distractor pool (per_question).")
    p.add_argument("--topk", type=int, default=5, help="Chunks retrieved per query.")
    p.add_argument("--max-iter", type=int, default=8, help="Max ReasonRAG refinement iterations.")
    p.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per generation call.")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="Sampling temperature for the generator (0 = greedy).")
    p.add_argument("--answer-format", type=str, default="answer")
    p.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    p.add_argument("--output", type=Path, default=Path("output/reasonrag_inference.json"))

    # Generator engine options.
    p.add_argument("--engine", choices=["auto", "vllm", "hf"], default="auto")
    p.add_argument("--device", type=str, default="cuda:0", help="Device for the hf backend / retriever.")
    p.add_argument("--cache-dir", type=str, default=None, help="HuggingFace cache/download dir.")
    p.add_argument("--max-model-len", type=int, default=8192, help="vLLM max_model_len.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM gpu mem fraction.")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    chunks_path = args.chunks if args.chunks and Path(args.chunks).is_file() else None
    chunk_ids_path = args.chunk_ids if args.chunk_ids and Path(args.chunk_ids).is_file() else None

    print(f"[data] questions: {args.questions}")
    print(f"[data] chunks:    {chunks_path}")
    print(f"[data] chunk-ids: {chunk_ids_path}")

    corpus = load_corpus(chunks_path)
    records = build_records(args.questions, corpus, chunk_ids_path)
    if args.limit is not None:
        records = records[: args.limit]
    print(f"[data] loaded {len(records)} questions; "
          f"avg pool size = {np.mean([len(r['pool']) for r in records]):.1f}")

    retriever = DenseRetriever(
        model_name=args.retriever_model,
        device=args.device,
        query_instruction=args.query_instruction,
    )

    global_pool = None
    if args.retrieval_scope == "global":
        if not corpus:
            raise ValueError(
                "--retrieval-scope global requires a --chunks corpus. Pass the "
                "dataset's full chunk file, e.g. --chunks ../data_baselines/2wiki_chunks.json "
                "(or hqa_chunks.json / Musique_chunks.json)."
            )
        global_pool = [
            {"id": c["id"], "title": c["title"], "text": c["text"], "is_gold": False}
            for c in corpus.values()
        ]
        print(f"[retriever] global scope over {len(global_pool)} chunks")

    print("[retriever] encoding chunk pools ...")
    t0 = time.time()
    precompute_pool_embeddings(records, retriever, global_pool)
    print(f"[retriever] done in {time.time() - t0:.1f}s")

    generator = build_generator(args)

    print("[inference] running ReasonRAG loop ...")
    t0 = time.time()
    outputs = run_inference(
        records=records,
        generator=generator,
        retriever=retriever,
        answer_format=args.answer_format,
        max_iter=args.max_iter,
        topk=args.topk,
        max_tokens=args.max_tokens,
    )
    print(f"[inference] finished {len(outputs)} questions in {time.time() - t0:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"[output] wrote {len(outputs)} results to {args.output}")


if __name__ == "__main__":
    main()
