#!/usr/bin/env python3
"""
Example Usage:

Start retriever server using the following command:
    python index_server.py \
    --chunk_index_pkl data_baselines/2wiki_index.pkl \
    --port 8000

Then run the inference script with the following command:
    python inference_searchr1.py \
    --output_file results_7b/2wiki/grpo.jsonl \
    --retriever_url http://127.0.0.1:8000 \
    --model_id PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.2 \
    --topk 5 \
    --questions data_baselines/2wiki_questions_500.json \
    --resume
"""

import argparse
import json
import re
import string
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT   = "/project/pi_sniekum_umass_edu/vgandhi/verl-tool-RAG"
TEST_DATA = f"{PROJECT}/data/OurAgentV3/training_data/test.parquet"

MODEL_ID  = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo"

# ── Eval config ───────────────────────────────────────────────────────────────
N_SAMPLES   = 1000
MAX_TURNS   = 10
MAX_TOKENS  = 512
TEMPERATURE = 0.6
TOPK        = 5

# ── Prompt (from examples/data_preprocess/search_r1.py) ───────────────────────
SYSTEM_CONTENT = "You are a helpful and harmless assistant."

PROMPT_PREFIX = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: "
)

STOP_STRINGS = ["</search>", "</answer>"]


# ── Retrieval ──────────────────────────────────────────────────────────────────

def _retriever_endpoints(base_url: str):
    # Accept either a base URL (http://host:port) or a full endpoint URL.
    if base_url.endswith("/retrieve") or base_url.endswith("/search"):
        return [base_url]
    return [f"{base_url}/retrieve", f"{base_url}/search"]


def _format_retrieve_response(result_items):
    """Returns (formatted_text, chunk_ids_list)."""
    formatted = []
    chunk_ids = []
    for i, item in enumerate(result_items):
        doc = item.get("document", item) if isinstance(item, dict) else {}
        contents = doc.get("contents", "") if isinstance(doc, dict) else ""
        title = contents.split("\n")[0] if contents else str(doc.get("title", ""))
        text = "\n".join(contents.split("\n")[1:]) if contents else str(doc.get("text", ""))
        formatted.append(f"Doc {i+1}(Title: {title}) {text}".strip())
        cid = doc.get("id") if isinstance(doc, dict) else None
        if cid is not None:
            chunk_ids.append(str(cid))
    return "\n".join(formatted), chunk_ids


def search(query: str, base_url: str, topk: int):
    """Returns (formatted_text, chunk_ids_list)."""
    try:
        errors = []
        for endpoint in _retriever_endpoints(base_url):
            try:
                if endpoint.endswith("/retrieve"):
                    r = requests.post(
                        endpoint,
                        json={"queries": [query], "topk": topk, "return_scores": True},
                        timeout=30,
                    )
                    r.raise_for_status()
                    payload = r.json()
                    result_items = payload.get("result", [[]])[0]
                    return _format_retrieve_response(result_items)
                r = requests.post(endpoint, json={"query": query, "topk": topk}, timeout=30)
                r.raise_for_status()
                return r.json()["result"], []
            except Exception as inner_e:
                errors.append(f"{endpoint}: {inner_e}")
        return f"(search error: {' | '.join(errors)})", []
    except Exception as e:
        return f"(search error: {e})", []


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent_loop(model, tokenizer, question: str, retriever_url: str, max_turns: int, topk: int):
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user",   "content": PROMPT_PREFIX + question},
    ]
    prompt  = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    context    = prompt
    answer     = None
    tool_calls = []
    all_chunk_ids: List[str] = []

    for _turn in range(max_turns + 1):
        inputs    = tokenizer(context, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=STOP_STRINGS,
                tokenizer=tokenizer,
            )

        gen_ids   = out[0][input_len:]
        generated = tokenizer.decode(gen_ids, skip_special_tokens=False)
        context  += generated

        if "</answer>" in generated:
            m = re.search(r"<answer>(.*?)</answer>", generated, re.DOTALL)
            if m:
                answer = m.group(1).strip()
            break

        if "</search>" in generated:
            m = re.search(r"<search>(.*?)</search>", generated, re.DOTALL)
            if m:
                query = m.group(1).strip()
                result, chunk_ids = search(query, retriever_url, topk)
                tool_calls.append({"tool": "search", "query": query, "chunk_ids": chunk_ids})
                all_chunk_ids.extend(chunk_ids)
                context += f"\n\n<information>\n{result.strip()}\n</information>\n\n"
        else:
            break

    trajectory = context[len(prompt):]
    return trajectory, answer, tool_calls, all_chunk_ids


# ── EM + F1 scoring ───────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def em_score(prediction: str, targets) -> int:
    if prediction is None:
        return 0
    if isinstance(targets, str):
        targets = [targets]
    pred_norm = normalize(prediction)
    return int(any(normalize(t) == pred_norm for t in targets))


def f1_score(prediction: str, targets) -> float:
    if prediction is None:
        return 0.0
    if isinstance(targets, str):
        targets = [targets]
    pred_tokens = normalize(prediction).split()
    best = 0.0
    for t in targets:
        gold_tokens = normalize(t).split()
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


def _extract_question_from_item(item: Dict[str, Any]) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    q = item.get("question")
    if isinstance(q, str) and q.strip():
        return q.strip()
    q = item.get("query")
    if isinstance(q, str) and q.strip():
        return q.strip()
    extra = item.get("extra_info")
    if isinstance(extra, dict):
        q = extra.get("question")
        if isinstance(q, str) and q.strip():
            return q.strip()
    return None


def _extract_ground_truth_from_item(item: Dict[str, Any]):
    # Common answer field variants across QA datasets.
    for key in ("answer", "answers", "ground_truth", "gold", "target"):
        if key in item and item[key] is not None:
            return item[key]
    reward_model = item.get("reward_model")
    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth")
        if isinstance(gt, dict) and gt.get("target") is not None:
            return gt.get("target")
    return None


def load_samples_from_questions_json(path: str, n_samples: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"--questions must point to a JSON list, got {type(data)}")

    samples: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        question = _extract_question_from_item(item)
        if not question:
            continue
        samples.append(
            {
                "question": question,
                "ground_truth": _extract_ground_truth_from_item(item),
                "qid": item.get("qid", item.get("id", idx)),
            }
        )
        if len(samples) >= n_samples:
            break
    return samples


def load_samples_from_parquet(path: str, n_samples: int) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    rows = df.head(n_samples)
    samples: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(rows.iterrows()):
        question = row["extra_info"]["question"]
        gt = row["reward_model"]["ground_truth"]["target"]
        samples.append({"question": question, "ground_truth": gt, "qid": idx})
    return samples


# ── Resume helpers ────────────────────────────────────────────────────────────

def load_completed_qids(path: str) -> set:
    """Read qids already present in a JSONL output file."""
    import os
    if not os.path.isfile(path):
        return set()
    done: set = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("qid")
            if qid is not None:
                done.add(str(qid))
    return done


def load_completed_results(path: str) -> List[Dict[str, Any]]:
    """Load all result dicts already in a JSONL output file."""
    import os
    if not os.path.isfile(path):
        return []
    results: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file",   required=True,
                        help="Output JSONL file (one JSON object per question)")
    parser.add_argument("--retriever_url", required=True,
                        help="Base URL for chunk retrieval server")
    parser.add_argument("--n_samples",     type=int, default=N_SAMPLES)
    parser.add_argument("--max_turns",     type=int, default=MAX_TURNS)
    parser.add_argument("--topk",          type=int, default=TOPK)
    parser.add_argument("--questions",     default=None,
                        help="Optional path to JSON list of question objects.")
    parser.add_argument("--test_data",     default=TEST_DATA)
    parser.add_argument("--model_id",      default=MODEL_ID)
    parser.add_argument("--resume",        action="store_true",
                        help="Skip questions whose qid already appears in --output_file "
                             "and append new results (same --questions / --test_data as original run).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # ── Best-effort sanity-check retrieval server ────────────────────────────
    try:
        ok = False
        if not (args.retriever_url.endswith("/retrieve") or args.retriever_url.endswith("/search")):
            try:
                requests.get(f"{args.retriever_url}/health", timeout=5).raise_for_status()
                ok = True
            except Exception:
                ok = False
        if not ok:
            probe, _ = search("test", args.retriever_url, args.topk)
            ok = "(search error:" not in probe
        if ok:
            print(f"Retrieval server OK: {args.retriever_url}", flush=True)
        else:
            print(f"WARNING: retrieval server may not be reachable: {args.retriever_url}", flush=True)
    except Exception as e:
        print(f"WARNING: retrieval server check failed: {e}", flush=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.questions:
        samples = load_samples_from_questions_json(args.questions, args.n_samples)
        print(
            f"Loaded {len(samples)} samples from questions JSON: {args.questions}",
            flush=True,
        )
    else:
        samples = load_samples_from_parquet(args.test_data, args.n_samples)
        print(f"Loaded {len(samples)} test samples from {args.test_data}", flush=True)

    # ── Resume: filter out already-completed samples ─────────────────────────
    file_mode = "w"
    prior_results: List[Dict[str, Any]] = []
    if args.resume:
        completed_qids = load_completed_qids(args.output_file)
        n_before = len(samples)
        samples = [s for s in samples if str(s.get("qid", "")) not in completed_qids]
        skipped = n_before - len(samples)
        if skipped:
            print(f"Resume: skipping {skipped} questions already in {args.output_file}, "
                  f"{len(samples)} remaining.", flush=True)
        prior_results = load_completed_results(args.output_file)
        file_mode = "a"

    if not samples:
        print("No questions to run (empty list or all already completed).", flush=True)
        if prior_results:
            _print_summary(prior_results, args.model_id)
        return

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print("Model ready.\n", flush=True)

    # ── Run inference (write each result immediately) ─────────────────────────
    all_results = list(prior_results)
    with open(args.output_file, file_mode, encoding="utf-8") as out_f:
        for i, sample in enumerate(samples):
            question = sample["question"]
            gt = sample.get("ground_truth")

            print(f"[{i+1}/{len(samples)}] {question[:80]}", flush=True)

            trajectory, predicted, tool_calls, chunk_ids = run_agent_loop(
                model, tokenizer, question, args.retriever_url, args.max_turns, args.topk
            )
            has_gt = gt is not None and (not isinstance(gt, str) or bool(gt.strip()))
            correct = em_score(predicted, gt) if has_gt else None
            f1 = f1_score(predicted, gt) if has_gt else None

            result = dict(
                qid=sample.get("qid"),
                question=question,
                ground_truth=gt,
                predicted=predicted,
                correct=correct,
                f1=f1,
                tool_calls=tool_calls,
                chunk_ids=chunk_ids,
                trajectory=trajectory,
            )
            all_results.append(result)

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            status = "CORRECT" if correct else "WRONG"
            tools_summary = ", ".join(
                f"{c['tool']}({str(list(c.values())[1])[:30]})" for c in tool_calls
            )
            if has_gt:
                print(f"  GT: {gt}  |  Pred: {predicted}  |  {status}  F1: {f1:.2f}", flush=True)
            else:
                print(f"  GT: N/A  |  Pred: {predicted}  |  Not scored", flush=True)
            print(f"  Tools: [{tools_summary}]", flush=True)
            print(f"  Chunks: {chunk_ids}\n", flush=True)

    _print_summary(all_results, args.model_id)
    print(f"Results written to: {args.output_file}", flush=True)


def _print_summary(results: List[Dict[str, Any]], model_id: str):
    scored = [r for r in results if r.get("correct") is not None and r.get("f1") is not None]
    n_correct = sum(r["correct"] for r in scored) if scored else 0
    avg_f1 = (sum(r["f1"] for r in scored) / len(scored)) if scored else None
    em = (n_correct / len(scored)) if scored else None
    print(f"\n{'='*60}", flush=True)
    print(f"  Model   : searchr1", flush=True)
    print(f"  Path    : {model_id}", flush=True)
    print(f"  Samples : {len(results)}", flush=True)
    if scored:
        print(f"  EM      : {n_correct}/{len(scored)} = {em:.3f}", flush=True)
        print(f"  F1      : {avg_f1:.3f}", flush=True)
    else:
        print("  EM      : N/A (no ground-truth answers in input)", flush=True)
        print("  F1      : N/A (no ground-truth answers in input)", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
