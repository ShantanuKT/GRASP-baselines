# Search-R1 Inference (Chunk Index)

Minimal setup for running **Search-R1** multi-hop QA with a local chunk-level retriever.

The model reasons in `<think>`, issues search queries in `<search>…</search>`, and answers in `<answer>…</answer>`. Each search is served by `index_server.py` from a precomputed `chunk_index.pkl`.

## Files

| File | Role |
|------|------|
| `index_server.py` | FastAPI retriever: loads `chunk_index.pkl`, encodes queries, returns top-k chunks |
| `inference_searchr1.py` | Search-R1 agent loop, EM/F1 scoring, JSONL output |

## Requirements

- Python 3.9+
- GPU recommended for the LLM (`device_map="cuda"`)
- Packages: `torch`, `transformers`, `sentence-transformers`, `fastapi`, `uvicorn`, `pydantic`, `numpy`, `requests`, `pandas`

The retriever encoder is taken from `model_name` inside the pickle (or `--encoder_model`). `sentence-transformers` is tried first; if that fails, Hugging Face mean-pooling is used.

## Index pickle format

`chunk_index.pkl` must be a dict with at least:

- `chunk_ids`: list of chunk IDs (row `i` of embeddings)
- `embeddings`: `[N, D]` array or tensor
- `chunks`: dict or list of `{id, title, text, ...}` (optional but needed for readable docs)
- `model_name`: encoder id used when the index was built (optional if you pass `--encoder_model`)

Embeddings are L2-normalized at load time; retrieval is cosine similarity.

## 1. Start the retriever

```bash
python index_server.py \
  --chunk_index_pkl /path/to/chunk_index.pkl \
  --port 8000
```

Useful flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk_index_pkl` | required | Path to the pickle index |
| `--port` | `8001` | HTTP port |
| `--host` | `0.0.0.0` | Bind address |
| `--topk` | `5` | Default k if the request omits `topk` |
| `--device` | `cuda:0` if available, else `cpu` | Encoder device |
| `--encoder_model` | pickle `model_name` | Override encoder |

### Retrieve API

`POST /retrieve`

```json
{
  "queries": ["who directed the film?"],
  "topk": 5,
  "return_scores": false
}
```

Response:

```json
{
  "result": [
    [
      {
        "id": "...",
        "title": "...",
        "text": "...",
        "contents": "title\\ntext"
      }
    ]
  ]
}
```

If `return_scores` is true, each hit is `{"document": {...}, "score": 0.12}`.

The inference client also tries `POST /search` if `/retrieve` is not used.

## 2. Run Search-R1 inference

Keep the retriever running, then:

```bash
python inference_searchr1.py \
  --output_file results/2wiki/grpo.jsonl \
  --retriever_url http://127.0.0.1:8000 \
  --model_id PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.2 \
  --topk 5 \
  --questions data/2wiki_questions_500.json \
  --resume
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output_file` | required | JSONL path (one object per question) |
| `--retriever_url` | required | Base URL (`http://host:port`) or full `/retrieve` URL |
| `--model_id` | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo` | Hugging Face model |
| `--questions` | unset | JSON list of question objects |
| `--test_data` | hardcoded parquet path | Used only if `--questions` is omitted |
| `--n_samples` | `1000` | Max questions to load |
| `--max_turns` | `10` | Max search/answer turns |
| `--topk` | `5` | Passages per search |
| `--resume` | off | Skip qids already in `--output_file` and append |

Generation uses temperature `0.6`, up to `512` new tokens per turn, and stops on `</search>` or `</answer>`.

### Questions JSON

A JSON array of objects. Question is taken from `question`, `query`, or `extra_info.question`. Gold answers from `answer`, `answers`, `ground_truth`, `gold`, `target`, or `reward_model.ground_truth.target`. IDs from `qid` or `id`.

Without `--questions`, samples are read from a parquet file with `extra_info.question` and `reward_model.ground_truth.target`.

## Output

Each line is a JSON object:

```json
{
  "qid": "...",
  "question": "...",
  "ground_truth": "...",
  "predicted": "...",
  "correct": 1,
  "f1": 1.0,
  "tool_calls": [{"tool": "search", "query": "...", "chunk_ids": ["..."]}],
  "chunk_ids": ["..."],
  "trajectory": "..."
}
```

`correct` is exact match after lowercasing, stripping articles/punctuation, and whitespace normalize. `f1` is token-level F1 against the best gold string. If there is no gold answer, those fields are `null`.

Results are flushed after every question. With `--resume`, existing qids are skipped and new lines are appended. At the end the script prints EM and mean F1 over scored examples.

## Typical two-process run

```bash
# Terminal 1
python index_server.py --chunk_index_pkl data_baselines/2wiki_index.pkl --port 8000

# Terminal 2
python inference_searchr1.py \
  --output_file results_7b/2wiki/grpo.jsonl \
  --retriever_url http://127.0.0.1:8000 \
  --model_id PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.2 \
  --topk 5 \
  --questions data_baselines/2wiki_questions_500.json \
  --resume
```
