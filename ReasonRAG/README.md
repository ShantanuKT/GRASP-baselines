# ReasonRAG baselines

Clone ReasonRAG from https://github.com/Applied-Machine-Learning-Lab/ReasonRAG

Generate prediction JSON files in `output/` for the shared `data_baselines` question splits (2Wiki, HotpotQA, MuSiQue).

## Environment

```bash
conda activate reasonrag
```

Cluster jobs also set Hugging Face caches:

```bash
export HF_HOME=""
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
```

GPU is recommended. `--engine auto` uses vLLM if available, otherwise Hugging Face.

Default generator: [`reasonrag/Qwen2.5-7B-Instruct-ReasonRAG`](https://huggingface.co/reasonrag/Qwen2.5-7B-Instruct-ReasonRAG)  
Default retriever: `BAAI/bge-base-en-v1.5`

## Data

All files live in `../data_baselines/` (repo root: `data_baselines/`).

| Dataset | Questions | Chunks | Extra |
|---------|-----------|--------|--------|
| 2Wiki | `2wiki_questions_500.json` | `2wiki_chunks.json` | — |
| HotpotQA | `hqa_questions_500.json` | `hqa_chunks.json` | — |
| MuSiQue | `musique_questions_500.json` | `Musique_chunks.json` | `musique_questions_500_chunk_ids_from_Musique_chunks.json` (required) |

## Run locally

```bash
conda activate reasonrag
cd ReasonRAG
```

### 2Wiki

```bash
python run_reasonrag_inference.py \
    --questions ../data_baselines/2wiki_questions_500.json \
    --chunks ../data_baselines/2wiki_chunks.json \
    --model reasonrag/Qwen2.5-7B-Instruct-ReasonRAG \
    --retriever-model BAAI/bge-base-en-v1.5 \
    --retrieval-scope per_question \
    --topk 3 \
    --max-iter 8 \
    --output output/2wiki.json
```

### HotpotQA

```bash
python run_reasonrag_inference.py \
    --questions ../data_baselines/hqa_questions_500.json \
    --chunks ../data_baselines/hqa_chunks.json \
    --model reasonrag/Qwen2.5-7B-Instruct-ReasonRAG \
    --retrieval-scope per_question \
    --topk 3 \
    --max-iter 8 \
    --output output/hotpotqa.json
```

### MuSiQue

```bash
python run_reasonrag_inference.py \
    --questions ../data_baselines/musique_questions_500.json \
    --chunks ../data_baselines/Musique_chunks.json \
    --chunk-ids ../data_baselines/musique_questions_500_chunk_ids_from_Musique_chunks.json \
    --model reasonrag/Qwen2.5-7B-Instruct-ReasonRAG \
    --retrieval-scope per_question \
    --topk 3 \
    --max-iter 8 \
    --output output/musique.json
```

`--retrieval-scope per_question` retrieves inside each question’s gold+distractor pool. Use `global` to retrieve over the full chunk file (requires `--chunks`). Add `--limit N` to run a subset.
