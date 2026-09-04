# IRCoT inference

IRCoT retrieves BM25 passages from Elasticsearch, generates step-by-step CoT with an OpenRouter chat model (`openai/gpt-5-mini`), then produces the final answer with a local Qwen2.5-7B Instruct server.

Clone IRCOT repository from : https://github.com/StonyBrookNLP/ircot

## Layout

| Path | Role |
|------|------|
| `ircot_inference.py` | Inference entrypoint (writes under `--pred-dir`) |

## Requirements

- Conda env `ircot` (as used on the cluster), or install `requirements.txt`
- GPU for the local Qwen vLLM server
- Java is bundled with `elasticsearch-7.10.2/jdk`
- Hugging Face cache (optional): the original job used `HF_HOME=/work/pi_dagarwal_umass_edu/project_22/stodmal/hf_cache`
- `configs/.env` must export `ARAG_API_KEY` (OpenRouter) and typically `HF_TOKEN`

Default ports (override with env vars if needed):

- Elasticsearch: `9200`
- Retriever: `8000` (`RETRIEVER_PORT`)
- LLM server: `8010` (`LLM_SERVER_PORT`)

If you change retriever or LLM ports, export `RETRIEVER_PORT` / `LLM_SERVER_PORT` (and `RETRIEVER_HOST` / `LLM_SERVER_HOST` if not localhost) **before** `python run_questions_with_chunks_pred_dir.py`. The servers must listen on the same ports.

## 1. Start Elasticsearch

```bash
cd /path/to/ircot_inference
mkdir -p logs
ES_DATA_DIR=/tmp/${USER}/es_ircot
mkdir -p "$ES_DATA_DIR"
cd elasticsearch-7.10.2
./bin/elasticsearch \
    -Epath.data="$ES_DATA_DIR" \
    -Ehttp.port=9200
```

Wait until `curl -fsS http://localhost:9200/` succeeds.

## 2. Start the retriever

From this folder:

```bash
cd /path/to/ircot_inference
uvicorn serve:app --port 8000 --host 0.0.0.0 --app-dir retriever_server
```

Health check: `curl -fsS http://localhost:8000/`

The retriever talks to Elasticsearch at `localhost:9200`.

## 3. Start the local Qwen LLM server

```bash
cd /path/to/ircot_inference
source configs/.env
MODEL_NAME=qwen2.5-7b-instruct uvicorn serve:app \
    --port 8010 \
    --host 0.0.0.0 \
    --app-dir llm_server
```

Health check: `curl -fsS http://localhost:8010/`

`MODEL_NAME` must be a key in `llm_server/serve.py` (`qwen2.5-7b-instruct` → `Qwen/Qwen2.5-7B-Instruct`).

## 4. Run IRCoT

Keep the three services running, then:

```bash
cd /path/to/ircot_inference
source configs/.env
export RETRIEVER_PORT=8000
export LLM_SERVER_PORT=8010

python run_questions_with_chunks_pred_dir.py \
    --config base_configs/ircot_qa_gpt5mini_openrouter_qwen2_5_3b_instruct_hotpotqa.jsonnet \
    --questions-json data_baselines/2wiki_questions_500.json \
    --chunks-json data_baselines/2wiki_chunks.json \
    --index-name wiki \
    --force-reindex \
    --force-predict \
    --pred-dir results_7b/2wiki
```

This will:

1. Write processed JSONL to `processed_data/hotpotqa/questions_from_data_json.jsonl`
2. Build (or rebuild) Elasticsearch index `wiki` from `2wiki_chunks.json`
3. Run IRCoT and write predictions under `results_7b/2wiki/ircot_qa_gpt5mini_openrouter_qwen2_5_3b_instruct_hotpotqa/`

Omit `--force-reindex` / `--force-predict` to reuse an existing index and resume incomplete predictions.

## Output

Under `--pred-dir` / `<config_basename>/`:

- `prediction__hotpotqa_to_hotpotqa__questions_from_data_json.json`
- `*_chains.txt`, `*_reasoning_steps.json`, `*_time_taken.txt`, `*_run_metadata.json`