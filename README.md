# GRASP: Baselines and Evaluation

This repository contains baseline implementations and evaluation materials used to compare **GRASP** against single-step retrieval, prompting-based agentic retrieval, and reinforcement-learning-based retrieval methods for multi-hop question answering.

GRASP learns to coordinate semantic search, keyword search, and contextual reading, selecting retrieval actions and evidence granularity during multi-step reasoning. The baseline experiments examine how retrieval paradigms and reward designs affect evidence recall and answer quality.

## Repository structure

| Directory | Baseline | Role in evaluation |
|---|---|---|
| `Single-step/` | Lexical, semantic, and hybrid retrieval | Tests retrieval without iterative agentic search |
| `IRCoT/` | Interleaving Retrieval with Chain-of-Thought | Prompting-based multi-step retrieval |
| `Search-R1/` | PPO and GRPO variants | RL-based multi-step retrieval with a single semantic retriever |
| `AutoRefine/` | AutoRefine | RL-based retrieval with answer-oriented refinement |
| `ReasonRAG/` | ReasonRAG | RL-based approach with process-level feedback |

Each directory contains method-specific materials. Refer to its scripts and configuration files for runnable entry points; the table above describes the experimental role rather than guaranteeing an identical layout in every directory.

## Experimental setup

### Datasets

We evaluate on three multi-hop QA benchmarks:

- **HotpotQA**
- **2WikiMultiHopQA**
- **MuSiQue**

### Models and comparisons

The retrieval-paradigm comparison uses **Qwen2.5-3B-Instruct**. The reward-design comparison uses **Qwen2.5-7B-Instruct**, including AutoRefine and ReasonRAG. The paper additionally reports GRASP at both scales.

| Method | Retrieval | Evidence granularity | Policy / learning |
|---|---|---|---|
| Single-step lexical | Lexical | Paragraph | Prompting |
| Single-step semantic | Semantic | Paragraph | Prompting |
| Single-step hybrid | Lexical + semantic | Paragraph | Prompting |
| IRCoT | Lexical | Paragraph | CoT prompting |
| Search-R1 | Semantic | Paragraph | PPO / GRPO |
| AutoRefine | Semantic | Paragraph | GRPO |
| ReasonRAG | Semantic | Paragraph | DPO |
| GRASP (proposed method) | Hybrid tools | Sentence-level search + contextual reading | GRPO |

For single-step hybrid retrieval, the paper retrieves 50 candidates from each retriever, merges and deduplicates them, reranks with **Qwen3-Reranker-0.6B**, and retains the top **5**. The evaluation uses retrieval depth **top-k = 5** across methods. IRCoT uses **gpt5-mini** for CoT traces.

### Evaluation metrics

- **Retrieval recall:** Coverage of gold supporting passages, aggregated over all retrieval steps for multi-step methods.
- **Exact Match (EM):** Exact normalized answer agreement.
- **Token-level F1:** Token overlap between predicted and gold answers.
- **Judge accuracy (JD):** Binary semantic correctness assessment by **gpt5-mini**, checking factual consistency and whether key answer information is present.

## Results reported in the paper

### Retrieval recall (3B comparison)

| Method | HotpotQA | 2Wiki | MuSiQue |
|---|---:|---:|---:|
| Single-step lexical | 0.61 | 0.49 | 0.35 |
| Single-step semantic | 0.73 | 0.61 | 0.47 |
| Single-step hybrid | 0.86 | 0.60 | 0.59 |
| IRCoT | 0.91 | 0.83 | 0.62 |
| Search-R1 (PPO) | 0.76 | 0.74 | 0.55 |
| Search-R1 (GRPO) | 0.74 | 0.77 | 0.62 |
| GRASP base (untrained) | 0.36 | 0.27 | 0.24 |
| **GRASP** | **0.90** | **0.90** | **0.70** |

*Source: Table 1 of the accompanying paper.*

### QA performance (3B comparison)

Values are **EM / F1 / JD**.

| Method | HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|
| Single-step lexical | 0.28 / 0.37 / 0.43 | 0.21 / 0.26 / 0.27 | 0.05 / 0.11 / 0.12 |
| Single-step semantic | 0.30 / 0.42 / 0.50 | 0.26 / 0.31 / 0.34 | 0.10 / 0.15 / 0.20 |
| Single-step hybrid | 0.37 / 0.48 / 0.56 | 0.27 / 0.32 / 0.35 | 0.13 / 0.20 / 0.23 |
| IRCoT | 0.24 / 0.36 / 0.63 | 0.27 / 0.40 / 0.58 | 0.08 / 0.15 / 0.25 |
| Search-R1 (PPO) | 0.37 / 0.47 / 0.50 | 0.42 / 0.48 / 0.49 | 0.18 / 0.26 / 0.26 |
| Search-R1 (GRPO) | 0.45 / 0.56 / 0.58 | 0.45 / 0.53 / 0.54 | 0.22 / 0.30 / 0.29 |
| GRASP base (untrained) | 0.17 / 0.24 / 0.27 | 0.15 / 0.20 / 0.21 | 0.04 / 0.10 / 0.10 |
| **GRASP (GRPO)** | **0.53 / 0.66 / 0.71** | **0.52 / 0.60 / 0.63** | **0.23 / 0.33 / 0.34** |

*Source: Table 2 of the accompanying paper.*

### QA performance (7B comparison)

Values are **EM / F1 / JD**.

| Method | HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|
| Search-R1 (PPO) | 0.48 / 0.60 / 0.63 | 0.51 / 0.59 / 0.59 | 0.25 / 0.33 / 0.35 |
| Search-R1 (GRPO) | 0.50 / 0.64 / 0.64 | 0.56 / 0.64 / 0.65 | 0.29 / 0.38 / 0.38 |
| AutoRefine (GRPO) | 0.52 / 0.65 / 0.69 | 0.50 / 0.58 / 0.60 | 0.31 / 0.41 / 0.42 |
| ReasonRAG (DPO) | 0.44 / 0.56 / 0.58 | 0.44 / 0.50 / 0.51 | 0.20 / 0.28 / 0.24 |
| **GRASP (GRPO)** | **0.58 / 0.72 / 0.75** | **0.59 / 0.66 / 0.68** | **0.32 / 0.41 / 0.42** |

*Source: Table 3 of the accompanying paper.*

## Reproduction notes

The baseline folders are organized by method, rather than by benchmark. To compare methods fairly, use the same benchmark splits, retrieval corpus, retrieval depth, answer normalization, and metric implementation. Record the model checkpoint, random seed, generation settings, and retrieved evidence for each run.

## Reference
[Paper Website](https://vgandhi13.github.io/GRASP-Project-Page/)
