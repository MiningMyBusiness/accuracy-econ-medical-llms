# Economics of Accuracy in Medical LLMs

This repository contains code and experiments for studying the relationship between computational budget (token allocation) and accuracy in medical language models, with a focus on test-time compute scaling.

## Overview

This project investigates how different reasoning strategies and token budgets affect the accuracy of language models on medical question-answering benchmarks. The research explores various prompting techniques including:

- **B0/P0**: Direct answer (no reasoning)
- **B1/P1**: Standard Chain-of-Thought (CoT)
- **B2**: Short-CoT with token caps
- **B3/P2**: Self-Consistency (k=5 samples with majority voting)
- **B4**: Reflection-based reasoning (two-pass)
- **B5**: RAG-only (extractive from context)
- **B6/PCCR**: Iterative CoT (multi-pass reasoning until word count threshold)

## Key Components

### Benchmarking Infrastructure

- **`mirage_benchmark_s1.py`**: Main benchmarking script for evaluating models on the MIRAGE medical benchmark using various baseline strategies
- **`hf_benchmark_s1.py`**: HuggingFace-based benchmarking for datasets like PubMedQA and medical exams
- **`transformer_budgetforcer.py`**: Custom LangChain wrapper implementing budget-forcing for single-pass s1-style reasoning with token budget constraints
- **`vllm_budgetforcer.py`**: vLLM-based implementation of budget forcing

### Prompt Engineering

- **`prompt_builders.py`**: Utilities for constructing prompts with different reasoning strategies, context formatting, and answer extraction
- **`distractor_utils.py`**: Tools for handling distractor options in multiple-choice questions

### Batch Processing & Orchestration

- **`run_hf_benchmark_orchestrator.py`**: Orchestrates batch experiments across different models and configurations
- **`run_gemma_batch_experiments_transformers_bf.py`**: Batch experiments for Gemma models with budget forcing
- **`run_mirage_s1_seq.py`**: Sequential execution of MIRAGE benchmark experiments

### Analysis & Visualization

- **`plot_econ_of_acc.py`**: Creates 2x2 panel plots showing accuracy vs. token budget across datasets
- **`plot_b6_iterations_boxplot.py`**: Visualizes iteration statistics for B6 (Iterative CoT) experiments
- **`cost-accuracy-plot.py`**: Analyzes cost-accuracy tradeoffs

### Data Processing

- **`grab_b6_iteration_data.py`**: Extracts iteration statistics from B6 experiment results
- **`reextract_final_answers_b6.py`**: Post-processes B6 results to extract final answers
- **`fix_pubmedqa_ground_truth.py`**: Corrects ground truth labels in PubMedQA dataset

## Datasets

The codebase supports multiple medical benchmarks:

- **MIRAGE**: Medical reasoning benchmark
- **PubMedQA**: Biomedical question answering (with and without context)
- **Medical Exams**: Clinical examination questions

## Models Evaluated

- Gemma-3 (1B, 4B, 27B)
- MedGemma (4B, 27B) - medically fine-tuned variants

## Results

Results are stored in the `results/` directory as JSONL files. Key findings are summarized in:

- `econ-of-acc-table.csv`: Accuracy vs. budget data
- `b6_iterations_table.csv`: Iteration statistics for iterative reasoning
- `figures/`: Generated plots and visualizations

## Usage

Example for running MIRAGE benchmark with different baselines:

```bash
python mirage_benchmark_s1.py --output_suffix experiment_name \
    --model_name qwen2.5vl:3b \
    --baseline_style B1 \
    --max_questions 100
```

For budget-forcing experiments with transformers:

```bash
python run_gemma_batch_experiments_transformers_bf.py
```

## Configuration

Model and experiment configurations are stored in `configs/` directory as JSON files, tracking parameters like model size, baseline style, token budgets, and timestamps.