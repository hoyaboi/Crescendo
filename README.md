# Crescendo Attack Evaluation

An experimental framework for evaluating Crescendo attacks. This project uses PyRIT's CrescendoAttack to execute multi-turn attacks and analyze the results.

## Overview

The Crescendo attack is a strategy that gradually guides a model to generate harmful content. This project provides tools to systematically evaluate Crescendo attacks and store the results.

## Reference

This project implements and evaluates the Crescendo attack proposed in the following paper:

> **Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack**

## Key Features

- Execute and evaluate Crescendo attacks
- Support for OpenAI and HuggingFace models
- Batch processing of multiple tasks
- Detailed logging for each turn (original/converted prompts, responses)
- Automatic result saving and summarization (incremental save for each task)
- API key management via .env file

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd crescendo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file and set your API keys:

```bash
# OpenAI API (when using OpenAI models)
OPENAI_API_KEY=your-openai-api-key

# HuggingFace (when using HuggingFace models)
HUGGINGFACE_TOKEN=your-huggingface-token
```

## Usage

### Basic Execution

```bash
python crescendo_eval.py
```

### Model Selection

```bash
# Set all models to gpt-4o
python crescendo_eval.py \
  --target-model gpt-4o \
  --attacker-model gpt-4o \
  --judge-model gpt-4o \
  --refusal-judge-model gpt-4o
```

### Adjust Experiment Parameters

```bash
python crescendo_eval.py \
  --max-turns 5 \
  --max-backtracks 3 \
  --objective-threshold 0.8
```

### Use Custom Task File

```bash
python crescendo_eval.py --tasks-file data/your_tasks.json
```

### Control Prompt Converter

By default, the emoji converter (EmojiConverter) is enabled. To disable the converter:

```bash
python crescendo_eval.py --no-converter
```

### Utility Commands

```bash
# List available models
python crescendo_eval.py --list-models

# Check API key status
python crescendo_eval.py --check-keys
```

## Command Line Options

### Model Settings

- `--target-model`: Target model to attack (default: gpt-4o-mini)
- `--attacker-model`: Model for generating attack prompts (default: gpt-4o-mini)
- `--judge-model`: Model for judging objective achievement (default: gpt-4o-mini)
- `--refusal-judge-model`: Model for detecting refusals (default: gpt-4o-mini)

### Experiment Parameters

- `--max-turns`: Maximum number of turns (default: 10)
- `--max-backtracks`: Maximum number of backtracks (default: 10)
- `--objective-threshold`: Objective achievement threshold (default: 0.8)
- `--no-converter`: Disable prompt converter (default: converter enabled)

### File Settings

- `--tasks-file`: Path to task JSON file (default: data/custom_tasks.json)
- `--output-file`: Result filename (default: auto-generated)

## Task File Format

Tasks are defined in JSON files:

```json
[
  {
    "name": "Task Name",
    "objective": "Attack objective description"
  }
]
```

## Supported Models

The following models are currently supported:

- **OpenAI**: `gpt-4`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **HuggingFace**: `llama-2-7b`, `llama-2-70b`, `llama-3-8b`, `llama-3-70b`

All available models can be checked using the `--list-models` option.

To add a new model, refer to the comments in the `models/model_configs.py` file.

## Results

### Result Files

Experiment results are saved in JSON format in the `outputs/results/` directory. **Results are automatically saved each time a task completes**, so even if the program is interrupted, all results up to that point are preserved.

Each result file contains the following information:

- Task name and objective
- Success status
- Number of turns used
- Execution time
- Final turn summary (Attacker Prompt, Target Response)
- **Turn history** (`turn_history`): Detailed information for all turns
  - Original prompt for each turn (`attacker_original`)
  - Converted prompt for each turn (`attacker_converted`, emoji conversion, etc.)
  - Target response for each turn (`target_response`)
  - Judge score for each turn (`judge_score`): A float value between 0.0 and 1.0 indicating the degree of objective achievement for that turn (based on threshold 0.8)
  - Backtracked prompt-response pairs for each turn (`backtracked_pairs`): A list of prompt-response pairs that were rejected and backtracked in that turn
    - Each backtracked pair includes `attacker_original`, `attacker_converted`, `target_response`, and `refusal_judge` (boolean) information
  - Backtrack count for each turn (`backtrack_count`): The number of `backtracked_pairs`
- Experiment settings

### Logging Files

Detailed logging for each task is saved separately in the `outputs/logs/` directory:

- File name format: `turn_logs_{task_name}_{timestamp}.json`
- Detailed information for each turn (original/converted prompts, responses)
- Only actual turns are recorded, excluding backtracks

## Project Structure

```
crescendo/
├── crescendo_eval.py          # Main execution script
├── src/
│   ├── orchestrator.py        # CrescendoExperiment class
│   └── utils.py               # Utility functions (result saving, logging)
├── models/
│   ├── model_configs.py       # Model configurations
│   └── model_factory.py       # Model factory
├── data/
│   └── custom_tasks.json      # Task definitions
├── outputs/
│   ├── results/               # Result storage directory
│   └── logs/                  # Turn logging storage directory
├── requirements.txt           # Dependency list
└── README.md                  # This file
```

## Requirements

- Python 3.11+
- PyRIT framework
- OpenAI API key (when using OpenAI models)
- HuggingFace token (when using HuggingFace models)
