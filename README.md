# ToolArtist

ToolArtist is a post-training project for tool-augmented image generation agents. The currently released data construction module is **DataRoller**, which rolls out multi-turn agent trajectories with web search, image search, and image generation tools. The generated trajectories can be used for later SFT and RL stages.

> [!NOTE]
> This repository is organized around three stages: **Data Construction**, **SFT**, and **RL**. The open-source part currently focuses on DataRoller.

## Data Construction

`DataRoller/` is the data construction module of ToolArtist. Given a batch of image generation requests, it runs a ReAct-style multimodal agent that can proactively search for factual information, retrieve reference images, generate images, inspect the generated results, and save the full rollout trajectory.

The workflow is:

1. Load image generation tasks from `DataRoller/data/{DATASET}.jsonl`.
2. Use `text_search` to retrieve factual information from the web.
3. Use `image_search` to obtain visual references.
4. Use `draw` to generate or edit images.
5. Let the agent inspect the generated result and iterate when needed.
6. Save the complete messages, final prediction, termination status, and generated image path.

### Project Structure

```text
DataRoller/
├── data/
│   ├── gen_sft.jsonl
│   └── gen_rl.jsonl
├── scrpits/
│   └── run.sh
├── prompt.py
├── react_agent.py
├── run_multi_react.py
├── tool_draw.py
├── tool_imagesearch.py
├── tool_reader.py
├── tool_textsearch.py
└── requirements.txt
```

Main files:

- `run_multi_react.py`: Entry point for data rollout. It reads `data/{DATASET}.jsonl` and writes rollout results.
- `react_agent.py`: Multi-turn ReAct agent implementation, including model calls, tool calls, multimodal message handling, and output formatting.
- `prompt.py`: System prompt and user prompt template.
- `tool_textsearch.py`: Web text search tool based on Serper and Jina Reader.
- `tool_imagesearch.py`: Image search tool based on Serper Images and Jina Reader.
- `tool_draw.py`: Image generation and image editing tool based on the Gemini image API.
- `tool_reader.py`: Ark-model-based document reader for extracting query-relevant information from fetched web pages.

### Environment Setup

We recommend Python 3.10 or higher.

```bash
cd DataRoller

conda create -n toolartist-dataroller python=3.10 -y
conda activate toolartist-dataroller
```

Install dependencies:

```bash
pip install "qwen-agent[gui,rag,code_interpreter,mcp]"
pip install soundfile openai pillow requests tiktoken rich
```

The current `requirements.txt` keeps the original install commands. You can also run:

```bash
bash requirements.txt
pip install openai pillow requests tiktoken rich
```

### API Configuration

DataRoller requires three external services:

- `ARK_API_KEY`: Used by the main rollout model and the document reader.
- `SERPER_API_KEY`: Used by text search and image search.
- `GEMINI_API_KEY`: Used by the `draw` tool for image generation and editing.

Set the environment variables before running:

```bash
export ARK_API_KEY="your-ark-api-key"
export ARK_MODEL="doubao-seed-2-0-pro-260215"
export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"

export SERPER_API_KEY="your-serper-api-key"

export GEMINI_API_KEY="your-gemini-api-key"
export GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta"
export GEMINI_MODEL="gemini-3-pro-image"
export GEMINI_ASPECT_RATIO="4:3"
export GEMINI_IMAGE_SIZE="1K"
export GEMINI_RESPONSE_MIME_TYPE="image/png"
```

Optional runtime settings:

```bash
export MODEL_NAME="doubao2.0"
export DATASET="gen_sft"
export OUTPUT_PATH="./outputs"
export MAX_ITEMS=10
export MAX_OUTPUT_TOKENS=4096
export TEMPERATURE=0.0
export TOP_P=1.0

# Optional proxies, if required by your environment.
export DEEP_SEARCH_PROXY=""
export DEEP_BROWSE_PROXY=""
```

Do not commit real API keys. The keys in `DataRoller/scrpits/run.sh` are placeholders and should be replaced locally or read from your shell environment.

### Input Data

Input files are placed under `DataRoller/data/`. The filename is selected by `DATASET`:

```text
DATASET=gen_sft -> DataRoller/data/gen_sft.jsonl
DATASET=gen_rl  -> DataRoller/data/gen_rl.jsonl
```

Each line should be a JSON object. Recommended format:

```json
{"idx": 1, "question": "A professional conference stage scene ...", "answer": "xxx"}
```

Fields:

- `question`: Required. The image generation request.
- `answer`: Required. The reference answer or placeholder answer. It will be copied into the output.
- `idx`: Optional sample index.

The script also supports some `messages`-style inputs, as long as it can extract the user question. Each sample still needs an `answer` field.

### Running DataRoller

Run the rollout entry directly:

```bash
cd DataRoller

python3 -u run_multi_react.py \
  --model "$ARK_MODEL" \
  --model_name "$MODEL_NAME" \
  --dataset "$DATASET" \
  --output "$OUTPUT_PATH" \
  --max_items "$MAX_ITEMS"
```

Example: run the first 10 samples from `gen_sft`:

```bash
cd DataRoller

export DATASET="gen_sft"
export MODEL_NAME="doubao2.0"
export OUTPUT_PATH="./outputs"
export MAX_ITEMS=10

python3 -u run_multi_react.py \
  --dataset gen_sft \
  --model_name doubao2.0 \
  --output ./outputs \
  --max_items 10
```

You can also use the existing helper script:

```bash
cd DataRoller
bash scrpits/run.sh
```

Note that `scrpits/run.sh` removes the configured output directory before running:

```bash
rm -rf "${OUTPUT_PATH:?}/${MODEL_NAME}/${DATASET}"
```

If you want to keep previous rollouts, call `run_multi_react.py` directly or edit the cleanup logic in `scrpits/run.sh`.

### Outputs

Main rollout file:

```text
DataRoller/outputs/{MODEL_NAME}/{DATASET}/iter1.jsonl
```

Default generated image directory:

```text
DataRoller/outputs/tool_resp_sft_en/
```

To change the image output directory:

```bash
export DRAW_OUTPUT_DIR="./outputs/tool_images"
```

Each line in `iter1.jsonl` is one rollout result with fields such as:

- `question`: Original image generation request.
- `answer`: Reference answer from the input data.
- `rollout_id`: Rollout id, currently `1`.
- `messages`: Full multi-turn conversation and tool-call trajectory.
- `prediction`: Final boxed answer, usually including the generated image path or failure reason.
- `termination`: Termination status, such as `answer`, `answer not found`, or `exceed available llm calls`.
- `error`: Present when a task fails with an exception.

Failed samples do not stop the whole run. They are written as JSONL records with an `error` field, so they can be filtered or rerun later.

### Troubleshooting

**Missing `ARK_API_KEY`**

Both the main model and the document reader require Ark access:

```bash
export ARK_API_KEY="your-ark-api-key"
```

**Empty search results**

Check that `SERPER_API_KEY` is valid and that your environment can access:

```text
https://google.serper.dev/search
https://google.serper.dev/images
```

**Image generation failure**

Check that `GEMINI_API_KEY` is valid and that `GEMINI_MODEL` and `GEMINI_BASE_URL` match the image generation service available to you.

**Large output files**

`react_agent.py` clears `image_url` blocks before writing user messages to JSONL, which avoids storing large base64 images in the rollout file. Full tool trajectories can still be long, so start with a small `MAX_ITEMS` value when testing.

## SFT

The SFT stage turns rollout traces into tokenized training data and then fine-tunes Emu3.5. The v2 pipeline is the one to use now: it works directly on repo-relative paths, uses the current rollout format, and runs conversion in parallel across GPUs.

### Environment Setup

Use one of the following setups. Both target Python 3.12.

#### Option 1: Conda

```bash
conda create -n toolartist-sft python=3.12 -y
conda activate toolartist-sft
bash env_scripts/sft_env.sh
```

This installs the SFT runtime stack, including torch 2.8.0, flash-attn, vLLM dependencies, transformers, datasets, trl, deepspeed, and the helper packages used by the training scripts.

#### Option 2: Docker

```bash
bash env_scripts/build_images.sh sft
docker run --gpus all -it --rm -v "$(pwd)":/workspace local/univr-sft:cu128-py312
```

If you prefer to build the image directly, use `env_scripts/Dockerfile.sft.public`:

```bash
docker build -f env_scripts/Dockerfile.sft.public -t local/univr-sft:cu128-py312 env_scripts
```

The image already creates a Python 3.12 conda env and starts in `/workspace`.

If you still have an older rollout dump with legacy prompts, normalize it with `SFT/replace_pe.py` before converting. The checked-in file at `Data/SFT/UPE_raw_gensearcher_sft_trace_relpath.jsonl` already matches the current workflow, so most runs can skip this step.

### 1. Build the training set

```bash
bash SFT/prepare_v2.sh
```

By default this reads `Data/SFT/UPE_raw_gensearcher_sft_trace_relpath.jsonl`, uses `Emu3.5/src/tokenizer_emu3_ibq` and `Emu3.5-VisionTokenizer`, loads generated images from `Data/SFT/image`, and writes `Data/SFT/converted_v2/sft.jsonl`.

### 2. Spot-check samples

```bash
python3 SFT/verify_sample.py \
  --sft Data/SFT/converted_v2/sft.jsonl \
  --raw Data/SFT/UPE_raw_gensearcher_sft_trace_relpath.jsonl \
  --tool-resp-dir Data/SFT/image \
  --vq-path Emu3.5-VisionTokenizer \
  --tokenizer-path Emu3.5/src/tokenizer_emu3_ibq \
  --out SFT/verify_out \
  --line-nos 1 5 11
```

This is optional. It decodes a few samples back into images and writes a walkthrough under `SFT/verify_out/`, which is useful before a full training run.

### 3. Fine-tune

```bash
bash SFT/sft.sh
```

`SFT/sft.sh` reads `Data/SFT/converted_v2/sft.jsonl` and writes to `outputs/sft` by default. Update the model path in `SFT/sft.sh` to point at your local Emu3.5 checkpoint before running.

## RL

The RL stage fine-tunes the SFT checkpoint with the same agentic rollout loop used for validation and data generation. The same rollout behavior is shared by the standalone service and the in-process GRPO worker.

### Where Things Live

- `Agentic_Image_Gen/run.py`: canonical multi-turn agent loop
- `Agentic_Image_Gen/server.py`: standalone HTTP service with `/encode_images`, `/generate`, and `/health`
- `Agentic_Image_Gen/start.sh`: launches the service
- `Agentic_Image_Gen/run.sh`: single-dataset rollout client
- `Agentic_Image_Gen/run_all.sh`: sharded rollout client
- `RL/UniVR_RL/verl/workers/rollout/emu_agentic/`: in-process rollout backend used by GRPO training
- `RL/UniVR_RL/examples/reward_function/`: reward functions
- `RL/UniVR_RL/examples/config_emu_agentic_200step_big.yaml`: current public RL config

### Environment

Use the RL Docker image or any local Python 3.12 environment with the same dependencies.

Build the public image:

```bash
bash env_scripts/build_images.sh rl
```

Or build directly:

```bash
docker build -f env_scripts/Dockerfile.rl.public -t local/univr-rl:cu128-py312 env_scripts
```

The image includes the RL dependencies, vLLM patches, and rollout runtime packages.

### Standalone Rollout Service

Start the service by pointing the rollout stack at your local model, VQ tokenizer, Emu3.5 source tree, and image output directory:

```bash
EMU_MODEL_PATH=... \
EMU_VQ_PATH=... \
EMU_ROOT=... \
EMU_IMAGE_SAVE_DIR=... \
bash Agentic_Image_Gen/start.sh
```

Then run a client against it:

```bash
EMU_SERVER_URL=http://127.0.0.1:23333 \
IMAGE_SAVE_DIR=./outputs/rollout/images \
bash Agentic_Image_Gen/run.sh Data/RL/gen_rl.jsonl ./outputs/rollout
```

For multi-GPU rollout, use `Agentic_Image_Gen/start_all.sh` together with `Agentic_Image_Gen/run_all.sh`.

This mode is useful for validation, dataset generation, and debugging without starting RL training.

### RL Training

The current public launcher is:

```bash
bash RL/UniVR_RL/examples/emu_agentic_grpo_200step_big.sh
```

That launcher wires together:

- `EMU_MODEL_PATH`: SFT checkpoint to optimize
- `EMU_VQ_PATH`: vision tokenizer
- `EMU_ROOT`: Emu3.5 source tree
- `EMU_AGENT_TOOL_DIR`: tool implementations
- `EMU_AGENTIC_TRAIN_DATA`: training JSONL
- `ARK_API_KEY`: required by the Doubao judge
- `ARK_MODEL`: judge model, defaulting to the configured Doubao model
- `ARK_BASE_URL`: Ark base URL for the judge
- `WANDB_MODE`: `online` or `offline`

The reward function lives in `RL/UniVR_RL/examples/reward_function/emu_agentic_gensearcher_dual.py`.
The rollout backend lives in `RL/UniVR_RL/verl/workers/rollout/emu_agentic/` and reuses the same `Agentic_Image_Gen/run.py` logic as the standalone service.

### Notes

- Keep repo docs relative, not tied to one machine path.
- Do not commit personal tokens or private credentials.
- If you use online wandb, provide your own credentials via login or a mounted `~/.netrc`.
