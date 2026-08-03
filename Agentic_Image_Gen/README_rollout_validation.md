# Rollout Validation

This note describes an optional smoke test for the standalone ToolArtist
rollout service in `Agentic_Image_Gen/`.

## Inputs

- Emu checkpoint: set `EMU_MODEL_PATH`, or place it at
  `checkpoints/emu3p5-sft`.
- VQ tokenizer: set `EMU_VQ_PATH`, or place it at
  `checkpoints/Emu3.5-VisionTokenizer`.
- Emu3.5 source root: defaults to `Emu3.5`.
- Agent tools: defaults to `DataRoller`.
- Dataset: defaults to `Data/RL/gen_rl.jsonl`.

## Single Server

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 \
EMU_TP_SIZE=1 \
EMU_PORT=23351 \
bash Agentic_Image_Gen/start.sh
```

In another shell:

```bash
EMU_SERVER_URL=http://127.0.0.1:23351 \
MAX_WORKERS=1 \
MAX_NEW_TOKENS=8192 \
FORCE_DRAW_ROUND=6 \
bash Agentic_Image_Gen/run.sh gen_rl rollout_validation/single/out
```

Generated traces are written under
`Agentic_Image_Gen/rollout_validation/single/out/`, and decoded images are
written under the configured `IMAGE_SAVE_DIR`.

## Multi-Server

`start_all.sh` launches one server per GPU and `run_all.sh` shards the JSONL
dataset across those servers.

```bash
GPUS=0,1,2,3,4,5,6,7 \
DATASET_NAME=gen_rl \
EMU_PORT_BASE=23400 \
bash Agentic_Image_Gen/start_all.sh
```

In another shell:

```bash
GPUS=0,1,2,3,4,5,6,7 \
EMU_PORT_BASE=23400 \
MAX_WORKERS=1 \
MAX_NEW_TOKENS=8192 \
FORCE_DRAW_ROUND=6 \
bash Agentic_Image_Gen/run_all.sh gen_rl rollout_validation/multi/out
```

Set `PROXY_URL` only if your environment needs an outbound proxy for image
downloads or search traffic.
