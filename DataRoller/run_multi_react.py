import argparse
import json
import os
from datetime import datetime

from prompt import SYSTEM_PROMPT, USER_PROMPT
from react_agent import MultiTurnReactAgent
import tool_draw  # noqa: F401
import tool_imagesearch  # noqa: F401
import tool_textsearch  # noqa: F401


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "doubao2.0")
DEFAULT_DATASET = os.getenv("DATASET", "gen_sft")
DEFAULT_OUTPUT = os.getenv("OUTPUT_PATH", os.path.join(BASE_DIR, "outputs"))
DEFAULT_MAX_ITEMS = int(os.getenv("MAX_ITEMS", "1"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "1.0"))
DEFAULT_ARK_BASE_URL = os.getenv(
    "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
)


def _load_items(path: str):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError("Input JSON must be a list of objects.")
        return items

    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")


def _extract_question(item):
    question = (item.get("question") or "").strip()
    if question:
        return question

    try:
        user_msg = item["messages"][1]["content"]
        return user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
    except Exception:
        return ""


def _build_error_result(task_info, exc):
    return {
        "question": task_info["item"].get("question", ""),
        "answer": task_info["item"].get("answer", ""),
        "rollout_id": task_info["rollout_id"],
        "error": f"Run failed: {exc}",
        "messages": [],
        "prediction": "[Failed]",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_items", type=int, default=DEFAULT_MAX_ITEMS)
    args = parser.parse_args()

    model_id = args.model
    model_name = args.model_name
    dataset_name = args.dataset
    output_base = args.output

    model_dir = os.path.join(output_base, model_name)
    dataset_dir = os.path.join(model_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Model id: {model_id}")
    print(f"Model name: {model_name}")
    print(f"Dataset name: {dataset_name}")
    print(f"Output directory: {dataset_dir}")
    print(f"Max items: {args.max_items}")

    data_filepath = os.path.join(BASE_DIR, "data", f"{dataset_name}.jsonl")
    try:
        items = _load_items(data_filepath)
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        raise SystemExit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        raise SystemExit(1)

    if args.max_items > 0:
        items = items[: args.max_items]

    print(f"Loaded {len(items)} item(s) from {data_filepath}")
    if not items:
        print("No items to run.")
        raise SystemExit(0)

    llm_cfg = {
        "model": model_id,
        "api_key": os.getenv("ARK_API_KEY", ""),
        "base_url": DEFAULT_ARK_BASE_URL,
        "generate_cfg": {
            "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "max_retries": 3,
        },
        "model_type": "oai",
        "model_server": DEFAULT_ARK_BASE_URL,
    }

    system_message = SYSTEM_PROMPT + "\nCurrent date: " + datetime.now().strftime("%Y-%m-%d")
    agent = MultiTurnReactAgent(
        llm=llm_cfg,
        function_list=["text_search", "draw", "image_search"],
        system_message=system_message,
    )

    output_file = os.path.join(dataset_dir, "iter1.jsonl")
    with open(output_file, "a", encoding="utf-8") as f:
        for idx, item in enumerate(items, 1):
            question = _extract_question(item)
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            task = {"item": item.copy(), "rollout_id": 1}
            print(f"Running item {idx}/{len(items)}: {question}")

            try:
                result = agent._run(task, model_id, USER_PROMPT)
            except Exception as exc:
                print(f'Task for question "{question}" generated an exception: {exc}')
                result = _build_error_result(task, exc)

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Results written to {output_file}")
