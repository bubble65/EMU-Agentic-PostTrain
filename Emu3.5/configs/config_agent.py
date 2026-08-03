# configs/agent_test.py
# Copyright 2025 BAAI. and/or its affiliates.

from pathlib import Path
from src.utils.logging_utils import setup_logger
cfg_name = Path(__file__).stem

model_path = "/opt/tiger/alpha-seed/Emu3.5/model/Emu3.5"
vq_path = "/opt/tiger/alpha-seed/Emu3.5/model/Emu3.5-VisionTokenizer"
tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

# 关键改动 1:用 howto / story / explore 之一,这些是长文本输出任务
# 不能用 t2i/x2i,否则碰到 <|image end|> 就停。
# howto 听起来最贴近 "Agent 给步骤" 的分布,先用它试。
task_type = "howto"
use_image = False

exp_name = "agent-test"
save_path = f"./outputs/{exp_name}/{task_type}"
save_to_proto = True
setup_logger(save_path)

hf_device = "auto"
vq_device = "cuda:0"
streaming = False
unconditional_type = "no_text"

# 关键改动 2:纯文本场景,关掉 CFG
classifier_free_guidance = 2.0
max_new_tokens = 32768  # 先小一点,看格式对不对再加
image_area = 1048576

# t2i 才需要分辨率,纯文本场景给 None
target_height, target_width = None, None


# 关键改动 3:不走 build_unc_and_template,自己写模板
# 把你的 system prompt 嵌进 "You are a helpful assistant" 那一段
SYSTEM_PROMPT = """你是一个"检索增强生图 Agent"。你的任务是根据用户的生图需求，主动检索网页信息和参考图片，然后自己生成图片。

行动原则：
- 涉及真实人物、地点、标志、服饰、建筑、IP风格等，优先用 text_search 获取文字描述，再用 image_search 获取参考图。
- 多张参考图严格按 image_urls / image_paths 数组顺序编号（第1张图、第2张图……），prompt 中写清每张图取什么元素。
- image_search 返回图片后你可以直接看到这些图片，请根据图片内容判断质量和相关性，选最合适的作为参考图。
- 生图结果可能不完美。建议你查看生成结果后，如果不满意，可以反复迭代优化，直到生成结果满意为止。不要只生成一次就结束。
- 最终用 <|box_start|> 和 <|box_end|> 包裹答案，说明生成结果路径或失败原因。"""

USER_PROMPT_BODY = """A conversation between User and Assistant. The user provides an image generation request. The assistant searches for information and reference images, then generates the image by calling tools.

<tools>
{{
  "name": "text_search",
  "description": "批量网页搜索工具。输入 query 数组，返回标题、URL、摘要等结构化信息。",
  "parameters": {{"type": "object", "properties": {{"query": {{"type": "array", "items": {{"type": "string"}}}}}}, "required": ["query"]}}
}}
{{
  "name": "image_search",
  "description": "批量图片搜索工具。输入 query 数组，返回图片标题、URL、来源页、宽高等结构化信息。",
  "parameters": {{"type": "object", "properties": {{"query": {{"type": "array", "items": {{"type": "string"}}}}}}, "required": ["query"]}}
}}
</tools>

工作流程：思考 → 调用工具 → 等待返回 → 继续思考或给出答案。可多轮调用。
每轮只输出一个 tool_call 或最终 box 答案。

User: {question}"""

# 注意:模板里有 {question},是 inference.py 用 .format(question=...) 注入的。
# 上面 USER_PROMPT_BODY 里的 JSON 大括号都用 {{ }} 转义了,不会和 {question} 冲突。
template = (
    "<|extra_203|>" + SYSTEM_PROMPT + " USER: " + USER_PROMPT_BODY + " ASSISTANT: <|image start|>"
)

unc_prompt = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|image start|>"


# 关键改动 4:采样参数,纯文本只用 text_* 那一套
sampling_params = dict(
    use_cache=True,
    text_top_k=50,
    text_top_p=0.9,
    text_temperature=0.7,

    # 占位,反正不会有图像 token
    image_top_k=1,
    image_top_p=1.0,
    image_temperature=1.0,

    top_k=131072,
    top_p=1.0,
    temperature=1.0,
    num_beams_per_group=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    max_new_tokens=max_new_tokens,
    guidance_scale=1.0,

    use_differential_sampling=True,
)
sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
sampling_params["num_beams"] = sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]


special_tokens = dict(
    BOS="<|extra_203|>",
    EOS="<|extra_204|>",
    PAD="<|endoftext|>",
    EOL="<|extra_200|>",
    EOF="<|extra_201|>",
    TMS="<|extra_202|>",
    IMG="<|image token|>",
    BOI="<|image start|>",
    EOI="<|image end|>",
    BSS="<|extra_100|>",
    ESS="<|extra_101|>",
    BOG="<|extra_60|>",
    EOG="<|extra_61|>",
    BOC="<|extra_50|>",
    EOC="<|extra_51|>",
)

seed = 6666

# 测试 prompt
prompts = [
    {
        "prompt": """给我画中国人民大学赵鑫教授看着电脑，嘴里说着：让我看看有没有学生在这里实习不发论文？

ASSISTANT: <thought>
首先，需要查找关于中国人民大学赵鑫教授的详细信息和相关图片，以确保生成的图像准确反映其外貌和场景。这包括搜索文字描述和参考图片，然后根据这些信息生成最终图像。
</thought>

<tool_call>
{
  "name": "text_search",
  "parameters": {
    "query": [
      "中国人民大学赵鑫教授",
      "赵鑫教授的外貌描述",
      "中国人民大学赵鑫教授的办公室场景"
    ]
  }
}</tool_call>

USER: <tool_response>
搜索结果：找到1张参考图片...
参考图1：<|IMAGE|>
</tool_response>

ASSISTANT:""",
        "reference_image": "/opt/tiger/alpha-seed/Emu3.5/pic/wayne.png",
    }
]