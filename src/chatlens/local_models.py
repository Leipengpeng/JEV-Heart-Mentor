"""Local-only loaders for the two downloaded models; no cloud fallback."""

import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from .config import MODELS_DIR

ROOT = MODELS_DIR


def load_decider(device="mps"):
    import torch
    from typical_ai.core import Typical, _Head
    from typical_ai.backbone import Backbone, pick_device
    from typical_ai.native import NativeHead

    checkpoint = torch.load(ROOT / "typical-small-1.7b/best.pt", map_location="cpu", weights_only=True)
    args = checkpoint["args"]
    if args["readout"] != "native":
        raise ValueError("Expected the downloaded native-readout checkpoint")
    dev = pick_device(device)
    # Reproduce the publisher's loader, replacing both Hub lookups with local files.
    backbone = Backbone(
        str(ROOT / "qwen3-1.7b-base"),
        lora_layers=args["lora_layers"],
        lora_r=args["lora_r"],
        device=dev,
        tap_layer=args["tap_layer"],
    )
    backbone.load_lora_state_dict(checkpoint["lora"])
    head = NativeHead(
        backbone.d,
        nc_head=args["nc_head"],
        null=args["null"],
        render=args["nc_render"],
        score_head=args["score_head"],
        noul_head=args["noul_head"],
    ).to(dev)
    head.load_state_dict(checkpoint["tower"], strict=True)
    head.eval()
    return Typical(_Head(backbone), head, dev, max_state=args["max_state"], max_states=2)


def load_llm():
    from mlx_lm import load

    return load(str(ROOT / "minicpm5-2b-mlx"), trust_remote_code=False)


def generate_reply(
    model,
    tokenizer,
    user_prompt,
    max_tokens=512,
    thinking_budget=None,
    temperature=0.7,
    json_schema=None,
    thinking_temperature=0.3,
    repetition_penalty=None,
    thinking_top_p=0.9,
    max_prompt_tokens=None,
):
    import time
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors

    mx.random.seed(42)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    initial_input_tokens = len(tokenizer.encode(prompt))
    if max_prompt_tokens is not None and initial_input_tokens > max_prompt_tokens:
        raise ValueError("完整模型输入超过token预算")
    processors = make_logits_processors(repetition_penalty=repetition_penalty, repetition_context_size=256)
    format_stats = None
    if json_schema is not None:
        from .structured_output import make_json_processor

        processor, format_stats = make_json_processor(tokenizer, json_schema)
        processors.append(processor)
    pieces = []
    last = None
    started = time.perf_counter()
    thinking_tokens = 0
    thinking_forced = False
    thinking_seconds = 0.0
    if thinking_budget is not None:
        # MiniCPM5 2B is trained in thinking mode. Bound that stage, then continue
        # the same assistant turn after its closing tag. This is an engineering
        # budget, not a claim that the published checkpoint supports no-think mode.
        thoughts = []
        stream = stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=thinking_budget,
            logits_processors=make_logits_processors(
                repetition_penalty=repetition_penalty, repetition_context_size=256
            ),
            sampler=make_sampler(temp=thinking_temperature, top_p=thinking_top_p),
        )
        for thought in stream:
            thoughts.append(thought.text)
            thinking_tokens = thought.generation_tokens
            if "</think>" in "".join(thoughts):
                break
        stream.close()
        thought_text = "".join(thoughts)
        thinking_forced = "</think>" not in thought_text
        thought_text = thought_text.split("</think>")[0]
        prompt = prompt + thought_text + "\n</think>\n\n"
        thinking_seconds = time.perf_counter() - started
    final_input_tokens = len(tokenizer.encode(prompt))
    if max_prompt_tokens is not None and final_input_tokens > max_prompt_tokens:
        raise ValueError("含思考的完整模型输入超过token预算")
    for last in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        logits_processors=processors,
        sampler=make_sampler(temp=temperature, top_p=0.95),
    ):
        pieces.append(last.text)
    if last is None:
        raise RuntimeError("MiniCPM produced no tokens")
    text = "".join(pieces)
    if thinking_budget is not None:
        text = "</think>\n" + text
    return {
        "text": text,
        "final_answer": text.split("</think>")[-1].strip(),
        "elapsed_seconds": time.perf_counter() - started,
        "prompt_tokens": last.prompt_tokens,
        "initial_input_tokens": initial_input_tokens,
        "final_input_tokens": final_input_tokens,
        "generation_tokens": last.generation_tokens,
        "thinking_tokens": thinking_tokens,
        "thinking_budget_forced": thinking_forced,
        "thinking_seconds": thinking_seconds,
        "prompt_tokens_per_second": last.prompt_tps,
        "generation_tokens_per_second": last.generation_tps,
        "mlx_peak_memory_gb": last.peak_memory,
        "schema_enforced": json_schema is not None,
        "format_stats": format_stats,
        "finish_reason": last.finish_reason,
    }
