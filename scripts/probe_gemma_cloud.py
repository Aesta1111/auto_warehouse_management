"""Run the authoritative quantized Gemma 4 A40 compatibility probe."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import median
from typing import Any, cast

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from warehouse_ai.compatibility import (
    GEMMA_PRIMARY_ID,
    GEMMA_PRIMARY_REVISION,
    SoftTokenProjector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=GEMMA_PRIMARY_ID)
    parser.add_argument("--revision", default=GEMMA_PRIMARY_REVISION)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def changed(before: dict[str, torch.Tensor], module: nn.Module) -> bool:
    current = dict(module.named_parameters())
    return any(
        not torch.equal(value, current[name].detach()) for name, value in before.items()
    )


def train_step(
    model: Any,
    projector: SoftTokenProjector,
    optimizer: torch.optim.Optimizer,
    latent: torch.Tensor,
    token_ids: torch.Tensor,
    per_layer_inputs: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    text_embeddings = model.get_input_embeddings()(token_ids)
    soft = projector(latent).to(dtype=text_embeddings.dtype)
    inputs_embeds = torch.cat([soft, text_embeddings], dim=1)
    labels = torch.cat(
        [
            torch.full(
                (token_ids.shape[0], soft.shape[1]),
                -100,
                dtype=torch.long,
                device=token_ids.device,
            ),
            token_ids,
        ],
        dim=1,
    )
    output = model(
        inputs_embeds=inputs_embeds,
        per_layer_inputs=per_layer_inputs,
        attention_mask=torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=token_ids.device
        ),
        labels=labels,
        use_cache=False,
    )
    loss = output.loss
    if loss is None or not bool(torch.isfinite(loss)):
        raise RuntimeError("non-finite or missing training loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.inference_mode()
def greedy_decision(
    model: Any,
    projector: SoftTokenProjector,
    latent: torch.Tensor,
    token_ids: torch.Tensor,
    per_layer_inputs: torch.Tensor,
    max_new_tokens: int,
) -> list[int]:
    text_embeddings = model.get_input_embeddings()(token_ids)
    soft = projector(latent).to(dtype=text_embeddings.dtype)
    inputs_embeds = torch.cat([soft, text_embeddings], dim=1)
    attention_mask = torch.ones(
        inputs_embeds.shape[:2], dtype=torch.long, device=token_ids.device
    )
    output = model(
        inputs_embeds=inputs_embeds,
        per_layer_inputs=per_layer_inputs,
        attention_mask=attention_mask,
        use_cache=True,
    )
    generated: list[int] = []
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    past = output.past_key_values
    for _ in range(max_new_tokens):
        generated.append(int(next_token.item()))
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=token_ids.device),
            ],
            dim=1,
        )
        output = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    return generated


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this authoritative probe requires a CUDA GPU")
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base = prepare_model_for_kbit_training(base)
    device = base.get_input_embeddings().weight.device
    prompt = '{"warehouse":"tiny","eligible":[1,2,3],"assignments":'
    token_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    soft_token_count = 4
    identity_ids = torch.cat(
        [
            torch.zeros((1, soft_token_count), dtype=torch.long, device=device),
            token_ids,
        ],
        dim=1,
    )
    per_layer_inputs = base.model.get_per_layer_inputs(identity_ids, None).detach()
    per_layer_inputs[:, :soft_token_count] = 0
    model = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model_config = cast(Any, model.config)
    projector = SoftTokenProjector(
        16, int(model_config.hidden_size), soft_token_count
    ).to(device)
    latent = torch.randn((1, 16), device=device)
    trainable = [
        *projector.parameters(),
        *(p for p in model.parameters() if p.requires_grad),
    ]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    projector_before = {
        name: parameter.detach().clone()
        for name, parameter in projector.named_parameters()
    }
    lora_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    base_frozen = all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    )

    for _ in range(args.warmup_steps):
        train_step(model, projector, optimizer, latent, token_ids, per_layer_inputs)
    torch.cuda.synchronize()
    losses: list[float] = []
    step_latencies: list[float] = []
    for _ in range(args.measured_steps):
        step_started = time.perf_counter()
        losses.append(
            train_step(model, projector, optimizer, latent, token_ids, per_layer_inputs)
        )
        torch.cuda.synchronize()
        step_latencies.append(time.perf_counter() - step_started)

    model.eval()
    decision_latencies: list[float] = []
    valid_json = 0
    for _ in range(args.decisions):
        decision_started = time.perf_counter()
        generated = greedy_decision(
            model,
            projector,
            latent,
            token_ids,
            per_layer_inputs,
            args.max_new_tokens,
        )
        torch.cuda.synchronize()
        decision_latencies.append(time.perf_counter() - decision_started)
        try:
            json.loads(tokenizer.decode(generated, skip_special_tokens=True))
            valid_json += 1
        except json.JSONDecodeError:
            pass

    summary: dict[str, Any] = {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "quantization": "NF4 double-quant, BF16 compute",
        "gpu": torch.cuda.get_device_name(),
        "base_frozen": base_frozen,
        "projector_updated": changed(projector_before, projector),
        "lora_updated": changed(lora_before, model),
        "losses_finite": all(math.isfinite(loss) for loss in losses),
        "final_loss": losses[-1],
        "training_step_median_seconds": median(step_latencies),
        "training_step_p95_seconds": percentile(step_latencies, 0.95),
        "decision_median_seconds": median(decision_latencies),
        "decision_p95_seconds": percentile(decision_latencies, 0.95),
        "decision_json_valid": valid_json,
        "decisions": args.decisions,
        "generated_tokens_per_decision": args.max_new_tokens,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
