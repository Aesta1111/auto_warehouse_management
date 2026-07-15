"""Local Gemma 4 soft-token and LoRA feasibility probe."""

from __future__ import annotations

from typing import Any, cast

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

GEMMA_PRIMARY_ID = "google/gemma-4-12B-it"
GEMMA_PRIMARY_REVISION = "0e2b1058541244490925fbacf8972041435691ac"
GEMMA_FALLBACK_ID = "google/gemma-4-E4B-it"
GEMMA_FALLBACK_REVISION = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"


class SoftTokenProjector(nn.Module):
    """Map one PPO latent vector to a fixed sequence of model embeddings."""

    def __init__(self, latent_size: int, hidden_size: int, soft_tokens: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.soft_tokens = soft_tokens
        self.network = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * soft_tokens),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        projected = self.network(latent)
        return projected.reshape(-1, self.soft_tokens, self.hidden_size)


def tiny_gemma_config() -> Gemma4TextConfig:
    """Return a randomly initialized CPU-scale Gemma 4 text configuration."""
    return Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=2,
        max_position_embeddings=128,
        sliding_window=32,
        layer_types=["sliding_attention", "full_attention"],
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8,
        use_cache=False,
    )


def _parameter_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _changed(before: dict[str, torch.Tensor], module: nn.Module) -> bool:
    current = dict(module.named_parameters())
    return any(
        not torch.equal(value, current[name].detach()) for name, value in before.items()
    )


def run_tiny_gemma_probe(seed: int = 0) -> dict[str, Any]:
    """Run one optimizer step through projected tokens and rank-8 LoRA."""
    torch.manual_seed(seed)
    config = tiny_gemma_config()
    base = Gemma4ForCausalLM(config)
    soft_token_count = 4
    text_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    identity_ids = torch.cat(
        [torch.zeros((1, soft_token_count), dtype=torch.long), text_ids], dim=1
    )

    # Gemma 4 uses extra token-identity embeddings at every layer. Arbitrary
    # inputs_embeds therefore require explicit per_layer_inputs. Soft positions
    # receive zeros so they do not masquerade as a vocabulary token.
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
    projector = SoftTokenProjector(
        latent_size=16,
        hidden_size=config.hidden_size,
        soft_tokens=soft_token_count,
    )

    base_frozen = all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    )
    projector_before = _parameter_snapshot(projector)
    lora_before = _parameter_snapshot(model)
    optimizer = torch.optim.AdamW(
        [*projector.parameters(), *(p for p in model.parameters() if p.requires_grad)],
        lr=1e-3,
    )

    latent = torch.randn((1, 16))
    soft_tokens = projector(latent)
    embedding_layer = cast(nn.Module, cast(Any, model).get_input_embeddings())
    text_embeddings = embedding_layer(text_ids)
    inputs_embeds = torch.cat([soft_tokens, text_embeddings], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)
    labels = torch.cat(
        [
            torch.full((1, soft_token_count), -100, dtype=torch.long),
            text_ids,
        ],
        dim=1,
    )
    output = model(
        inputs_embeds=inputs_embeds,
        per_layer_inputs=per_layer_inputs,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = output.loss
    if loss is None:
        raise RuntimeError("Gemma probe did not return a loss")
    loss.backward()

    projector_grad_norm = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in projector.parameters()
        if parameter.grad is not None
    )
    lora_grad_norm = sum(
        float(parameter.grad.detach().abs().sum())
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.grad is not None
    )
    gradients_finite = all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in [*projector.parameters(), *model.parameters()]
        if parameter.requires_grad and parameter.grad is not None
    )
    optimizer.step()

    return {
        "schema_version": 1,
        "seed": seed,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "primary_model": {
            "id": GEMMA_PRIMARY_ID,
            "revision": GEMMA_PRIMARY_REVISION,
        },
        "fallback_model": {
            "id": GEMMA_FALLBACK_ID,
            "revision": GEMMA_FALLBACK_REVISION,
        },
        "soft_tokens": soft_token_count,
        "latent_size": 16,
        "hidden_size": config.hidden_size,
        "lora_rank": 8,
        "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss)),
        "base_frozen": base_frozen,
        "projector_grad_norm": projector_grad_norm,
        "lora_grad_norm": lora_grad_norm,
        "gradients_finite": gradients_finite,
        "projector_updated": _changed(projector_before, projector),
        "lora_updated": _changed(lora_before, model),
        "per_layer_soft_identity_zeroed": bool(
            torch.count_nonzero(per_layer_inputs[:, :soft_token_count]) == 0
        ),
    }
