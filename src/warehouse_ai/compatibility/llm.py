"""Bounded, model-agnostic compatibility helpers for the LLM prototype."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


LLM_BASE_MODEL_ID = "Qwen/Qwen3.5-2B"
LLM_GGUF_REPOSITORY = "bartowski/Qwen_Qwen3.5-2B-GGUF"
LLM_GGUF_REVISION = "7d26695454df6de5fbcce2e58681e62dae06ce43"
LLM_GGUF_FILENAME = "Qwen_Qwen3.5-2B-Q8_0.gguf"
LLM_GGUF_SHA256 = "be647507ce6cde229b838924d47bfff9763171105563f7f908670dae57c4dbe2"


class SoftTokenProjector(nn.Module):
    """Map one controller latent vector to a fixed sequence of embeddings."""

    def __init__(self, latent_size: int, hidden_size: int, soft_tokens: int) -> None:
        super().__init__()
        if latent_size <= 0 or hidden_size <= 0 or soft_tokens <= 0:
            raise ValueError("projector dimensions must be positive")
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


class _TinyFrozenLanguageModel(nn.Module):
    """A tiny local differentiable stand-in, not a claim about Qwen internals."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, inputs_embeds: torch.Tensor, labels: torch.Tensor) -> Any:
        logits = self.output(inputs_embeds)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )
        return SimpleNamespace(loss=loss)


def run_tiny_llm_probe(seed: int = 0) -> dict[str, Any]:
    """Check projector gradient flow beside a frozen toy causal-LM boundary.

    The production prototype uses a frozen Q8 GGUF through llama.cpp. This test
    deliberately does not infer or promise Transformers, LoRA, or soft-token
    support for the Qwen checkpoint.
    """

    torch.manual_seed(seed)
    vocab_size = 64
    hidden_size = 32
    soft_token_count = 4
    base = _TinyFrozenLanguageModel(vocab_size, hidden_size)
    for parameter in base.parameters():
        parameter.requires_grad = False
    projector = SoftTokenProjector(16, hidden_size, soft_token_count)
    text_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    latent = torch.randn((1, 16))
    text_embeddings = base.get_input_embeddings()(text_ids)
    inputs_embeds = torch.cat([projector(latent), text_embeddings], dim=1)
    labels = torch.cat([text_ids, text_ids], dim=1)
    before = {
        name: parameter.detach().clone()
        for name, parameter in projector.named_parameters()
    }
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    output = base(inputs_embeds=inputs_embeds, labels=labels)
    output.loss.backward()
    gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in projector.parameters()
    )
    projector_grad_norm = float(
        torch.sqrt(
            sum(
                parameter.grad.detach().pow(2).sum()
                for parameter in projector.parameters()
                if parameter.grad is not None
            )
        )
    )
    optimizer.step()
    current = dict(projector.named_parameters())
    projector_updated = any(
        not torch.equal(value, current[name].detach()) for name, value in before.items()
    )
    base_frozen = all(not parameter.requires_grad for parameter in base.parameters())
    return {
        "schema_version": 2,
        "seed": seed,
        "torch": torch.__version__,
        "base_model_id": LLM_BASE_MODEL_ID,
        "gguf_repository": LLM_GGUF_REPOSITORY,
        "gguf_filename": LLM_GGUF_FILENAME,
        "soft_tokens": soft_token_count,
        "latent_size": 16,
        "hidden_size": hidden_size,
        "loss": float(output.loss.detach()),
        "loss_finite": bool(torch.isfinite(output.loss)),
        "base_frozen": base_frozen,
        "gradients_finite": gradients_finite,
        "projector_grad_norm": projector_grad_norm,
        "projector_updated": projector_updated,
        "qwen_transformers_lora_verified": False,
    }
