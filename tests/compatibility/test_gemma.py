from __future__ import annotations

import torch

from warehouse_ai.compatibility import SoftTokenProjector, run_tiny_gemma_probe


def test_soft_token_projector_shape_and_gradient_flow() -> None:
    projector = SoftTokenProjector(latent_size=8, hidden_size=16, soft_tokens=4)
    projected = torch.ones((2, 8))

    output = projector(projected)
    output.sum().backward()

    assert output.shape == (2, 4, 16)
    assert all(parameter.grad is not None for parameter in projector.parameters())


def test_tiny_gemma_soft_tokens_and_lora_update() -> None:
    summary = run_tiny_gemma_probe(seed=0)

    assert summary["loss_finite"]
    assert summary["base_frozen"]
    assert summary["gradients_finite"]
    assert summary["projector_grad_norm"] > 0
    assert summary["lora_grad_norm"] > 0
    assert summary["projector_updated"]
    assert summary["lora_updated"]
    assert summary["per_layer_soft_identity_zeroed"]
