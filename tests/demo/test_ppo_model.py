from __future__ import annotations

import torch
import pytest

from warehouse_ai.demo.ppo.model import Mlp, PpoModel


def _module() -> PpoModel:
    return PpoModel(
        observation_size=3,
        directive_count=4,
        hidden_sizes=(64,),
        learning_rate=3e-4,
        clip_range=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        normalize_advantage=True,
    )


def test_model_masks_exact_probabilities_and_exposes_latents() -> None:
    module = _module()
    observations = torch.zeros((2, 3))
    masks = torch.tensor([[True, False, True, False], [False, True, False, False]])
    output = module(observations, masks)
    probabilities = module.probabilities(observations, masks)

    assert output.policy_latent.shape == (2, 64)
    assert output.value_latent.shape == (2, 64)
    assert output.values.shape == (2,)
    assert torch.equal(probabilities[~masks], torch.zeros_like(probabilities[~masks]))
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))


def test_mlp_uses_module_list_and_exposes_last_hidden_layer() -> None:
    module = Mlp(input_size=3, output_size=2, hidden_sizes=(5, 4))
    output, latent = module.forward_with_latent(torch.ones((2, 3)))

    assert isinstance(module.layers, torch.nn.ModuleList)
    assert [(layer.in_features, layer.out_features) for layer in module.layers] == [
        (3, 5),
        (5, 4),
        (4, 2),
    ]
    assert output.shape == (2, 2)
    assert latent.shape == (2, 4)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_mlp_rejects_empty_or_non_positive_dimensions() -> None:
    for dimensions in ((0, 2, (4,)), (3, 0, (4,)), (3, 2, ()), (3, 2, (0,))):
        with pytest.raises(ValueError, match="dimensions and hidden sizes"):
            Mlp(*dimensions)


def test_deterministic_ties_choose_lowest_id_and_sampling_respects_mask() -> None:
    module = _module()
    with torch.no_grad():
        last_layer = module.policy_network.layers[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        last_layer.weight.zero_()
        last_layer.bias.zero_()
    observations = torch.zeros((2, 3))
    masks = torch.tensor([[False, True, True, False], [True, False, False, True]])
    deterministic = module.act(observations, masks, deterministic=True)
    assert deterministic.actions.tolist() == [1, 0]

    generator = torch.Generator().manual_seed(123)
    for _ in range(20):
        sampled = module.act(
            observations, masks, deterministic=False, generator=generator
        )
        assert torch.all(masks.gather(1, sampled.actions[:, None]))
