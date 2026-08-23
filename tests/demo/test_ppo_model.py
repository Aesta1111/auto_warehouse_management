from __future__ import annotations

import torch

from warehouse_ai.demo.ppo.model import PpoModel


def _module() -> PpoModel:
    return PpoModel(
        observation_size=3,
        directive_count=4,
        hidden_size=64,
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


def test_deterministic_ties_choose_lowest_id_and_sampling_respects_mask() -> None:
    module = _module()
    with torch.no_grad():
        module.policy_head.weight.zero_()
        module.policy_head.bias.zero_()
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
