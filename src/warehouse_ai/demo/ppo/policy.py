"""Deterministic directive policy with lazy backend adapters."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from warehouse_ai.demo.artifacts import sha256_file
from warehouse_ai.demo.contracts.directives import (
    DirectiveContext,
    DirectivePrediction,
    DirectiveVocabulary,
)
from warehouse_ai.demo.contracts.ppo import DirectivePpoManifest
from warehouse_ai.demo.directives import build_directive_registry
from warehouse_ai.demo.ppo.model import (
    DirectivePpoModel,
    architecture_digest,
    implementation_digest,
)


def _number(values: dict[str, Any], name: str) -> float:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"manifest hyperparameter {name!r} must be numeric")
    return float(value)


def _boolean(values: dict[str, Any], name: str) -> bool:
    value = values.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"manifest hyperparameter {name!r} must be boolean")
    return value


def _validate_mask(
    directive_mask: NDArray[np.bool_], vocabulary: DirectiveVocabulary
) -> NDArray[np.bool_]:
    mask = np.asarray(directive_mask, dtype=np.bool_).copy()
    if mask.shape != (vocabulary.size,):
        raise ValueError("directive mask shape does not match vocabulary")
    if not bool(np.any(mask)):
        raise ValueError("directive mask must allow at least one directive")
    return mask


class PolicyAdapter(Protocol):
    backend: str

    def predict(
        self, observation: NDArray[np.float32], mask: NDArray[np.bool_]
    ) -> tuple[NDArray[np.float64], float]: ...


class TorchPolicyAdapter:
    backend = "lightning-fabric"

    def __init__(self, model: DirectivePpoModel) -> None:
        self.model = model

    def predict(
        self, observation: NDArray[np.float32], mask: NDArray[np.bool_]
    ) -> tuple[NDArray[np.float64], float]:
        device = next(self.model.parameters()).device
        observations = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        masks = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.inference_mode():
            output = self.model(observations, masks)
            probabilities = torch.softmax(output.policy_logits, dim=-1)[0]
        return (
            probabilities.detach().cpu().numpy().astype(np.float64),
            float(output.values.item()),
        )


class Sb3PolicyAdapter:
    """Preserved schema-v1 reference without an eager SB3 import."""

    backend = "sb3-maskable-ppo"

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(
        self, observation: NDArray[np.float32], mask: NDArray[np.bool_]
    ) -> tuple[NDArray[np.float64], float]:
        observation_tensor, _ = self.model.policy.obs_to_tensor(observation)
        distribution = self.model.policy.get_distribution(
            observation_tensor, action_masks=mask
        )
        probabilities = getattr(distribution.distribution, "probs", None)
        if probabilities is None:
            raise RuntimeError("SB3 policy returned no categorical probabilities")
        values = self.model.policy.predict_values(observation_tensor)
        return (
            probabilities[0].detach().cpu().numpy().astype(np.float64),
            float(values.reshape(-1)[0].item()),
        )


class DirectivePpoPolicy:
    """Backend-neutral deterministic facade used by controllers."""

    def __init__(
        self,
        adapter: PolicyAdapter,
        vocabulary: DirectiveVocabulary,
        checkpoint_sha256: str,
    ) -> None:
        if len(checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a SHA-256 digest")
        self.adapter = adapter
        self.backend = adapter.backend
        self.vocabulary = vocabulary
        self.checkpoint_sha256 = checkpoint_sha256.lower()

    @classmethod
    def from_module(
        cls,
        module: DirectivePpoModel,
        vocabulary: DirectiveVocabulary,
        checkpoint_sha256: str,
    ) -> DirectivePpoPolicy:
        if module.directive_count != vocabulary.size:
            raise ValueError("PPO directive count does not match vocabulary")
        return cls(TorchPolicyAdapter(module), vocabulary, checkpoint_sha256)

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        vocabulary: DirectiveVocabulary,
        manifest: DirectivePpoManifest,
    ) -> DirectivePpoPolicy:
        actual_digest = sha256_file(checkpoint)
        if actual_digest != manifest.checkpoint_sha256:
            raise ValueError("PPO checkpoint digest does not match its manifest")
        if vocabulary.digest() != manifest.vocabulary_sha256:
            raise ValueError("current vocabulary digest does not match manifest")
        registry = build_directive_registry(vocabulary)
        if registry.behavior_digest() != manifest.behavior_sha256:
            raise ValueError("current directive rules do not match manifest")
        if (
            manifest.schema_version == 3
            and registry.engine_version != manifest.behavior_engine_version
        ):
            raise ValueError("current directive engine version does not match manifest")

        if manifest.schema_version == 1:
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(str(checkpoint), device="cpu")
            if int(getattr(model.action_space, "n", -1)) != vocabulary.size:
                raise ValueError("SB3 action space does not match vocabulary")
            return cls(Sb3PolicyAdapter(model), vocabulary, actual_digest)

        if manifest.backend != "lightning-fabric":
            raise ValueError("repository-owned checkpoint has the wrong backend")
        if manifest.implementation_sha256 != implementation_digest():
            raise ValueError("PPO implementation digest does not match manifest")
        values = manifest.hyperparameters
        dimensions = (
            int(_number(values, "observation_size")),
            int(_number(values, "directive_count")),
            int(_number(values, "hidden_size")),
        )
        if manifest.architecture_sha256 != architecture_digest(*dimensions):
            raise ValueError("PPO architecture digest does not match manifest")
        module = DirectivePpoModel(
            observation_size=dimensions[0],
            directive_count=dimensions[1],
            hidden_size=dimensions[2],
            learning_rate=_number(values, "learning_rate"),
            clip_range=_number(values, "clip_range"),
            entropy_coefficient=_number(values, "entropy_coefficient"),
            value_coefficient=_number(values, "value_coefficient"),
            normalize_advantage=_boolean(values, "normalize_advantage"),
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(state_dict, dict):
            raise ValueError("PPO checkpoint has no model state dictionary")
        module.load_state_dict(state_dict, strict=True)
        module.eval()
        return cls.from_module(module, vocabulary, actual_digest)

    def predict(
        self,
        context: DirectiveContext,
        directive_mask: NDArray[np.bool_],
    ) -> DirectivePrediction:
        mask = _validate_mask(directive_mask, self.vocabulary)
        start = time.perf_counter()
        probabilities, value = self.adapter.predict(context.observation.encode(), mask)
        probabilities = np.where(mask, probabilities, 0.0)
        total = float(probabilities.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("masked directive probabilities have no positive mass")
        probabilities /= total
        selected_id = int(np.argmax(probabilities))
        if not bool(mask[selected_id]):
            raise RuntimeError("masked PPO selected an inapplicable directive")
        return DirectivePrediction(
            state_id=context.state_id,
            episode_step=context.observation.step,
            directive=self.vocabulary.by_id(selected_id),
            probabilities=tuple(float(value) for value in probabilities),
            value_estimate=value,
            policy_latency_ms=(time.perf_counter() - start) * 1000.0,
            checkpoint_sha256=self.checkpoint_sha256,
        )
