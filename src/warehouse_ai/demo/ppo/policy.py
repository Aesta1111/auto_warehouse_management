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
    WarehouseContext,
    PpoPolicyOutput,
    DirectiveCatalog,
)
from warehouse_ai.demo.contracts.ppo import PpoManifest
from warehouse_ai.demo.directives import build_registry
from warehouse_ai.demo.ppo.model import (
    MODEL_IMPLEMENTATION_VERSION,
    PpoModel,
    architecture_digest,
    implementation_digest,
    legacy_architecture_digest,
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


def _positive_ints(values: dict[str, Any], name: str) -> tuple[int, ...]:
    value = values.get(name)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"manifest hyperparameter {name!r} must be non-empty")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(
                f"manifest hyperparameter {name!r} must contain positive integers"
            )
        result.append(item)
    return tuple(result)


def _migrate_legacy_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Translate the frozen one-hidden-layer key layout to Mlp keys."""
    mapping = {
        "policy_path.0.weight": "policy_network.layers.0.weight",
        "policy_path.0.bias": "policy_network.layers.0.bias",
        "policy_head.weight": "policy_network.layers.1.weight",
        "policy_head.bias": "policy_network.layers.1.bias",
        "value_path.0.weight": "value_network.layers.0.weight",
        "value_path.0.bias": "value_network.layers.0.bias",
        "value_head.weight": "value_network.layers.1.weight",
        "value_head.bias": "value_network.layers.1.bias",
    }
    present = set(mapping).intersection(state_dict)
    if not present:
        return state_dict
    if present != set(mapping):
        raise ValueError("legacy PPO checkpoint has an incomplete state dictionary")
    if set(mapping.values()).intersection(state_dict):
        raise ValueError("PPO checkpoint mixes legacy and MLP state keys")
    return {mapping.get(name, name): value for name, value in state_dict.items()}


def _validate_mask(
    directive_mask: NDArray[np.bool_], catalog: DirectiveCatalog
) -> NDArray[np.bool_]:
    mask = np.asarray(directive_mask, dtype=np.bool_).copy()
    if mask.shape != (catalog.size,):
        raise ValueError("directive mask shape does not match catalog")
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

    def __init__(self, model: PpoModel) -> None:
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


class PpoPolicy:
    """Backend-neutral deterministic facade used by controllers."""

    def __init__(
        self,
        adapter: PolicyAdapter,
        catalog: DirectiveCatalog,
        checkpoint_sha256: str,
    ) -> None:
        if len(checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a SHA-256 digest")
        self.adapter = adapter
        self.backend = adapter.backend
        self.catalog = catalog
        self.checkpoint_sha256 = checkpoint_sha256.lower()

    @classmethod
    def from_module(
        cls,
        module: PpoModel,
        catalog: DirectiveCatalog,
        checkpoint_sha256: str,
    ) -> PpoPolicy:
        if module.directive_count != catalog.size:
            raise ValueError("PPO directive count does not match catalog")
        return cls(TorchPolicyAdapter(module), catalog, checkpoint_sha256)

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        catalog: DirectiveCatalog,
        manifest: PpoManifest,
    ) -> PpoPolicy:
        actual_digest = sha256_file(checkpoint)
        if actual_digest != manifest.checkpoint_sha256:
            raise ValueError("PPO checkpoint digest does not match its manifest")
        if catalog.digest() != manifest.vocabulary_sha256:
            raise ValueError("current catalog digest does not match manifest")
        registry = build_registry(catalog)
        if registry.behavior_digest() != manifest.behavior_sha256:
            raise ValueError("current directive semantics do not match manifest")
        if (
            manifest.schema_version >= 3
            and registry.engine_version != manifest.behavior_engine_version
        ):
            raise ValueError("current directive engine version does not match manifest")

        if manifest.schema_version == 1:
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(str(checkpoint), device="cpu")
            if int(getattr(model.action_space, "n", -1)) != catalog.size:
                raise ValueError("SB3 action space does not match catalog")
            return cls(Sb3PolicyAdapter(model), catalog, actual_digest)

        if manifest.backend != "lightning-fabric":
            raise ValueError("repository-owned checkpoint has the wrong backend")
        assert manifest.model_implementation_version is not None
        assert manifest.gae_version is not None
        assert manifest.loss_version is not None
        expected_implementation = implementation_digest(
            manifest.model_implementation_version,
            manifest.gae_version,
            manifest.loss_version,
        )
        if manifest.implementation_sha256 != expected_implementation:
            raise ValueError("PPO implementation digest does not match manifest")
        values = manifest.hyperparameters
        observation_size = int(_number(values, "observation_size"))
        directive_count = int(_number(values, "directive_count"))
        if manifest.schema_version >= 4:
            if manifest.model_implementation_version != MODEL_IMPLEMENTATION_VERSION:
                raise ValueError("schema-v4+ PPO model implementation is unsupported")
            hidden_sizes = _positive_ints(values, "hidden_sizes")
            expected_architecture = architecture_digest(
                observation_size, directive_count, hidden_sizes
            )
        else:
            hidden_size = int(_number(values, "hidden_size"))
            hidden_sizes = (hidden_size,)
            expected_architecture = legacy_architecture_digest(
                observation_size,
                directive_count,
                hidden_size,
                manifest.model_implementation_version,
            )
        if manifest.architecture_sha256 != expected_architecture:
            raise ValueError("PPO architecture digest does not match manifest")
        module = PpoModel(
            observation_size=observation_size,
            directive_count=directive_count,
            hidden_sizes=hidden_sizes,
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
        module.load_state_dict(_migrate_legacy_state_dict(state_dict), strict=True)
        module.eval()
        return cls.from_module(module, catalog, actual_digest)

    def predict(
        self,
        context: WarehouseContext,
        directive_mask: NDArray[np.bool_],
    ) -> PpoPolicyOutput:
        mask = _validate_mask(directive_mask, self.catalog)
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
        return PpoPolicyOutput(
            state_id=context.state_id,
            episode_step=context.observation.step,
            directive=self.catalog.by_id(selected_id),
            probabilities=tuple(float(value) for value in probabilities),
            value_estimate=value,
            policy_latency_ms=(time.perf_counter() - start) * 1000.0,
            checkpoint_sha256=self.checkpoint_sha256,
        )
