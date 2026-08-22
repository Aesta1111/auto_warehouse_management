"""Deprecated PPO training imports; prefer ``demo.ppo.training``."""

from warehouse_ai.demo.ppo.collector import DirectiveRolloutCollector
from warehouse_ai.demo.ppo.training import DirectivePpoTrainingLoop

__all__ = ["DirectivePpoTrainingLoop", "DirectiveRolloutCollector"]
