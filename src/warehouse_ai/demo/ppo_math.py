"""Deprecated PPO math imports; prefer ``demo.ppo.math``."""

from warehouse_ai.demo.ppo.math import (
    GAE_VERSION,
    LOSS_VERSION,
    compute_gae,
    compute_ppo_loss,
)

__all__ = ["GAE_VERSION", "LOSS_VERSION", "compute_gae", "compute_ppo_loss"]
