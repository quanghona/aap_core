from importlib.metadata import version

from aap_core.policy_trainer import (
    BaseExplorationModule,
    BasePolicyTrainer,
    DPOTrainer,
    EpsilonGreedyExploration,
    GRPOTrainer,
    PPOTrainer,
    RandomExploration,
    ReinforcePPTrainer,
    SimpleReplayBuffer,
)

# Single source of truth is [project] version in pyproject.toml. uv_build does not
# support dynamic = ["version"], so the version is read from the installed
# distribution metadata instead of duplicated in source.
__version__ = version("aap_core")

__all__ = [
    "__version__",
    "BaseExplorationModule",
    "BasePolicyTrainer",
    "DPOTrainer",
    "EpsilonGreedyExploration",
    "GRPOTrainer",
    "PPOTrainer",
    "RandomExploration",
    "ReinforcePPTrainer",
    "SimpleReplayBuffer",
]
