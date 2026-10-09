import glob
import os
import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from aap_core.policy import BasePolicy
from aap_core.prompt_augmenter import PromptOptimizationEnv
from aap_core.types import AgentMessage


@dataclass
class ReplayBufferEntry:
    """A single transition stored in the replay buffer.

    Minimal dataclass containing only the information needed for the
    policy gradient update step.

    Attributes:
        obs: Prompt embedding vector of shape (obs_dim,).
        action: Index of the augmenter applied (0 to num_augmenters-1).
        reward: Reward from the reward model.
        log_prob: Log probability of the action under the policy.
        mask: Validity mask (1.0 = valid, 0.0 = invalid).
    """

    obs: np.ndarray
    action: int
    reward: float
    log_prob: float
    mask: float


class BaseReplayBuffer(ABC):
    """Abstract base class for replay buffers.

    All replay buffer implementations must provide:
    - ``push``: add a new transition
    - ``sample``: draw a batch of transitions
    - ``__len__``: current buffer size
    - ``is_empty``: whether the buffer has any data
    - ``clear``: remove all stored transitions

    Subclasses may also override ``_sample_indices`` for custom sampling
    strategies (e.g., prioritized sampling).
    """

    def __init__(self, capacity: int) -> None:
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions the buffer can hold.
                Must be positive.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._size = 0
        self._head = 0  # write pointer for circular buffer

    @property
    def capacity(self) -> int:
        """Maximum number of transitions the buffer can hold."""
        return self._capacity

    @property
    def size(self) -> int:
        """Current number of transitions in the buffer."""
        return self._size

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached its capacity."""
        return self._size >= self._capacity

    @abstractmethod
    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        mask: float,
    ) -> None:
        """Add a single transition to the buffer.

        Args:
            obs: Prompt embedding vector.
            action: Index of the augmenter applied.
            reward: Reward from the reward model.
            log_prob: Log probability of the action.
            mask: Validity mask.
        """
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys ``'obs'``, ``'action'``, ``'reward'``,
            ``'log_prob'``, ``'mask'``. Each value is a batched array
            of shape (batch_size, ...) or (batch_size,).
        """
        pass

    @abstractmethod
    def _sample_indices(self, batch_size: int) -> List[int]:
        """Sample indices for a batch.

        Override this method in subclasses to implement custom sampling
        strategies (e.g., prioritized sampling).

        Args:
            batch_size: Number of indices to sample.

        Returns:
            List of integer indices into the buffer.
        """
        pass

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return self._size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(capacity={self._capacity}, size={self._size})"
        )

    def is_empty(self) -> bool:
        """Whether the buffer has no stored transitions."""
        return self._size == 0

    def clear(self) -> None:
        """Remove all transitions from the buffer."""
        self._size = 0
        self._head = 0


class SimpleReplayBuffer(BaseReplayBuffer):
    """Uniform random sampling replay buffer with circular storage.

    This is a simple FIFO circular buffer that stores transitions and
    samples uniformly at random. It is suitable for off-policy trainers
    (REINFORCE++, GRPO, DPO) where any past experience is equally valuable.

    Memory-efficient: uses pre-allocated numpy arrays and overwrites
    old entries when the buffer is full.

    Example::

        buffer = SimpleReplayBuffer(capacity=10000)
        for _ in range(5000):
            buffer.push(obs, action, reward, log_prob, mask)
        batch = buffer.sample(batch_size=32)
    """

    def __init__(self, capacity: int) -> None:
        """Initialize the simple replay buffer.

        Args:
            capacity: Maximum number of transitions.
        """
        super().__init__(capacity)
        self._obs: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._log_probs: List[float] = []
        self._masks: List[float] = []

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        mask: float,
    ) -> None:
        """Add a transition to the buffer.

        If the buffer is not full, appends to the end. If full, overwrites
        the oldest entry (circular buffer behavior).

        Args:
            obs: Prompt embedding vector.
            action: Index of the augmenter applied.
            reward: Reward from the reward model.
            log_prob: Log probability of the action.
            mask: Validity mask.
        """
        if self._size < self._capacity:
            self._obs.append(np.asarray(obs, dtype=np.float32))
            self._actions.append(action)
            self._rewards.append(reward)
            self._log_probs.append(log_prob)
            self._masks.append(mask)
            self._size += 1
        else:
            idx = self._head % self._capacity
            self._obs[idx] = np.asarray(obs, dtype=np.float32)
            self._actions[idx] = action
            self._rewards[idx] = reward
            self._log_probs[idx] = log_prob
            self._masks[idx] = mask
        self._head += 1

    def _sample_indices(self, batch_size: int) -> List[int]:
        """Sample uniform random indices.

        Args:
            batch_size: Number of indices to sample.

        Returns:
            List of random indices.
        """
        n = min(batch_size, self._size)
        return random.sample(range(self._size), n)

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of transitions with uniform probability.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys "obs", "action", "reward", "log_prob", "mask",
            each containing a numpy array of sampled values.
        """
        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty buffer")

        indices = self._sample_indices(batch_size)
        return {
            "obs": np.stack([self._obs[i] for i in indices], axis=0),
            "action": np.array([self._actions[i] for i in indices], dtype=np.int64),
            "reward": np.array([self._rewards[i] for i in indices], dtype=np.float32),
            "log_prob": np.array(
                [self._log_probs[i] for i in indices], dtype=np.float32
            ),
            "mask": np.array([self._masks[i] for i in indices], dtype=np.float32),
        }

    def clear(self) -> None:
        """Remove all transitions and free memory."""
        super().clear()
        self._obs.clear()
        self._actions.clear()
        self._rewards.clear()
        self._log_probs.clear()
        self._masks.clear()


class PrioritizedReplayBuffer(BaseReplayBuffer):
    """Prioritized Experience Replay buffer with advantage-weighted sampling.

    Implements the prioritized sampling strategy from Schaul et al. (2016)
    "Prioritized Experience Replay". Transitions with higher TD-error
    (or advantage in policy gradient context) are sampled with higher
    probability, enabling more efficient learning from informative samples.

    Uses a binary heap for O(log N) priority updates and sampling.

    Memory-efficient: stores only the priority tree and transition data.

    Example::

        buffer = PrioritizedReplayBuffer(capacity=10000)
        for _ in range(5000):
            buffer.push(obs, action, reward, log_prob, mask)
        batch = buffer.sample(batch_size=32)

    Reference: https://arxiv.org/abs/1511.05952
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ) -> None:
        """Initialize the prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions.
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized).
                Higher values emphasize high-priority samples more.
            beta: Importance sampling exponent for bias correction.
                Increases from 0 to 1 during training.
            beta_increment: Amount to increase beta per sample call.

        Raises:
            ValueError: If alpha or beta is not in [0, 1].
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0, 1]")

        super().__init__(capacity)
        self._alpha = alpha
        self._beta = beta
        self._beta_increment = beta_increment

        # Priority tree (1-indexed, size 2*capacity)
        self._tree_size = 2 * capacity
        self._tree: List[float] = [0.0] * self._tree_size

        # Data storage
        self._obs: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._log_probs: List[float] = []
        self._masks: List[float] = []
        self._priorities: List[float] = [1e-4] * capacity  # Initial priorities

    @property
    def alpha(self) -> float:
        """Priority exponent controlling sampling bias."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Current beta value for importance sampling weight computation."""
        return self._beta

    def __repr__(self) -> str:
        """String representation of the buffer."""
        return (
            f"PrioritizedReplayBuffer(capacity={self._capacity}, "
            f"size={self._size}, alpha={self._alpha})"
        )

    def _parent(self, idx: int) -> int:
        """Get parent index in the binary heap."""
        return idx // 2

    def _left_child(self, idx: int) -> int:
        """Get left child index in the binary heap."""
        return 2 * idx

    def _right_child(self, idx: int) -> int:
        """Get right child index in the binary heap."""
        return 2 * idx + 1

    def _update(self, idx: int, priority: float) -> None:
        """Update priority at leaf index and propagate up.

        Args:
            idx: Leaf index in the tree (1-indexed).
            priority: New priority value.
        """
        priority = max(priority, 1e-4)  # Minimum priority to avoid zero
        tree_idx = idx + self._capacity  # Map data index to tree leaf
        self._tree[tree_idx] = priority

        # Propagate up to root
        while tree_idx > 1:
            parent_idx = self._parent(tree_idx)
            left_idx = self._left_child(parent_idx)
            right_idx = self._right_child(parent_idx)

            # Parent priority is max of children
            parent_priority = max(
                self._tree[left_idx] if left_idx < self._tree_size else 0.0,
                self._tree[right_idx] if right_idx < self._tree_size else 0.0,
            )
            if self._tree[parent_idx] == parent_priority:
                break
            self._tree[parent_idx] = parent_priority
            tree_idx = parent_idx

    def _get_priority(self, idx: int) -> float:
        """Get priority at data index.

        Args:
            idx: Data index (0-indexed).

        Returns:
            Priority value.
        """
        tree_idx = idx + self._capacity
        return self._tree[tree_idx]

    def _sample(self, batch_size: int) -> Tuple[List[int], np.ndarray]:
        """Sample a batch of indices using prioritized sampling.

        Uses the segment tree approach: divide [0, sum_priorities] into
        batch_size equal-width segments, then sample one point uniformly
        from each segment and find the leaf with the highest priority
        covering that point.

        Args:
            batch_size: Number of indices to sample.

        Returns:
            Tuple of (sampled indices, importance sampling weights).
        """
        n = min(batch_size, self._size)
        if n == 0:
            return [], np.array([], dtype=np.float32)

        # Get total priority (root of tree)
        total_priority = self._tree[1]
        if total_priority <= 0:
            total_priority = 1e-4

        # Divide into segments within [0, total_priority)
        segment_length = total_priority / n
        indices = []
        weights = []

        for _ in range(n):
            # Sample a point uniformly from [low, high) within [0, total_priority)
            low = random.random() * segment_length
            high = low + segment_length
            point = random.uniform(low, min(high, total_priority))

            # Find the leaf covering this point
            tree_idx = 1
            while tree_idx < self._capacity:
                left_idx = self._left_child(tree_idx)
                right_idx = self._right_child(tree_idx)
                if self._tree[left_idx] >= point:
                    tree_idx = left_idx
                else:
                    point -= self._tree[left_idx]
                    tree_idx = right_idx

            # Get data index
            data_idx = tree_idx - self._capacity
            if 0 <= data_idx < self._size:
                indices.append(data_idx)
                # Compute importance sampling weight
                p_i = self._tree[tree_idx] / total_priority
                w_i = (p_i * self._size) ** (-self._beta)
                weights.append(w_i)

        # Normalize weights
        if weights:
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]
            weights = np.array(weights, dtype=np.float32)
        else:
            weights = np.array([], dtype=np.float32)

        return indices, weights

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        mask: float,
        priority: Optional[float] = None,
    ) -> None:
        """Add a transition to the buffer with specified priority.

        If the buffer is not full, appends to the end. If full, overwrites
        the oldest entry (circular buffer behavior).

        Args:
            obs: Prompt embedding vector.
            action: Index of the augmenter applied.
            reward: Reward from the reward model.
            log_prob: Log probability of the action.
            mask: Validity mask.
            priority: Priority value (0-1). If None, uses max priority (1.0).
        """
        if priority is None:
            priority = 1.0  # Max initial priority
        priority = max(priority, 1e-4)  # Clamp to minimum

        if self._size < self._capacity:
            self._obs.append(np.asarray(obs, dtype=np.float32))
            self._actions.append(action)
            self._rewards.append(reward)
            self._log_probs.append(log_prob)
            self._masks.append(mask)
            self._priorities[self._size] = priority
            self._size += 1
            # Update tree with new entry
            self._update(self._size - 1, priority)
        else:
            idx = self._head % self._capacity
            self._obs[idx] = np.asarray(obs, dtype=np.float32)
            self._actions[idx] = action
            self._rewards[idx] = reward
            self._log_probs[idx] = log_prob
            self._masks[idx] = mask
            # Update priority
            self._priorities[idx] = priority
            self._update(idx, priority)
        self._head += 1

    def update_priority(self, idx: int, priority: float) -> None:
        """Update priority of a specific transition.

        Args:
            idx: Data index to update.
            priority: New priority value.
        """
        if 0 <= idx < self._size:
            priority = max(priority, 1e-4)
            self._priorities[idx] = priority
            self._update(idx, priority)

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for specific indices after training.

        This should be called after an update step with the computed
        TD-errors or advantages for the sampled transitions.

        Args:
            indices: List of data indices to update.
            priorities: List of new priority values (absolute TD-errors).
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < self._size:
                self._priorities[idx] = priority
                self._update(idx, priority)

    def get_importance_sampling_weights(self, indices: List[int]) -> np.ndarray:
        """Compute importance sampling weights for sampled indices.

        Weights correct for the bias introduced by non-uniform sampling.
        w = (N * P(i))^-beta / max_weight

        Args:
            indices: List of data indices.

        Returns:
            Array of importance sampling weights.
        """
        total_priority = self._tree[1]
        if total_priority <= 0:
            total_priority = 1e-4

        n = self._size
        weights = []
        for idx in indices:
            if 0 <= idx < n:
                p_i = self._get_priority(idx) / total_priority
                w_i = (p_i * n) ** (-self._beta)
                weights.append(w_i)

        # Normalize by max weight
        if weights:
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]
        return np.array(weights, dtype=np.float32)

    def decay_beta(self) -> None:
        """Increment beta towards 1.0 for more uniform sampling over time."""
        self._beta = min(self._beta + self._beta_increment, 1.0)

    def _sample_indices(self, batch_size: int) -> List[int]:
        """Sample indices using prioritized sampling (legacy method).

        Returns only indices without weights for compatibility with
        the base class interface.

        Args:
            batch_size: Number of indices to sample.

        Returns:
            List of sampled indices.
        """
        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty buffer")

        indices, _ = self._sample(batch_size)
        return indices

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of transitions with priority-weighted probability.

        Higher priority transitions are sampled with higher probability.
        Returns a dictionary with batch data and importance sampling weights
        for bias correction during training.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys "obs", "action", "reward", "log_prob",
            "mask", "weights", and "indices".
        """
        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty buffer")

        # Increment beta for more uniform sampling over time
        self._beta = min(self._beta + self._beta_increment, 1.0)

        indices, weights = self._sample(batch_size)
        return {
            "obs": np.stack([self._obs[i] for i in indices], axis=0),
            "action": np.array([self._actions[i] for i in indices], dtype=np.int64),
            "reward": np.array([self._rewards[i] for i in indices], dtype=np.float32),
            "log_prob": np.array(
                [self._log_probs[i] for i in indices], dtype=np.float32
            ),
            "mask": np.array([self._masks[i] for i in indices], dtype=np.float32),
            "weights": weights,
            "indices": np.array(indices, dtype=np.int64),
        }

    def clear(self) -> None:
        """Remove all transitions and free memory."""
        super().clear()
        self._obs.clear()
        self._actions.clear()
        self._rewards.clear()
        self._log_probs.clear()
        self._masks.clear()
        self._priorities = [1e-4] * self._capacity
        self._tree = [0.0] * self._tree_size


class BaseExplorationModule(ABC):
    """Abstract base class for exploration strategies in policy training.

    This class defines the interface for exploration modules that determine
    whether the policy should explore (sample stochastically) or exploit
    (select greedy actions) at each step during episode rollout.

    The exploration module is used during the ``fit()`` method's environment
    interaction loop, NOT during the gradient update step.

    Example usage with a custom module::

        class MyExploration(BaseExplorationModule):
            def should_explore(self, step: int, episode: int, **kwargs) -> bool:
                return torch.rand(1).item() > 0.5

            def get_exploration_config(self) -> dict:
                return {"strategy": "my_custom"}

    Example usage with torchRL::

        from torchrl.envs import EpsilonGreedyActionSelection

        class TorchRLWrapper(BaseExplorationModule):
            def __init__(self, torchrl_exploration):
                self._exploration = torchrl_exploration

            def should_explore(self, step: int, episode: int, **kwargs) -> bool:
                return self._exploration.step()

            def get_exploration_config(self) -> dict:
                return {"source": "torchrl"}

    """

    @abstractmethod
    def should_explore(self, step: int, episode: int, **kwargs) -> bool:
        """Determine whether to explore at the current step.

        Args:
            step: Current step index within the episode (0-based).
            episode: Current episode index (0-based).
            **kwargs: Additional context (e.g., ``'obs'``, ``'logits'``, ``'info'``).

        Returns:
            ``True`` if the policy should explore (sample stochastically),
            ``False`` if it should exploit (select greedy action).
        """
        pass

    @abstractmethod
    def get_exploration_config(self) -> Dict[str, Any]:
        """Return a dictionary of exploration hyperparameters for logging.

        Returns:
            Dictionary of configuration parameters (e.g., epsilon, temperature).
        """
        pass

    def reset(self, episode: int) -> None:
        """Reset exploration state at the beginning of an episode.

        Override this method if the exploration strategy needs per-episode
        state (e.g., resetting an epsilon schedule).

        Args:
            episode: Current episode index (0-based).
        """
        pass


class EpsilonGreedyExploration(BaseExplorationModule):
    """Epsilon-greedy exploration with linear decay.

    Explores with probability epsilon, exploits with probability 1-epsilon.
    Epsilon decays linearly from ``eps_init`` to ``eps_final`` over
    ``decay_episodes``.

    Args:
        eps_init: Initial exploration probability (default ``1.0``).
        eps_final: Final exploration probability after decay (default ``0.05``).
        decay_episodes: Number of episodes over which to decay epsilon
            (default ``1000``).
        min_eps: Minimum epsilon floor (default ``0.01``).
    """

    def __init__(
        self,
        eps_init: float = 1.0,
        eps_final: float = 0.05,
        decay_episodes: int = 1000,
        min_eps: float = 0.01,
    ) -> None:
        self.eps_init = eps_init
        self.eps_final = max(eps_final, min_eps)
        self.decay_episodes = max(1, decay_episodes)
        self.current_eps: float = eps_init

    def should_explore(self, step: int, episode: int, **kwargs) -> bool:
        # Linear decay
        progress = min(episode / self.decay_episodes, 1.0)
        self.current_eps = self.eps_final + (self.eps_init - self.eps_final) * (
            1.0 - progress
        )
        self.current_eps = max(self.current_eps, self.eps_final)
        return torch.rand(1).item() < self.current_eps

    def get_exploration_config(self) -> Dict[str, Any]:
        return {
            "eps_init": self.eps_init,
            "eps_final": self.eps_final,
            "decay_episodes": self.decay_episodes,
            "current_eps": self.current_eps,
        }


class RandomExploration(EpsilonGreedyExploration):
    """Convenience class for constant explore/exploit ratio (no decay).

    This is a special case of EpsilonGreedyExploration where ``eps_init``
    equals ``eps_final``, resulting in a constant exploration probability
    regardless of the episode number. Useful for fixed exploration ratios.

    Args:
        explore_ratio: Probability of exploring (default ``0.0``, pure exploit).
    """

    def __init__(self, explore_ratio: float = 0.0) -> None:
        # Initialize with no decay: eps_init = eps_final = explore_ratio
        # Pass min_eps=0 to allow true 0% exploration (unlike EpsilonGreedyExploration)
        super().__init__(
            eps_init=explore_ratio,
            eps_final=explore_ratio,
            decay_episodes=1,  # Not used since eps_init == eps_final
            min_eps=0.0,  # Allow ratios from 0.0 to 1.0
        )
        # Store explore_ratio for direct access
        self.explore_ratio = explore_ratio

    def get_exploration_config(self) -> Dict[str, Any]:
        # Override to report explore_ratio for clarity
        return {"explore_ratio": self.explore_ratio}


class BasePolicyTrainer(ABC):
    def __init__(
        self,
        policy_model: BasePolicy,
        env: PromptOptimizationEnv,
        max_episodes: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        exploration_module: Optional[BaseExplorationModule] = None,
        replay_buffer: Optional[BaseReplayBuffer] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.policy_model = policy_model
        self.env = env
        self.max_episodes = max_episodes
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.replay_buffer = replay_buffer  # Placeholder for potential replay buffer
        self.eps = eps

        # Set up exploration module: default to pure exploit
        self.exploration_module = exploration_module or RandomExploration()

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @classmethod
    def _compute_kl_loss(
        cls,
        old_logits: torch.Tensor,
        batch_indices: slice,
        logits: torch.Tensor,
        action_loss: torch.Tensor,
        kl_coef: float,
        kl_loss_epoch: float,
    ) -> Tuple[torch.Tensor, float]:
        """Compute KL divergence loss between old and current policy.

        KL(π_old || π_new) = Σ π_old(a|s) * (log π_old(a|s) - log π_new(a|s))

        Args:
            old_logits: Previous policy logits for all samples.
            batch_indices: Slice of indices for the current mini-batch.
            logits: Current policy logits from forward pass.
            action_loss: Current action loss tensor.
            kl_coef: Coefficient for KL loss term.
            kl_loss_epoch: Running KL loss accumulator.

        Returns:
            Tuple of (updated action_loss, updated kl_loss_epoch).
        """
        old_logits_batch = old_logits[batch_indices]
        current_logits = logits.squeeze(1)
        old_probs = F.softmax(old_logits_batch, dim=-1)
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        kl_divergence = F.kl_div(
            current_log_probs,
            old_probs,
            reduction="none",
        ).sum(dim=-1)
        kl_loss = kl_divergence.mean()
        action_loss = action_loss + kl_coef * kl_loss
        kl_loss_epoch += kl_loss.item()
        return action_loss, kl_loss_epoch


class ReinforcePPTrainer(BasePolicyTrainer):
    """
    REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models.

    This class implements the REINFORCE++ algorithm which enhances the classical REINFORCE
    algorithm with PPO techniques but without a critic network. Key features:

    1. Mean reward baseline: Subtracts mean reward to reduce variance
    2. Reward normalization: Normalizes rewards across samples for stability
    3. PPO-style clipping: Uses clip-based policy gradient
    4. No critic network: Uses cumulative returns directly

    Reference: https://arxiv.org/abs/2501.03262
    """

    def __init__(
        self,
        policy_model: BasePolicy,
        env: PromptOptimizationEnv,
        max_episodes: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        exploration_module: Optional[BaseExplorationModule] = None,
        clip_param: float = 0.2,
        num_mini_batch: int = 4,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        use_kl_loss: bool = False,
        kl_coef: float = 1e-5,
        use_reward_baseline: bool = True,
        use_reward_normalization: bool = True,
        gamma: float = 1.0,
        eps: float = 1e-8,
        **kwargs,
    ):
        """
        Initialize REINFORCE++ optimizer.

        Args:
            policy_model: The policy model to optimize
            max_episodes: Number of episodes to train for
            optimizer: The optimizer to use for training
            lr_scheduler: Optional learning rate scheduler for dynamic learning rate
            exploration_module: Optional exploration exploitation module
            clip_param: PPO-style clipping parameter
            num_mini_batch: Number of mini-batches per epoch
            entropy_coef: Entropy regularization coefficient
            max_grad_norm: Maximum gradient norm for clipping
            use_kl_loss: Whether to use KL loss from GRPO
            kl_coef: KL divergence coefficient
            use_reward_baseline: Whether to use mean reward baseline
            use_reward_normalization: Whether to normalize rewards
            gamma: Discount factor for return calculation (default 1.0 for REINFORCE++)
            eps: Epsilon for optimizer numerical stability
        """
        # REINFORCE++ is an on-policy algorithm: data is collected from the
        # current policy and used for a single update pass. No replay buffer.
        super().__init__(
            policy_model,
            env,
            max_episodes,
            optimizer,
            lr_scheduler,
            exploration_module,
            None,  # replay_buffer: on-policy, not supported
            eps,
        )
        self.clip_param = clip_param
        self.num_mini_batch = num_mini_batch
        self.entropy_coef = entropy_coef
        self.use_kl_loss = use_kl_loss
        self.kl_coef = kl_coef
        self.use_reward_baseline = use_reward_baseline
        self.use_reward_normalization = use_reward_normalization
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma

    def _compute_returns(
        self, rewards: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cumulative returns using REINFORCE approach (no GAE).

        Args:
            rewards: Rewards of shape (batch_size, sequence_length)
            masks: Binary masks indicating valid positions (batch_size, sequence_length)

        Returns:
            Returns of shape (batch_size, sequence_length)
        """
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros_like(rewards[:, 0])

        # Reverse time steps
        for t in reversed(range(rewards.size(1))):
            cumulative_return = (
                cumulative_return * self.gamma * masks[:, t] + rewards[:, t]
            )
            returns[:, t] = cumulative_return

        return returns

    def _compute_advantages(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages using mean baseline (REINFORCE++ approach).

        Args:
            returns: Cumulative returns of shape (batch_size, sequence_length)

        Returns:
            Advantages of shape (batch_size, sequence_length)
        """
        # Flatten returns for normalization
        flat_returns = returns.flatten()

        # Compute mean baseline
        mean_return = flat_returns.mean()

        # Compute advantages
        advantages = returns - mean_return

        return advantages

    def _normalize_rewards(
        self, rewards: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize rewards across samples for stability.

        Args:
            rewards: Rewards of shape (batch_size, sequence_length)
            masks: Binary masks indicating valid positions

        Returns:
            Normalized rewards
        """
        # Apply mask
        masked_rewards = rewards * masks

        # Compute mean and std per sample
        batch_size = rewards.size(0)
        normalized_rewards = torch.zeros_like(rewards)

        for i in range(batch_size):
            sample_rewards = masked_rewards[i]
            valid_rewards = sample_rewards[sample_rewards != 0]
            if len(valid_rewards) > 1:
                mean = valid_rewards.mean()
                std = valid_rewards.std() + self.eps
                normalized_rewards[i] = (sample_rewards - mean) / std
            elif len(valid_rewards) == 1:
                normalized_rewards[i] = (
                    sample_rewards  # No normalization for single reward
                )
            else:
                normalized_rewards[i] = torch.zeros_like(sample_rewards)

        return normalized_rewards

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_logits: Optional[torch.Tensor] = None,
    ):
        """
        Perform one update step of REINFORCE++.

        Args:
            obs: Observations
            actions: Actions taken
            rewards: Rewards received
            masks: Binary masks for valid positions
            old_log_probs: Previous log probabilities
            old_logits: Previous logits for KL loss computation (optional,
                not available when sampling from replay buffer)

        Returns:
            Tuple with loss values
        """
        returns = self._compute_returns(rewards, masks)
        advantages = self._compute_advantages(returns)
        if self.use_reward_normalization:
            rewards = self._normalize_rewards(rewards, masks)

        # Normalize advantages
        adv_std = advantages.std()
        if adv_std > self.eps:
            advantages = (advantages - advantages.mean()) / adv_std

        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        kl_loss_epoch = 0.0
        batch_size = rewards.size(0)
        indices = torch.randperm(batch_size)

        for i in range(0, batch_size, self.num_mini_batch):
            batch_indices = indices[i : i + self.num_mini_batch]

            # Get batch data
            obs_batch = obs[batch_indices]
            actions_batch = actions[batch_indices]
            masks_batch = masks[batch_indices]
            old_log_probs_batch = (
                old_log_probs[batch_indices] if old_log_probs is not None else None
            )
            advantages_batch = advantages[batch_indices]

            # Forward pass
            _, action_log_probs, dist_entropy, logits = (
                self.policy_model.evaluate_actions(
                    obs_batch, actions_batch, masks_batch
                )
            )

            # Compute policy loss with PPO-style clipping
            if old_log_probs_batch is not None:
                ratio = torch.exp(action_log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * advantages_batch
                )
                action_loss = -torch.min(surr1, surr2).mean()
            else:
                # If no old log probs, use unclipped policy gradient
                action_loss = -(action_log_probs * advantages_batch).mean()

            # Add entropy regularization
            action_loss = action_loss - self.entropy_coef * dist_entropy.mean()

            # Add KL loss if enabled
            if self.use_kl_loss and old_logits is not None:
                action_loss, kl_loss_epoch = BasePolicyTrainer._compute_kl_loss(
                    old_logits,
                    slice(i, i + self.num_mini_batch),
                    logits,
                    action_loss,
                    self.kl_coef,
                    kl_loss_epoch,
                )

            # Backward pass
            self.optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Step the learning rate scheduler if provided
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.mean().item()

        num_updates = batch_size // self.num_mini_batch

        return (
            action_loss_epoch / max(num_updates, 1),
            dist_entropy_epoch / max(num_updates, 1),
            kl_loss_epoch / max(num_updates, 1),
        )

    def fit(
        self,
        checkpoint_every: int = 10,
        earlystop_last: int = 100,
        record_every: int = 1000,
        use_wandb: bool = True,
        wandb_project: str = "prompt_optimization",
        checkpoint_dir: str = "./ckpt",
        **kwargs,
    ) -> None:
        """Train the policy using REINFORCE++ or PPO algorithm.

        Args:
            max_episodes: Maximum number of training episodes
            checkpoint_every: Save checkpoint every N episodes
            earlystop_last: Early stop if no improvement in N episodes
            record_every: Record episode every N episodes
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: WandB project name
            checkpoint_dir: Directory to save checkpoints
            **kwargs: Additional arguments
        """
        # Initialize WandB if enabled
        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    config={
                        "max_episodes": self.max_episodes,
                        "checkpoint_every": checkpoint_every,
                        "earlystop_last": earlystop_last,
                        "record_every": record_every,
                    },
                )
            except ImportError:
                warnings.warn("wandb not installed, skipping WandB logging")
                use_wandb = False

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load latest checkpoint if exists
        ckpt_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
        ckpt_files = glob.glob(ckpt_pattern)
        episode_start = 0
        best_reward = -float("inf")
        if ckpt_files:
            episodes = []
            for f in ckpt_files:
                try:
                    ep = int(os.path.basename(f).split("_")[1].split(".")[0])
                    episodes.append(ep)
                except ValueError:
                    pass
            if episodes:
                latest_episode = max(episodes)
                ckpt_path = os.path.join(
                    checkpoint_dir, f"checkpoint_{latest_episode}.pt"
                )
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                self.policy_model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                episode_start = latest_episode + 1
                best_reward = checkpoint.get("best_reward", -float("inf"))
                print(f"Loaded checkpoint from episode {latest_episode}")

        # Training loop
        episode_rewards = []
        no_improvement_count = 0

        for episode in range(episode_start, self.max_episodes):
            # Reset environment
            obs, info = self.env.reset()
            self.exploration_module.reset(episode)
            episode_reward = 0.0
            episode_data = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "masks": [],
                "log_probs": [],
                "old_logits": [],
            }

            # Collect trajectory
            done = False
            truncated = False
            step_count = 0

            while not (done or truncated) and step_count < self.env.max_steps:
                # Convert observation to tensor with proper shape for policy model
                # obs shape: (embedding_dim,) -> (1, 1, embedding_dim) for batch=1, seq=1
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                )

                # Get action from policy with exploration
                with torch.no_grad():
                    logits = self.policy_model(obs_tensor)
                    # Store logits for KL divergence computation
                    episode_data["old_logits"].append(logits[0, 0].detach())
                    # Determine whether to explore or exploit
                    should_explore = self.exploration_module.should_explore(
                        step=step_count, episode=episode
                    )
                    # Extract action from policy output using helper method
                    action_tensor, log_prob = self.policy_model.get_action(
                        logits, deterministic=not should_explore
                    )
                    action = int(action_tensor.item())
                    log_prob = log_prob[0].item()

                # Take step in environment
                next_obs, reward, done, truncated, info = self.env.step(action)

                # Store transition
                episode_data["observations"].append(obs)
                episode_data["actions"].append(action)
                episode_data["rewards"].append(reward)
                episode_data["masks"].append(1.0 if not (done or truncated) else 0.0)
                episode_data["log_probs"].append(log_prob)

                episode_reward += reward
                obs = next_obs
                step_count += 1

            # Convert to tensors
            obs_tensor = torch.tensor(episode_data["observations"], dtype=torch.float32)
            actions_tensor = torch.tensor(episode_data["actions"], dtype=torch.long)
            rewards_tensor = torch.tensor(episode_data["rewards"], dtype=torch.float32)
            masks_tensor = torch.tensor(episode_data["masks"], dtype=torch.float32)
            old_log_probs_tensor = torch.tensor(
                episode_data["log_probs"], dtype=torch.float32
            )
            old_logits_tensor = torch.stack(episode_data["old_logits"], dim=0)

            # Update policy
            action_loss, dist_entropy, kl_loss = self.update(
                obs_tensor.unsqueeze(0),
                actions_tensor.unsqueeze(0),
                rewards_tensor.unsqueeze(0),
                masks_tensor.unsqueeze(0),
                old_log_probs_tensor.unsqueeze(0),
                old_logits_tensor,
            )

            # Track metrics
            episode_rewards.append(episode_reward)

            # Check for improvement (early stopping)
            mean_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)) :])

            if mean_reward > best_reward:
                best_reward = mean_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Logging
            if use_wandb:
                wandb.log(
                    {
                        "episode": episode,
                        "episode_reward": episode_reward,
                        "mean_reward_100": mean_reward,
                        "best_reward": best_reward,
                        "steps": step_count,
                        "action_loss": action_loss,
                        "dist_entropy": dist_entropy,
                        "kl_loss": kl_loss,
                    }
                )

            print(
                f"Episode {episode}: reward={episode_reward:.4f}, mean_100={mean_reward:.4f}"
            )

            # Checkpointing
            if (episode + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pt")
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": self.policy_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_reward": best_reward,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

            # Early stopping
            if no_improvement_count >= earlystop_last:
                print(f"Early stopping at episode {episode}")
                break

        if use_wandb:
            wandb.finish()
        print("done")


class GRPOTrainer(BasePolicyTrainer):
    """Group Relative Policy Optimization trainer.

    This trainer implements a critic-free, group-relative advantage approach
    for prompt optimization. It uses groups of outputs sampled from the same
    prompt to compute relative advantages and optionally normalizes rewards
    within each group.

    Supports both on-policy and off-policy training modes:

    - **On-policy** (default): Group data is collected from the current policy
      and used for a single update pass. No replay buffer.
    - **Off-policy with replay buffer**: Uses a replay buffer to store transitions
      and samples from past experiences. Implements the clipped surrogate objective
      with importance sampling as described in Mroueh et al. (2025)
      "Revisiting Group Relative Policy Optimization: Insights into On-Policy
      and Off-Policy Training".
    - **Off-policy without replay buffer**: Uses the current episode's data with
      the off-policy clipped surrogate objective. The importance sampling ratio
      is approximated as π_k/α ≈ 1 for stability, providing training stability
      similar to off-policy training without requiring a separate behavior policy.

    References:
        - https://arxiv.org/pdf/2402.03300 (original GRPO)
        - https://arxiv.org/pdf/2505.22257 (off-policy GRPO)
    """

    def __init__(
        self,
        policy_model: BasePolicy,
        env: PromptOptimizationEnv,
        max_episodes: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        exploration_module: Optional[BaseExplorationModule] = None,
        replay_buffer: Optional[BaseReplayBuffer] = None,
        clip_param: float = 0.2,
        num_mini_batch: int = 4,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        use_kl_loss: bool = False,
        kl_coef: float = 1e-5,
        group_size: int = 1,
        use_reward_normalization: bool = True,
        gamma: float = 1.0,
        eps: float = 1e-8,
        use_off_policy: bool = False,
        use_zero_variance_masking: bool = False,
        **kwargs,
    ):
        super().__init__(
            policy_model,
            env,
            max_episodes,
            optimizer,
            lr_scheduler,
            exploration_module,
            replay_buffer,  # allow replay buffer for off-policy
            eps,
        )
        self.clip_param = clip_param
        self.num_mini_batch = num_mini_batch
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_kl_loss = use_kl_loss
        self.kl_coef = kl_coef
        self.group_size = max(1, group_size)
        self.use_reward_normalization = use_reward_normalization
        self.gamma = gamma
        self.use_off_policy = use_off_policy
        self.use_zero_variance_masking = use_zero_variance_masking

    def _compute_returns(
        self, rewards: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros_like(rewards[:, 0])

        for t in reversed(range(rewards.size(1))):
            cumulative_return = (
                cumulative_return * self.gamma * masks[:, t] + rewards[:, t]
            )
            returns[:, t] = cumulative_return

        return returns

    def _compute_group_advantages(self, returns: torch.Tensor) -> torch.Tensor:
        final_returns = returns[:, 0]
        batch_size = final_returns.size(0)

        if self.use_off_policy:
            if self.replay_buffer is not None and not self.replay_buffer.is_empty():
                # Off-policy GRPO with replay buffer: advantage is estimated using
                # statistics from the behavior policy α (stored in replay buffer).
                # A^α(x, y) = (r(x, y) - μ_α,r(x)) / σ_α,r,ε(x)
                # where μ and σ are computed from the off-policy distribution.

                # For off-policy, we compute group statistics from the buffer
                # to estimate the behavior policy's reward distribution.
                # Sample a batch from the buffer to compute μ_α and σ_α.
                buffer_batch = self.replay_buffer.sample(
                    min(batch_size * self.group_size, len(self.replay_buffer))
                )
                buffer_rewards = torch.tensor(
                    buffer_batch["reward"], dtype=torch.float32
                )

                # Compute behavior policy statistics
                buffer_mean = buffer_rewards.mean()
                buffer_std = buffer_rewards.std(unbiased=False)

                # Compute advantage using behavior policy statistics
                advantages = (final_returns - buffer_mean) / (buffer_std + self.eps)

                # Zero-variance masking (DAPO-style): mask samples where the
                # behavior policy has zero variance (fully correct or incorrect).
                # This prevents the total variation term from dominating the
                # policy improvement lower bound (Theorem 1 in Mroueh et al. 2025).
                if self.use_zero_variance_masking and buffer_std < self.eps:
                    advantages = torch.zeros_like(advantages)

                normalized_returns = advantages
            else:
                # Off-policy GRPO without replay buffer: use current episode's
                # group statistics. The advantage is computed from the current
                # policy's samples, but the clipped surrogate objective with
                # importance sampling provides off-policy training stability.
                # This is equivalent to on-policy advantage estimation with
                # off-policy clipping (π_k/α ≈ 1 approximation).
                if batch_size % self.group_size != 0:
                    raise ValueError(
                        "GRPO requires batch_size to be divisible by group_size"
                    )

                group_returns = final_returns.view(-1, self.group_size)
                group_mean = group_returns.mean(dim=1, keepdim=True)

                if self.use_reward_normalization:
                    group_std = group_returns.std(dim=1, unbiased=False, keepdim=True)
                    normalized = (group_returns - group_mean) / (group_std + self.eps)
                else:
                    normalized = group_returns - group_mean

                normalized_returns = normalized.view(batch_size)
        else:
            # On-policy GRPO: advantage is estimated using statistics from
            # the current policy's group samples.
            # A^π_k(x, y) = (r(x, y) - mean({r_l})) / std({r_l}) + ε
            if batch_size % self.group_size != 0:
                raise ValueError(
                    "GRPO requires batch_size to be divisible by group_size"
                )

            group_returns = final_returns.view(-1, self.group_size)
            group_mean = group_returns.mean(dim=1, keepdim=True)

            if self.use_reward_normalization:
                group_std = group_returns.std(dim=1, unbiased=False, keepdim=True)
                normalized = (group_returns - group_mean) / (group_std + self.eps)
            else:
                normalized = group_returns - group_mean

            normalized_returns = normalized.view(batch_size)

        return normalized_returns.unsqueeze(-1).expand_as(returns)

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_logits: Optional[torch.Tensor] = None,
    ):
        """Perform one update step of GRPO.

        Args:
            obs: Observations
            actions: Actions taken
            rewards: Rewards received
            masks: Binary masks for valid positions
            old_log_probs: Previous log probabilities
            old_logits: Previous logits for KL loss computation (optional,
                not available when sampling from replay buffer)

        Returns:
            Tuple of (action_loss, dist_entropy, kl_loss)
        """
        returns = self._compute_returns(rewards, masks)
        advantages = self._compute_group_advantages(returns)

        # Normalize advantages for stability
        adv_std = advantages.std()
        if adv_std > self.eps:
            advantages = (advantages - advantages.mean()) / adv_std

        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        kl_loss_epoch = 0.0
        batch_size = rewards.size(0)
        indices = torch.randperm(batch_size)

        for i in range(0, batch_size, self.num_mini_batch):
            batch_indices = indices[i : i + self.num_mini_batch]

            obs_batch = obs[batch_indices]
            actions_batch = actions[batch_indices]
            masks_batch = masks[batch_indices]
            old_log_probs_batch = (
                old_log_probs[batch_indices] if old_log_probs is not None else None
            )
            advantages_batch = advantages[batch_indices]

            _, action_log_probs, dist_entropy, logits = (
                self.policy_model.evaluate_actions(
                    obs_batch, actions_batch, masks_batch
                )
            )

            if self.use_off_policy:
                # Off-policy GRPO: use importance sampling with clipped
                # surrogate objective (Mroueh et al. 2025, Eq. 6).
                #
                # L_α^c(π) = E_{y~α} [ min( (π/α) * A^α,
                #     clip(π/α, 1-ε, 1+ε) * A^α ) ]
                #
                # In practice, we approximate π_k/α ≈ 1 for stability when
                # α is close to π_k (the behavior policy).
                if old_log_probs_batch is not None:
                    # Compute importance sampling ratio: π(y|x) / α(y|x)
                    # Using the approximation π_k/α ≈ 1 for stability
                    # (as recommended in off-policy PPO literature)
                    ratio = torch.exp(action_log_probs - old_log_probs_batch)

                    surr1 = ratio * advantages_batch
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1.0 - self.clip_param,
                            1.0 + self.clip_param,
                        )
                        * advantages_batch
                    )
                    action_loss = -torch.min(surr1, surr2).mean()
                else:
                    # No old log probs: use current policy log probs directly
                    # (treats ratio as 1, i.e., π/α ≈ 1)
                    action_loss = -(action_log_probs * advantages_batch).mean()
            else:
                # On-policy GRPO: standard PPO-style clipping
                if old_log_probs_batch is not None:
                    ratio = torch.exp(action_log_probs - old_log_probs_batch)
                    surr1 = ratio * advantages_batch
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1.0 - self.clip_param,
                            1.0 + self.clip_param,
                        )
                        * advantages_batch
                    )
                    action_loss = -torch.min(surr1, surr2).mean()
                else:
                    action_loss = -(action_log_probs * advantages_batch).mean()

            action_loss = action_loss - self.entropy_coef * dist_entropy.mean()

            # Add KL loss if enabled
            if self.use_kl_loss and old_logits is not None:
                action_loss, kl_loss_epoch = BasePolicyTrainer._compute_kl_loss(
                    old_logits,
                    slice(i, i + self.num_mini_batch),
                    logits,
                    action_loss,
                    self.kl_coef,
                    kl_loss_epoch,
                )

            self.optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.mean().item()

        num_updates = batch_size // self.num_mini_batch
        return (
            action_loss_epoch / max(num_updates, 1),
            dist_entropy_epoch / max(num_updates, 1),
            kl_loss_epoch / max(num_updates, 1),
        )

    def fit(
        self,
        checkpoint_every: int = 10,
        earlystop_last: int = 100,
        record_every: int = 1000,
        use_wandb: bool = True,
        wandb_project: str = "prompt_optimization",
        checkpoint_dir: str = "./ckpt",
        **kwargs,
    ):
        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    config={
                        "max_episodes": self.max_episodes,
                        "checkpoint_every": checkpoint_every,
                        "earlystop_last": earlystop_last,
                        "record_every": record_every,
                    },
                )
            except ImportError:
                warnings.warn("wandb not installed, skipping WandB logging")
                use_wandb = False

        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_pattern = os.path.join(checkpoint_dir, "grpo_checkpoint_*.pt")
        ckpt_files = glob.glob(ckpt_pattern)
        episode_start = 0
        best_reward = -float("inf")
        if ckpt_files:
            episodes = []
            for f in ckpt_files:
                try:
                    ep = int(os.path.basename(f).split("_")[2].split(".")[0])
                    episodes.append(ep)
                except ValueError:
                    pass
            if episodes:
                latest_episode = max(episodes)
                ckpt_path = os.path.join(
                    checkpoint_dir, f"grpo_checkpoint_{latest_episode}.pt"
                )
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                self.policy_model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                episode_start = latest_episode + 1
                best_reward = checkpoint.get("best_reward", -float("inf"))
                print(f"Loaded checkpoint from episode {latest_episode}")

        episode_rewards = []
        no_improvement_count = 0

        for episode in range(episode_start, self.max_episodes):
            obs, _ = self.env.reset()
            self.exploration_module.reset(episode)
            episode_reward = 0.0
            episode_data = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "masks": [],
                "log_probs": [],
                "old_logits": [],
            }

            done = False
            truncated = False
            step_count = 0

            while not (done or truncated) and step_count < self.env.max_steps:
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                )
                with torch.no_grad():
                    logits = self.policy_model(obs_tensor)
                    # Store logits for KL divergence computation
                    episode_data["old_logits"].append(logits[0, 0].detach())
                    # Determine whether to explore or exploit
                    should_explore = self.exploration_module.should_explore(
                        step=step_count, episode=episode
                    )
                    action_tensor, log_prob = self.policy_model.get_action(
                        logits, deterministic=not should_explore
                    )
                    _, action_log_probs, _, _ = self.policy_model.evaluate_actions(
                        obs_tensor,
                        action_tensor,
                        torch.ones_like(action_tensor, dtype=torch.float32),
                    )

                action = int(action_tensor.item())
                log_prob = log_prob[0].item()

                next_obs, reward, done, truncated, _ = self.env.step(action)

                episode_data["observations"].append(obs)
                episode_data["actions"].append(action)
                episode_data["rewards"].append(reward)
                episode_data["masks"].append(1.0 if not (done or truncated) else 0.0)
                episode_data["log_probs"].append(log_prob)

                # Off-policy: push transition to replay buffer
                if self.use_off_policy and self.replay_buffer is not None:
                    self.replay_buffer.push(
                        obs=np.asarray(obs, dtype=np.float32),
                        action=action,
                        reward=reward,
                        log_prob=log_prob,
                        mask=1.0 if not (done or truncated) else 0.0,
                    )

                episode_reward += reward
                obs = next_obs
                step_count += 1

            if len(episode_data["observations"]) == 0:
                continue

            obs_tensor = torch.tensor(episode_data["observations"], dtype=torch.float32)
            actions_tensor = torch.tensor(episode_data["actions"], dtype=torch.long)
            rewards_tensor = torch.tensor(episode_data["rewards"], dtype=torch.float32)
            masks_tensor = torch.tensor(episode_data["masks"], dtype=torch.float32)
            old_log_probs_tensor = torch.tensor(
                episode_data["log_probs"], dtype=torch.float32
            )
            old_logits_tensor = torch.stack(episode_data["old_logits"], dim=0)

            if self.use_off_policy and self.replay_buffer is not None:
                # Off-policy with replay buffer: sample from buffer for update.
                # This allows reusing past experience for multiple updates,
                # reducing communication overhead in distributed training.
                if self.replay_buffer.is_empty():
                    print(
                        f"Episode {episode}: Buffer empty, skipping update. "
                        f"Collecting experience..."
                    )
                    action_loss = 0.0
                    dist_entropy = 0.0
                    kl_loss = 0.0
                else:
                    # Sample a batch from the replay buffer
                    # Use group_size to determine batch size for GRPO
                    buffer_batch_size = max(
                        self.group_size,
                        self.num_mini_batch,
                    )
                    buffer_batch = self.replay_buffer.sample(
                        min(buffer_batch_size, len(self.replay_buffer))
                    )
                    # Convert buffer batch to tensors
                    obs_buffer = torch.tensor(
                        buffer_batch["obs"], dtype=torch.float32
                    ).unsqueeze(0)
                    actions_buffer = torch.tensor(
                        buffer_batch["action"], dtype=torch.long
                    ).unsqueeze(0)
                    rewards_buffer = torch.tensor(
                        buffer_batch["reward"], dtype=torch.float32
                    ).unsqueeze(0)
                    masks_buffer = torch.tensor(
                        buffer_batch["mask"], dtype=torch.float32
                    ).unsqueeze(0)
                    log_probs_buffer = torch.tensor(
                        buffer_batch["log_prob"], dtype=torch.float32
                    ).unsqueeze(0)

                    action_loss, dist_entropy, kl_loss = self.update(
                        obs_buffer,
                        actions_buffer,
                        rewards_buffer,
                        masks_buffer,
                        log_probs_buffer,
                    )
            else:
                # On-policy or off-policy without replay buffer: use current
                # episode's data for update. When use_off_policy=True without
                # a buffer, the off-policy clipped surrogate objective is still
                # applied (π_k/α ≈ 1 approximation), providing training stability.
                action_loss, dist_entropy, kl_loss = self.update(
                    obs_tensor.unsqueeze(0),
                    actions_tensor.unsqueeze(0),
                    rewards_tensor.unsqueeze(0),
                    masks_tensor.unsqueeze(0),
                    old_log_probs_tensor.unsqueeze(0),
                    old_logits_tensor,
                )

            episode_rewards.append(episode_reward)
            mean_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)) :])

            if mean_reward > best_reward:
                best_reward = mean_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Log buffer size for off-policy mode
            buffer_size = (
                len(self.replay_buffer)
                if self.use_off_policy and self.replay_buffer is not None
                else 0
            )

            if use_wandb:
                log_data = {
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "mean_reward_100": mean_reward,
                    "best_reward": best_reward,
                    "steps": step_count,
                    "action_loss": action_loss,
                    "dist_entropy": dist_entropy,
                    "kl_loss": kl_loss,
                }
                if self.use_off_policy:
                    log_data["buffer_size"] = buffer_size
                wandb.log(log_data)

            print(
                f"Episode {episode}: reward={episode_reward:.4f}, mean_100={mean_reward:.4f}"
            )
            if self.use_off_policy:
                print(f"  Buffer size: {buffer_size}")

            if (episode + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir, f"grpo_checkpoint_{episode}.pt"
                )
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": self.policy_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_reward": best_reward,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

            if no_improvement_count >= earlystop_last:
                print(f"Early stopping at episode {episode}")
                break

        if use_wandb:
            wandb.finish()
        print("done")


class PPOTrainer(BasePolicyTrainer):
    """Proximal Policy Optimization (PPO) trainer for prompt augmentation.

    Implements the PPO-Clip algorithm from Schulman et al. (2017)
    "Proximal Policy Optimization Algorithms", adapted for prompt
    optimization in the ``aap_core`` framework.

    Supports both **on-policy** and **off-policy** training modes:

    - **On-policy** (default): Trajectories are collected from the current
      policy and used for multiple update epochs, then discarded. No replay
      buffer. This is the standard PPO setting.
    - **Off-policy with replay buffer**: Uses a replay buffer to store
      transitions and samples from past experiences. Applies importance
      sampling (IS) to correct for the distribution mismatch between the
      behavior policy that generated the data and the current policy being
      optimized. Supports Truncated Importance Sampling (TIS) to bound IS
      ratio variance. Based on the off-policy PPO formulation from Mroueh
      et al. (2025) and the RPG framework (ICLR 2026).
    - **Off-policy without replay buffer**: Uses the current episode's data
      with the off-policy clipped surrogate objective. The importance
      sampling ratio is approximated as ``π_k/α ≈ 1`` for stability,
      providing training stability similar to off-policy training without
      requiring a separate behavior policy.

    Key components:

    1. **Clipped surrogate objective** — limits the policy update ratio
       ``r_t(θ) = π_θ(a|s) / π_θ_old(a|s)`` to ``[1-ε, 1+ε]``, preventing
       destructive large updates.

    2. **Value function loss** — a critic network (built into the
       ``BasePolicy`` actor-critic) predicts state values; the loss
       minimises the mean-squared error between predicted values and
       computed returns (discounted cumulative rewards). Supports optional
       clipped value predictions for additional stability.

    3. **Entropy bonus** — encourages exploration by adding the policy
       entropy to the loss (subtracted, so gradient ascent).

    4. **Advantage normalisation** — advantages ``(returns - values)`` are
       standardised (zero mean, unit variance) before the update.

    5. **Optional KL penalty** — when ``use_kl_loss=True``, a KL divergence
       term is added to the action loss to penalise deviation from the
       previous policy.

    6. **Importance sampling (off-policy)** — when ``use_off_policy=True``,
       the clipped surrogate objective is weighted by the IS ratio
       ``r_θ(s,a) = π_θ(a|s) / α(a|s)`` to correct for distribution
       mismatch between the data-collection policy α and the current policy
       π_θ. Truncated IS (TIS) caps the ratio at ``tis_cap`` to bound
       variance.

    Training loop (per ``fit()`` call):

    **On-policy mode:**

    1. Reset environment and collect a trajectory: ``(s_t, a_t, r_t,
       log_prob_t, value_t, logits_t)`` for ``t = 0, …, T-1``.
    2. Compute returns via backward pass: ``G_t = Σ γ^(t-k) r_k``.
    3. Compute advantages: ``A_t = G_t - V(s_t)``, normalise.
    4. Run ``update()`` — iterate over ``num_mini_batch`` mini-batches,
       applying the clipped surrogate objective, value loss, and entropy
       bonus for ``num_epochs`` passes.
    5. Log metrics (WandB), checkpoint, and check early stopping.

    **Off-policy mode (with replay buffer):**

    1. Collect a transition ``(s, a, r, log_prob)`` and push to replay
       buffer.
    2. Sample a batch from the buffer.
    3. Compute advantages from sampled data.
    4. Run ``update()`` with IS ratio correction and TIS.
    5. Log metrics, checkpoint, and check early stopping.

    Example usage (on-policy)::

        trainer = PPOTrainer(
            actor_critic=policy,
            env=env,
            max_episodes=500,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            num_mini_batch=4,
            value_loss_coef=0.5,
            entropy_coef=0.01,
        )
        trainer.fit(checkpoint_every=50, earlystop_last=100)

    Example usage (off-policy)::

        buffer = SimpleReplayBuffer(capacity=10000)
        trainer = PPOTrainer(
            actor_critic=policy,
            env=env,
            max_episodes=500,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            num_mini_batch=4,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            replay_buffer=buffer,
            use_off_policy=True,
            tis_cap=2.0,
        )
        trainer.fit(checkpoint_every=50, earlystop_last=100)

    References:
        - Schulman et al., "Proximal Policy Optimization Algorithms",
          2017. https://arxiv.org/abs/1707.06347
        - Schulman et al., "Trust Region Policy Optimization", 2015.
          https://arxiv.org/abs/1502.05477
        - Mroueh et al., "Revisiting Group Relative Policy Optimization:
          Insights into On-Policy and Off-Policy Training", 2025.
        - ICLR 2026, "On the Design of KL-Regularized Policy Gradient
          Algorithms for LLM Reasoning" (RPG framework).

    Args:
        actor_critic: Actor-critic policy model (``BasePolicy``) that
            provides both action logits and value predictions.
        env: Prompt optimization environment (``PromptOptimizationEnv``).
        max_episodes: Maximum number of training episodes.
        optimizer: PyTorch optimizer for policy parameter updates.
        lr_scheduler: Optional learning rate scheduler.
        num_mini_batch: Number of mini-batches to split each trajectory
            into during the update step.
        value_loss_coef: Weight for the value function loss in the total
            loss: ``L = L_CLIP + c_1 * L_VAL - c_2 * entropy``.
        entropy_coef: Weight for the entropy bonus (exploration
            encouragement).
        exploration_module: Optional exploration strategy (e.g.,
            ``EpsilonGreedyExploration``). Defaults to pure exploit.
        replay_buffer: Optional replay buffer for off-policy training.
            When provided and ``use_off_policy=True``, transitions are
            stored and sampled from the buffer for updates.
        use_off_policy: If ``True``, enables off-policy training with
            importance sampling correction. Requires a ``replay_buffer``
            for full off-policy functionality; without a buffer, uses the
            current episode data with ``π_k/α ≈ 1`` approximation.
            Default ``False``.
        clip_param: PPO clipping epsilon ``ε`` for the surrogate objective
            and (optionally) value function. Default ``0.2``.
        max_grad_norm: Maximum L2 norm for gradient clipping. Default ``1.0``.
        use_clipped_value_loss: If ``True``, uses the clipped value loss
            from the original PPO paper, clamping value predictions to
            ``[V_old - ε, V_old + ε]``. Default ``True``.
        use_kl_loss: If ``True``, adds a KL divergence penalty between the
            current and old policy to the action loss. Default ``False``.
        kl_coef: Coefficient for the KL loss term. Default ``1e-5``.
        gamma: Discount factor for computing returns (``γ ∈ [0, 1]``).
            Default ``0.99``.
        tis_cap: Truncated importance sampling cap. The IS ratio
            ``r_θ(s,a)`` is capped at this value to bound variance in
            off-policy training. Only used when ``use_off_policy=True``.
            Default ``2.0``.
        eps: Small constant for numerical stability in advantage
            normalisation. Default ``1e-8``.
    """

    def __init__(
        self,
        actor_critic: BasePolicy,
        env: PromptOptimizationEnv,
        max_episodes: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        exploration_module: Optional[BaseExplorationModule] = None,
        replay_buffer: Optional[BaseReplayBuffer] = None,
        use_off_policy: bool = False,
        clip_param: float = 0.2,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        use_kl_loss: bool = False,
        kl_coef: float = 1e-5,
        gamma: float = 0.99,
        tis_cap: float = 2.0,
        eps: float = 1e-8,
        **kwargs,
    ):
        # PPO supports both on-policy and off-policy training.
        # On-policy: trajectories are collected from the current policy and
        # used for multiple update epochs, then discarded.
        # Off-policy: uses a replay buffer to store transitions and samples
        # from past experiences, applying importance sampling correction.
        super().__init__(
            actor_critic,
            env,
            max_episodes,
            optimizer,
            lr_scheduler,
            exploration_module,
            replay_buffer,
            eps,
        )
        self.clip_param = clip_param
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_kl_loss = use_kl_loss
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.use_off_policy = use_off_policy
        self.tis_cap = tis_cap

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_logits: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        """Perform one update step of PPO.

        Args:
            obs: Observations
            actions: Actions taken
            rewards: Rewards received
            masks: Binary masks for valid positions
            old_log_probs: Previous log probabilities
            values: Previous value predictions
            returns: Computed returns
            old_logits: Previous logits for KL loss computation

        Returns:
            Tuple of (value_loss, action_loss, dist_entropy, kl_loss)
        """
        advantages = returns - values
        adv_std = advantages.std()
        if adv_std > self.eps:
            advantages = (advantages - advantages.mean()) / adv_std

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        kl_loss_epoch = 0.0
        num_updates = 0

        batch_size = rewards.size(0)
        indices = torch.randperm(batch_size)

        for i in range(0, batch_size, self.num_mini_batch):
            batch_indices = indices[i : i + self.num_mini_batch]
            obs_batch = obs[batch_indices].unsqueeze(1)
            actions_batch = actions[batch_indices]
            masks_batch = masks[batch_indices]
            old_log_probs_batch = old_log_probs[batch_indices]
            values_batch = values[batch_indices]
            return_batch = returns[batch_indices]
            adv_batch = advantages[batch_indices]

            values_pred, action_log_probs, dist_entropy, logits = (
                self.policy_model.evaluate_actions(
                    obs_batch,
                    actions_batch,
                    masks_batch,
                )
            )

            ratio = torch.exp(action_log_probs - old_log_probs_batch.unsqueeze(-1))

            # Off-policy: apply importance sampling ratio correction
            if self.use_off_policy:
                # Truncated importance sampling (TIS) to bound variance
                # r_θ(s,a) = π_θ(a|s) / α(a|s), capped at tis_cap
                log_ratio = (
                    action_log_probs - old_log_probs_batch.unsqueeze(-1)
                ).clamp(-10.0, 10.0)
                tis_ratio = torch.exp(log_ratio).clamp(max=self.tis_cap)
                tis_ratio = tis_ratio.detach()  # stop gradient through IS ratio

                # Apply IS weight to the clipped surrogate objective
                surr1 = tis_ratio * ratio * adv_batch.unsqueeze(-1)
                surr2 = (
                    tis_ratio
                    * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv_batch.unsqueeze(-1)
                )
                action_loss = -torch.min(surr1, surr2).mean()
            else:
                surr1 = ratio * adv_batch.unsqueeze(-1)
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                ) * adv_batch.unsqueeze(-1)
                action_loss = -torch.min(surr1, surr2).mean()

            # Add KL loss if enabled
            if self.use_kl_loss:
                action_loss, kl_loss_epoch = BasePolicyTrainer._compute_kl_loss(
                    old_logits,
                    slice(i, i + self.num_mini_batch),
                    logits,
                    action_loss,
                    self.kl_coef,
                    kl_loss_epoch,
                )

            if self.use_clipped_value_loss:
                value_pred_clipped = values_batch + (values_pred - values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values_pred - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = 0.5 * (return_batch - values_pred).pow(2).mean()

            loss = (
                value_loss * self.value_loss_coef
                + action_loss
                - dist_entropy.mean() * self.entropy_coef
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.mean().item()
            num_updates += 1

        return (
            value_loss_epoch / max(num_updates, 1),
            action_loss_epoch / max(num_updates, 1),
            dist_entropy_epoch / max(num_updates, 1),
            kl_loss_epoch / max(num_updates, 1),
        )

    def fit(
        self,
        checkpoint_every: int = 10,
        earlystop_last: int = 100,
        record_every: int = 1000,
        use_wandb: bool = True,
        wandb_project: str = "prompt_optimization",
        checkpoint_dir: str = "./ckpt",
        **kwargs,
    ):
        """Train the policy using PPO with on-policy or off-policy interaction.

        On-policy mode: collects trajectories from the current policy and
        updates using the clipped surrogate objective.

        Off-policy mode: collects transitions to a replay buffer and samples
        from it for updates with importance sampling correction.

        Args:
            checkpoint_every: Save checkpoint every N episodes.
            earlystop_last: Early stop if no improvement in N episodes.
            record_every: Record episode every N episodes (not implemented).
            use_wandb: Whether to use Weights & Biases for logging.
            wandb_project: WandB project name.
            checkpoint_dir: Directory to save checkpoints.
            **kwargs: Additional arguments. Supported keys: gamma.
        """

        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    config={
                        "max_episodes": self.max_episodes,
                        "checkpoint_every": checkpoint_every,
                        "earlystop_last": earlystop_last,
                        "record_every": record_every,
                    },
                )
            except ImportError:
                warnings.warn("wandb not installed, skipping WandB logging")
                use_wandb = False

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load latest checkpoint if exists
        ckpt_pattern = os.path.join(checkpoint_dir, "ppo_checkpoint_*.pt")
        ckpt_files = glob.glob(ckpt_pattern)
        episode_start = 0
        best_reward = -float("inf")
        if ckpt_files:
            episodes = []
            for f in ckpt_files:
                try:
                    ep = int(
                        os.path.basename(f).split("_")[2].split(".")[0]
                    )  # ppo_checkpoint_{ep}.pt
                    episodes.append(ep)
                except ValueError:
                    pass
            if episodes:
                latest_episode = max(episodes)
                ckpt_path = os.path.join(
                    checkpoint_dir, f"ppo_checkpoint_{latest_episode}.pt"
                )
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                self.policy_model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                episode_start = latest_episode + 1
                best_reward = checkpoint.get("best_reward", -float("inf"))
                print(f"Loaded checkpoint from episode {latest_episode}")

        episode_rewards = []
        no_improvement_count = 0

        for episode in range(episode_start, self.max_episodes):
            obs, _ = self.env.reset()
            self.exploration_module.reset(episode)
            episode_reward = 0.0
            episode_data = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "masks": [],
                "log_probs": [],
                "values": [],
                "old_logits": [],
            }

            done = False
            truncated = False
            step_count = 0

            while not (done or truncated) and step_count < self.env.max_steps:
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                )
                with torch.no_grad():
                    logits = self.policy_model(obs_tensor)
                    # Determine whether to explore or exploit
                    should_explore = self.exploration_module.should_explore(
                        step=step_count, episode=episode
                    )
                    action_tensor, log_prob = self.policy_model.get_action(
                        logits, deterministic=not should_explore
                    )
                    values, action_log_probs, _, _ = self.policy_model.evaluate_actions(
                        obs_tensor,
                        action_tensor,
                        torch.ones_like(action_tensor, dtype=torch.float32),
                    )

                action = int(action_tensor.item())
                log_prob = log_prob[0].item()
                value = values[0].item()

                next_obs, reward, done, truncated, _ = self.env.step(action)

                # Off-policy: push transition to replay buffer
                if self.use_off_policy and self.replay_buffer is not None:
                    self.replay_buffer.push(
                        obs=np.asarray(obs, dtype=np.float32),
                        action=action,
                        reward=reward,
                        log_prob=log_prob,
                        mask=1.0 if not (done or truncated) else 0.0,
                    )

                episode_data["observations"].append(obs)
                episode_data["actions"].append(action)
                episode_data["rewards"].append(reward)
                episode_data["masks"].append(1.0 if not (done or truncated) else 0.0)
                episode_data["log_probs"].append(log_prob)
                episode_data["values"].append(value)
                episode_data["old_logits"].append(logits[0, 0].detach())

                episode_reward += reward
                obs = next_obs
                step_count += 1

            if len(episode_data["observations"]) == 0:
                continue

            obs_tensor = torch.tensor(episode_data["observations"], dtype=torch.float32)
            actions_tensor = torch.tensor(episode_data["actions"], dtype=torch.long)
            rewards_tensor = torch.tensor(episode_data["rewards"], dtype=torch.float32)
            masks_tensor = torch.tensor(episode_data["masks"], dtype=torch.float32)
            old_log_probs_tensor = torch.tensor(
                episode_data["log_probs"], dtype=torch.float32
            )
            old_logits_tensor = torch.stack(episode_data["old_logits"], dim=0)
            values_tensor = torch.tensor(episode_data["values"], dtype=torch.float32)

            returns = torch.zeros_like(rewards_tensor)
            cumulative_return = torch.zeros_like(rewards_tensor[0])
            for t in reversed(range(rewards_tensor.size(0))):
                cumulative_return = (
                    cumulative_return * self.gamma * masks_tensor[t] + rewards_tensor[t]
                )
                returns[t] = cumulative_return

            if self.use_off_policy and self.replay_buffer is not None:
                # Off-policy with replay buffer: sample from buffer for update.
                # This allows reusing past experience for multiple updates,
                # reducing communication overhead in distributed training.
                if self.replay_buffer.is_empty():
                    print(
                        f"Episode {episode}: Buffer empty, skipping update. "
                        f"Collecting experience..."
                    )
                    action_loss = 0.0
                    dist_entropy = 0.0
                    kl_loss = 0.0
                    value_loss = 0.0
                else:
                    # Sample a batch from the replay buffer
                    buffer_batch = self.replay_buffer.sample(
                        min(self.num_mini_batch, len(self.replay_buffer))
                    )
                    # Convert buffer batch to tensors
                    obs_buffer = torch.tensor(
                        buffer_batch["obs"], dtype=torch.float32
                    ).unsqueeze(0)
                    actions_buffer = torch.tensor(
                        buffer_batch["action"], dtype=torch.long
                    ).unsqueeze(0)
                    rewards_buffer = torch.tensor(
                        buffer_batch["reward"], dtype=torch.float32
                    ).unsqueeze(0)
                    masks_buffer = torch.tensor(
                        buffer_batch["mask"], dtype=torch.float32
                    ).unsqueeze(0)
                    log_probs_buffer = torch.tensor(
                        buffer_batch["log_prob"], dtype=torch.float32
                    ).unsqueeze(0)
                    # For off-policy, values and returns are not used from buffer
                    # (advantage is computed from reward - baseline)
                    values_buffer = torch.zeros_like(rewards_buffer)
                    returns_buffer = rewards_buffer.clone()
                    old_logits_buffer = torch.zeros_like(
                        torch.tensor(
                            [
                                logits[0, 0].detach().cpu().numpy()
                                for logits in episode_data["old_logits"]
                            ],
                            dtype=torch.float32,
                        ).unsqueeze(0)
                        if episode_data["old_logits"]
                        else torch.zeros(1, 1)
                    )

                    value_loss, action_loss, dist_entropy, kl_loss = self.update(
                        obs_buffer,
                        actions_buffer,
                        rewards_buffer,
                        masks_buffer,
                        log_probs_buffer,
                        values_buffer,
                        returns_buffer,
                        old_logits_buffer,
                    )
            else:
                # On-policy or off-policy without replay buffer: use current
                # episode's data for update. When use_off_policy=True without
                # a buffer, the off-policy clipped surrogate objective is still
                # applied (π_k/α ≈ 1 approximation), providing training stability.
                value_loss, action_loss, dist_entropy, kl_loss = self.update(
                    obs_tensor,
                    actions_tensor,
                    rewards_tensor,
                    masks_tensor,
                    old_log_probs_tensor,
                    values_tensor,
                    returns,
                    old_logits_tensor,
                )

            episode_rewards.append(episode_reward)
            mean_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)) :])

            if mean_reward > best_reward:
                best_reward = mean_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Log buffer size for off-policy mode
            buffer_size = (
                len(self.replay_buffer)
                if self.use_off_policy and self.replay_buffer is not None
                else 0
            )

            if use_wandb:
                log_data = {
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "mean_reward_100": mean_reward,
                    "best_reward": best_reward,
                    "steps": step_count,
                    "value_loss": value_loss,
                    "action_loss": action_loss,
                    "dist_entropy": dist_entropy,
                }
                if self.use_kl_loss:
                    log_data["kl_loss"] = kl_loss
                if self.use_off_policy:
                    log_data["buffer_size"] = buffer_size
                wandb.log(log_data)
            print(
                f"Episode {episode}: reward={episode_reward:.4f}, mean_100={mean_reward:.4f}"
            )
            if self.use_off_policy:
                print(f"  Buffer size: {buffer_size}")

            if (episode + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"ppo_checkpoint_{episode}.pt")
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": self.policy_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_reward": best_reward,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

            if no_improvement_count >= earlystop_last:
                print(f"Early stopping at episode {episode}")
                break

        if use_wandb:
            wandb.finish()
        print("done")


class DPOTrainer(BasePolicyTrainer):
    """Direct Preference Optimization (DPO) trainer for prompt augmentation.

    DPO is a parameter-efficient and computationally efficient approach to optimize
    prompts based on preference pairs. Unlike RLHF which requires a separate reward
    model, DPO directly optimizes the policy using preference data.

    Key differences from RLHF:
    - No explicit reward model training required
    - No critic network needed
    - Single-stage training on offline preference pairs
    - Implicit reward function derived from policy

    For prompt optimization:
    - We collect pairs of augmentations (preferred, rejected)
    - The preferred augmentation has higher reward (from reward_model)
    - We optimize the policy to prefer higher-reward augmentations

    Reference: https://arxiv.org/pdf/2305.18290

    Args:
        policy_model: The policy model to optimize
        env: PromptOptimizationEnv with reward_model and augmenters
        max_episodes: Maximum number of training episodes
        optimizer: PyTorch optimizer for parameter updates
        lr_scheduler: Optional learning rate scheduler
        exploration_module: Optional exploration module (unused for now)
        replay_buffer: Optional replay buffer (unused for now)
        beta: Temperature parameter controlling preference learning strength (default 0.1)
        num_mini_batch: Number of mini-batches per epoch (default 4)
        max_grad_norm: Maximum gradient norm for clipping (default 1.0)
        use_reference_policy: Whether to maintain separate reference policy (default True)
        eps: Epsilon for numerical stability (default 1e-8)
    """

    def __init__(
        self,
        policy_model: BasePolicy,
        env: PromptOptimizationEnv,
        max_episodes: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        exploration_module: Optional[BaseExplorationModule] = None,
        replay_buffer: Optional[BaseReplayBuffer] = None,
        beta: float = 0.1,
        num_mini_batch: int = 4,
        max_grad_norm: float = 1.0,
        use_reference_policy: bool = True,
        eps: float = 1e-8,
        **kwargs,
    ):
        """Initialize DPO trainer.

        Args:
            policy_model: The policy model to optimize
            env: PromptOptimizationEnv with reward_model for evaluating augmentations
            max_episodes: Maximum number of training episodes
            optimizer: PyTorch optimizer
            lr_scheduler: Optional learning rate scheduler
            exploration_module: Optional exploration module (not used currently)
            replay_buffer: Optional replay buffer (not used currently)
            beta: Temperature parameter (0.1-0.5 typical)
            num_mini_batch: Number of mini-batches for gradient updates
            max_grad_norm: Maximum gradient norm for clipping
            use_reference_policy: Whether to keep reference policy frozen
            eps: Numerical stability epsilon
        """
        super().__init__(
            policy_model,
            env,
            max_episodes,
            optimizer,
            lr_scheduler,
            exploration_module,
            replay_buffer,
            eps,
        )
        self.beta = beta
        self.num_mini_batch = num_mini_batch
        self.max_grad_norm = max_grad_norm
        self.use_reference_policy = use_reference_policy
        self.eps = eps

        # Create reference policy as a copy of the policy model if specified
        if self.use_reference_policy:
            self.reference_policy = type(policy_model)(
                policy_model.action_space,
                policy_model.observation_space,
            )
            # Copy state dict from policy model
            self.reference_policy.load_state_dict(policy_model.state_dict())
            # Freeze reference policy parameters
            for param in self.reference_policy.parameters():
                param.requires_grad = False
        else:
            self.reference_policy = None

    def _collect_preference_pairs(
        self, num_pairs: int = 32
    ) -> Tuple[list, list, list, list]:
        """Collect preference pairs by interacting with environment.

        For each base prompt, we try different augmentations and compare rewards
        to create preference pairs (preferred, rejected).

        Args:
            num_pairs: Number of preference pairs to collect

        Returns:
            Tuple of (preferred_prompts, rejected_prompts, preferred_rewards, rejected_rewards)
        """
        preferred_prompts = []
        rejected_prompts = []
        preferred_rewards = []
        rejected_rewards = []

        for _ in range(num_pairs):
            # Reset environment to get a starting prompt
            obs, info = self.env.reset()
            current_prompt = info["current_prompt"]

            # Collect augmentations and their rewards
            augmentation_rewards = []
            augmented_prompts = []

            for action_idx in range(len(self.env._augmenters)):
                augmenter = self.env._augmenters[action_idx]
                try:
                    # Apply augmentation
                    message = AgentMessage(query=current_prompt)
                    result_message = augmenter(message)
                    augmented_prompt = result_message.query

                    # Evaluate reward
                    reward = self.env._reward_model(augmented_prompt)

                    augmentation_rewards.append(reward)
                    augmented_prompts.append(augmented_prompt)
                except Exception:
                    # If augmentation fails, assign very low reward
                    augmentation_rewards.append(-float("inf"))
                    augmented_prompts.append(None)

            # Find preferred (highest reward) and rejected (lowest reward)
            valid_indices = [
                i for i, r in enumerate(augmentation_rewards) if r != -float("inf")
            ]

            if len(valid_indices) >= 2:
                # Sort by reward
                valid_indices_sorted = sorted(
                    valid_indices, key=lambda i: augmentation_rewards[i]
                )

                # Preferred: highest reward
                preferred_idx = valid_indices_sorted[-1]
                # Rejected: lowest reward (or second best if only 2)
                rejected_idx = valid_indices_sorted[0]

                preferred_prompts.append(augmented_prompts[preferred_idx])
                rejected_prompts.append(augmented_prompts[rejected_idx])
                preferred_rewards.append(augmentation_rewards[preferred_idx])
                rejected_rewards.append(augmentation_rewards[rejected_idx])

        return preferred_prompts, rejected_prompts, preferred_rewards, rejected_rewards

    def _form_preference_pairs_from_buffer(
        self, num_pairs: int = 32
    ) -> Tuple[list, list, list, list]:
        """Form preference pairs from replay buffer entries.

        Samples transitions from the replay buffer and groups them by
        similar observations to create preference pairs based on rewards.

        Args:
            num_pairs: Number of preference pairs to form

        Returns:
            Tuple of (preferred_prompts, rejected_prompts, preferred_rewards, rejected_rewards)
        """
        if self.replay_buffer is None or self.replay_buffer.is_empty():
            return [], [], [], []

        # Sample a larger batch to have enough candidates for pairing
        batch_size = min(num_pairs * 4, self.replay_buffer.size)
        entries = self.replay_buffer.sample(batch_size)

        if len(entries) < 2:
            return [], [], [], []

        preferred_prompts = []
        rejected_prompts = []
        preferred_rewards = []
        rejected_rewards = []

        # Group entries by observation similarity (within a threshold)
        # For simplicity, we bucket observations by quantizing the first few dimensions
        obs_dim = entries[0].obs.shape[0]
        num_buckets = min(num_pairs * 2, len(entries) // 2)
        buckets = [[] for _ in range(num_buckets)]

        for entry in entries:
            # Simple hashing based on observation quantization
            bucket_idx = int(np.mean(entry.obs[: min(10, obs_dim)] * 10)) % num_buckets
            buckets[bucket_idx].append(entry)

        # Form pairs within each bucket based on reward difference
        for bucket in buckets:
            if len(bucket) < 2:
                continue

            # Sort by reward
            bucket_sorted = sorted(bucket, key=lambda e: e.reward, reverse=True)

            # Take highest and lowest reward as preferred/rejected pair
            preferred = bucket_sorted[0]
            rejected = bucket_sorted[-1]

            # Only form pair if there's a meaningful reward difference
            if preferred.reward - rejected.reward > 1e-6:
                # Store reward as proxy for prompt (we'll convert to obs later)
                preferred_rewards.append(preferred.reward)
                rejected_rewards.append(rejected.reward)
                # We'll convert to prompts/obs in fit() using the stored observations
                preferred_prompts.append(preferred)
                rejected_prompts.append(rejected)

            if len(preferred_prompts) >= num_pairs:
                break

        return preferred_prompts, rejected_prompts, preferred_rewards, rejected_rewards

    def _compute_dpo_loss(
        self,
        policy_log_probs_w: torch.Tensor,
        policy_log_probs_l: torch.Tensor,
        reference_log_probs_w: torch.Tensor,
        reference_log_probs_l: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DPO loss using the Bradley-Terry model.

        The DPO loss encourages the policy to increase the log probability of
        preferred augmentations while decreasing it for rejected ones.

        Loss = -log(sigma(β * (log_pi_w/pi_ref_w - log_pi_l/pi_ref_l)))

        Args:
            policy_log_probs_w: Log probs from policy for winning (preferred) actions
            policy_log_probs_l: Log probs from policy for losing (rejected) actions
            reference_log_probs_w: Log probs from reference for winning actions
            reference_log_probs_l: Log probs from reference for losing actions

        Returns:
            DPO loss (scalar)
        """
        # Compute log probability ratios (implicit rewards)
        # r_theta = beta * (log(π_theta(y_w|x)) - log(π_ref(y_w|x)))
        log_ratio_w = policy_log_probs_w - reference_log_probs_w
        log_ratio_l = policy_log_probs_l - reference_log_probs_l

        # DPO loss using sigmoid: -log(sigma(beta * (r_w - r_l)))
        log_ratios_diff = self.beta * (log_ratio_w - log_ratio_l)
        losses = -F.logsigmoid(log_ratios_diff)

        return losses.mean()

    def update(
        self,
        preferred_obs: torch.Tensor,
        rejected_obs: torch.Tensor,
        preferred_actions: torch.Tensor,
        rejected_actions: torch.Tensor,
        preferred_masks: torch.Tensor,
        rejected_masks: torch.Tensor,
    ) -> Tuple[float, float]:
        """Perform one update step using DPO loss.

        Args:
            preferred_obs: Observations for preferred prompts
            rejected_obs: Observations for rejected prompts
            preferred_actions: Actions taken for preferred prompts
            rejected_actions: Actions taken for rejected prompts
            preferred_masks: Masks for preferred prompts
            rejected_masks: Masks for rejected prompts

        Returns:
            Tuple of (dpo_loss, implicit_reward_diff)
        """
        dpo_loss_epoch = 0.0
        implicit_reward_diff_epoch = 0.0
        num_updates = 0

        batch_size = preferred_obs.size(0)
        indices = torch.randperm(batch_size)

        for i in range(0, batch_size, self.num_mini_batch):
            batch_indices = indices[i : i + self.num_mini_batch]

            # Get batch data
            pref_obs_batch = preferred_obs[batch_indices]
            rej_obs_batch = rejected_obs[batch_indices]
            pref_actions_batch = preferred_actions[batch_indices]
            rej_actions_batch = rejected_actions[batch_indices]
            pref_masks_batch = preferred_masks[batch_indices]
            rej_masks_batch = rejected_masks[batch_indices]

            # Forward pass for preferred actions through policy
            _, policy_log_probs_w, _, _ = self.policy_model.evaluate_actions(
                pref_obs_batch, pref_actions_batch, pref_masks_batch
            )

            # Forward pass for rejected actions through policy
            _, policy_log_probs_l, _, _ = self.policy_model.evaluate_actions(
                rej_obs_batch, rej_actions_batch, rej_masks_batch
            )

            # Get log probs from reference policy (no gradient)
            if self.use_reference_policy:
                with torch.no_grad():
                    _, reference_log_probs_w, _, _ = (
                        self.reference_policy.evaluate_actions(
                            pref_obs_batch, pref_actions_batch, pref_masks_batch
                        )
                    )
                    _, reference_log_probs_l, _, _ = (
                        self.reference_policy.evaluate_actions(
                            rej_obs_batch, rej_actions_batch, rej_masks_batch
                        )
                    )
            else:
                # If no reference policy, use detached policy logits
                with torch.no_grad():
                    _, reference_log_probs_w, _, _ = self.policy_model.evaluate_actions(
                        pref_obs_batch, pref_actions_batch, pref_masks_batch
                    )
                    _, reference_log_probs_l, _, _ = self.policy_model.evaluate_actions(
                        rej_obs_batch, rej_actions_batch, rej_masks_batch
                    )

            # Compute DPO loss
            dpo_loss = self._compute_dpo_loss(
                policy_log_probs_w,
                policy_log_probs_l,
                reference_log_probs_w,
                reference_log_probs_l,
            )

            # Compute implicit reward difference for logging
            with torch.no_grad():
                implicit_reward_w = self.beta * (
                    policy_log_probs_w - reference_log_probs_w
                )
                implicit_reward_l = self.beta * (
                    policy_log_probs_l - reference_log_probs_l
                )
                implicit_reward_diff = (implicit_reward_w - implicit_reward_l).mean()

            # Backward pass
            self.optimizer.zero_grad()
            dpo_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Step the learning rate scheduler if provided
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            dpo_loss_epoch += dpo_loss.item()
            implicit_reward_diff_epoch += implicit_reward_diff.item()
            num_updates += 1

        return (
            dpo_loss_epoch / max(num_updates, 1),
            implicit_reward_diff_epoch / max(num_updates, 1),
        )

    def fit(
        self,
        checkpoint_every: int = 10,
        earlystop_last: int = 100,
        record_every: int = 1000,
        use_wandb: bool = True,
        wandb_project: str = "prompt_optimization",
        checkpoint_dir: str = "./ckpt",
        pairs_per_episode: int = 32,
        **kwargs,
    ) -> None:
        """Train the policy using DPO on preference pairs.

        DPO training proceeds by:
        1. Collecting preference pairs through environment interaction
        2. Computing log probabilities under policy and reference
        3. Optimizing using the DPO loss (Bradley-Terry model)

        Args:
            checkpoint_every: Save checkpoint every N episodes
            earlystop_last: Early stop if no improvement in N episodes
            record_every: Record metrics every N episodes
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: WandB project name
            checkpoint_dir: Directory to save checkpoints
            pairs_per_episode: Number of preference pairs to collect per episode
            **kwargs: Additional arguments
        """
        # Initialize WandB if enabled
        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    config={
                        "max_episodes": self.max_episodes,
                        "beta": self.beta,
                        "checkpoint_every": checkpoint_every,
                        "earlystop_last": earlystop_last,
                        "pairs_per_episode": pairs_per_episode,
                    },
                )
            except ImportError:
                warnings.warn("wandb not installed, skipping WandB logging")
                use_wandb = False

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load latest checkpoint if exists
        ckpt_pattern = os.path.join(checkpoint_dir, "dpo_checkpoint_*.pt")
        ckpt_files = glob.glob(ckpt_pattern)
        episode_start = 0
        best_reward_diff = -float("inf")
        if ckpt_files:
            episodes = []
            for f in ckpt_files:
                try:
                    ep = int(os.path.basename(f).split("_")[2].split(".")[0])
                    episodes.append(ep)
                except ValueError:
                    pass
            if episodes:
                latest_episode = max(episodes)
                ckpt_path = os.path.join(
                    checkpoint_dir, f"dpo_checkpoint_{latest_episode}.pt"
                )
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                self.policy_model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if self.use_reference_policy:
                    self.reference_policy.load_state_dict(
                        checkpoint["reference_policy_state_dict"]
                    )
                episode_start = latest_episode + 1
                best_reward_diff = checkpoint.get("best_reward_diff", -float("inf"))
                print(f"Loaded checkpoint from episode {latest_episode}")

        # Training loop
        episode_reward_diffs = []
        no_improvement_count = 0

        for episode in range(episode_start, self.max_episodes):
            # Collect or sample preference pairs
            # Use replay buffer if available (off-policy), otherwise collect online
            if (
                self.replay_buffer is not None
                and not self.replay_buffer.is_empty()
                and episode > 0
            ):
                # Off-policy: sample from replay buffer
                (
                    preferred_prompts,
                    rejected_prompts,
                    preferred_rewards,
                    rejected_rewards,
                ) = self._form_preference_pairs_from_buffer(num_pairs=pairs_per_episode)
                use_buffer = True
            else:
                # On-policy: collect fresh pairs from environment
                (
                    preferred_prompts,
                    rejected_prompts,
                    preferred_rewards,
                    rejected_rewards,
                ) = self._collect_preference_pairs(num_pairs=pairs_per_episode)
                use_buffer = False

            if len(preferred_prompts) == 0:
                print(f"Episode {episode}: No valid preference pairs collected")
                continue

            # Convert prompts/entries to observations
            if use_buffer:
                # preferred_prompts/rejected_prompts are ReplayBufferEntry objects
                preferred_obs_list = [e.obs for e in preferred_prompts]
                rejected_obs_list = [e.obs for e in rejected_prompts]
                preferred_actions_list = [e.action for e in preferred_prompts]
                rejected_actions_list = [e.action for e in rejected_prompts]
                preferred_masks_list = [e.mask for e in preferred_prompts]
                rejected_masks_list = [e.mask for e in rejected_prompts]
            else:
                # preferred_prompts/rejected_prompts are prompt strings
                preferred_obs_list = []
                rejected_obs_list = []
                for pref_prompt, rej_prompt in zip(preferred_prompts, rejected_prompts):
                    pref_embedding = self.env._embedding_model(pref_prompt).astype(
                        np.float32
                    )
                    rej_embedding = self.env._embedding_model(rej_prompt).astype(
                        np.float32
                    )
                    preferred_obs_list.append(pref_embedding)
                    rejected_obs_list.append(rej_embedding)
                # For online collection, use action 0 as placeholder
                preferred_actions_list = [0] * len(preferred_prompts)
                rejected_actions_list = [0] * len(rejected_prompts)
                preferred_masks_list = [1.0] * len(preferred_prompts)
                rejected_masks_list = [1.0] * len(rejected_prompts)

            preferred_obs = torch.tensor(
                np.stack(preferred_obs_list, axis=0), dtype=torch.float32
            )
            rejected_obs = torch.tensor(
                np.stack(rejected_obs_list, axis=0), dtype=torch.float32
            )

            # Add sequence dimension for compatibility with policy model
            preferred_obs = preferred_obs.unsqueeze(1)  # (batch, 1, obs_dim)
            rejected_obs = rejected_obs.unsqueeze(1)  # (batch, 1, obs_dim)

            preferred_actions = torch.tensor(
                preferred_actions_list, dtype=torch.long
            ).unsqueeze(-1)
            rejected_actions = torch.tensor(
                rejected_actions_list, dtype=torch.long
            ).unsqueeze(-1)
            preferred_masks = torch.tensor(
                preferred_masks_list, dtype=torch.float32
            ).unsqueeze(-1)
            rejected_masks = torch.tensor(
                rejected_masks_list, dtype=torch.float32
            ).unsqueeze(-1)

            # Update policy using DPO loss
            dpo_loss, reward_diff = self.update(
                preferred_obs,
                rejected_obs,
                preferred_actions,
                rejected_actions,
                preferred_masks,
                rejected_masks,
            )

            # Track metrics
            episode_reward_diffs.append(reward_diff)

            # Check for improvement (early stopping)
            mean_reward_diff = np.mean(
                episode_reward_diffs[-min(100, len(episode_reward_diffs)) :]
            )

            if mean_reward_diff > best_reward_diff:
                best_reward_diff = mean_reward_diff
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Logging
            if use_wandb:
                mean_preferred_reward = float(np.mean(preferred_rewards))
                mean_rejected_reward = float(np.mean(rejected_rewards))
                wandb.log(
                    {
                        "episode": episode,
                        "dpo_loss": dpo_loss,
                        "reward_diff": reward_diff,
                        "mean_reward_diff_100": mean_reward_diff,
                        "best_reward_diff": best_reward_diff,
                        "mean_preferred_reward": mean_preferred_reward,
                        "mean_rejected_reward": mean_rejected_reward,
                        "pairs_collected": len(preferred_prompts),
                    }
                )

            print(
                f"Episode {episode}: loss={dpo_loss:.4f}, reward_diff={reward_diff:.4f}, "
                f"mean_diff_100={mean_reward_diff:.4f}"
            )

            # Checkpointing
            if (episode + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"dpo_checkpoint_{episode}.pt")
                checkpoint_dict = {
                    "episode": episode,
                    "model_state_dict": self.policy_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_reward_diff": best_reward_diff,
                }
                if self.use_reference_policy:
                    checkpoint_dict["reference_policy_state_dict"] = (
                        self.reference_policy.state_dict()
                    )

                torch.save(checkpoint_dict, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            # Early stopping
            if no_improvement_count >= earlystop_last:
                print(f"Early stopping at episode {episode}")
                break

        if use_wandb:
            wandb.finish()
        print("done")
