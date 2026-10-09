"""Test cases for exploration modules in policy training."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.optim.lr_scheduler
from aap_core.policy import BasePolicy
from aap_core.policy_trainer import (
    BaseExplorationModule,
    EpsilonGreedyExploration,
    RandomExploration,
)
from aap_core.prompt_augmenter import IdentityPromptAugmenter, PromptOptimizationEnv
from gymnasium import spaces


class SimplePolicy(BasePolicy):
    """Simple policy model for testing."""

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_space: spaces.Box,
    ):
        super().__init__(action_space, observation_space)
        self.action_dim = int(action_space.n)
        self.obs_dim = observation_space.shape[0]
        self.fc = nn.Linear(self.obs_dim, self.action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy."""
        batch_size, seq_len, _ = obs.shape
        logits = self.fc(obs.view(-1, self.obs_dim))
        return logits.view(batch_size, seq_len, self.action_dim)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
    ):
        """Evaluate actions and return values."""
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        if masks.ndim == 1:
            masks = masks.unsqueeze(-1)
        logits = self.forward(obs)
        batch_size, seq_len, _ = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        values = torch.zeros(batch_size, seq_len, device=obs.device)
        return values, action_log_probs, entropy, logits


class TestRandomExploration:
    """Test RandomExploration class."""

    def test_default_explore_ratio(self):
        """Test default explore ratio is 0.0."""
        exploration = RandomExploration()
        assert exploration.explore_ratio == 0.0

    def test_custom_explore_ratio(self):
        """Test custom explore ratio."""
        exploration = RandomExploration(explore_ratio=0.8)
        assert exploration.explore_ratio == 0.8

    def test_should_explore_returns_bool(self):
        """Test should_explore returns boolean."""
        exploration = RandomExploration()
        result = exploration.should_explore(step=0, episode=0)
        assert isinstance(result, bool)

    def test_should_explore_with_zero_ratio(self):
        """Test should_explore with 0 ratio always returns False."""
        exploration = RandomExploration(explore_ratio=0.0)
        for _ in range(100):
            assert exploration.should_explore(step=0, episode=0) is False

    def test_should_explore_with_one_ratio(self):
        """Test should_explore with 1 ratio always returns True."""
        exploration = RandomExploration(explore_ratio=1.0)
        for _ in range(100):
            assert exploration.should_explore(step=0, episode=0) is True

    def test_should_explore_ratio_distribution(self):
        """Test that explore ratio produces expected distribution."""
        torch.manual_seed(42)
        exploration = RandomExploration(explore_ratio=0.5)
        num_samples = 1000
        explore_count = sum(
            exploration.should_explore(step=i, episode=0) for i in range(num_samples)
        )
        ratio = explore_count / num_samples
        assert 0.4 < ratio < 0.6, f"Expected ratio ~0.5, got {ratio}"

    def test_get_exploration_config(self):
        """Test get_exploration_config returns correct dict."""
        exploration = RandomExploration(explore_ratio=0.7)
        config = exploration.get_exploration_config()
        assert isinstance(config, dict)
        assert config["explore_ratio"] == 0.7

    def test_reset_is_noop(self):
        """Test reset method is a no-op for RandomExploration."""
        exploration = RandomExploration()
        # Should not raise
        exploration.reset(episode=0)
        exploration.reset(episode=100)


class TestEpsilonGreedyExploration:
    """Test EpsilonGreedyExploration class."""

    def test_default_params(self):
        """Test default parameters."""
        exploration = EpsilonGreedyExploration()
        assert exploration.eps_init == 1.0
        assert exploration.eps_final == 0.05
        assert exploration.decay_episodes == 1000
        assert exploration.current_eps == 1.0

    def test_custom_params(self):
        """Test custom parameters."""
        exploration = EpsilonGreedyExploration(
            eps_init=0.9, eps_final=0.01, decay_episodes=500, min_eps=0.005
        )
        assert exploration.eps_init == 0.9
        assert exploration.eps_final == 0.01
        assert exploration.decay_episodes == 500

    def test_should_explore_returns_bool(self):
        """Test should_explore returns boolean."""
        exploration = EpsilonGreedyExploration()
        result = exploration.should_explore(step=0, episode=0)
        assert isinstance(result, bool)

    def test_epsilon_decay(self):
        """Test epsilon decays over episodes."""
        exploration = EpsilonGreedyExploration(
            eps_init=1.0, eps_final=0.05, decay_episodes=100
        )
        # At episode 0, epsilon should be eps_init
        exploration.should_explore(step=0, episode=0)
        assert exploration.current_eps == 1.0

        # At episode 50 (halfway), epsilon should be ~0.525
        exploration.should_explore(step=0, episode=50)
        expected = 0.05 + (1.0 - 0.05) * 0.5
        assert abs(exploration.current_eps - expected) < 0.01

        # At episode 100 (fully decayed), epsilon should be eps_final
        exploration.should_explore(step=0, episode=100)
        assert exploration.current_eps == 0.05

        # At episode 200 (past decay), epsilon should stay at eps_final
        exploration.should_explore(step=0, episode=200)
        assert exploration.current_eps == 0.05

    def test_epsilon_never_below_final(self):
        """Test epsilon never goes below eps_final."""
        exploration = EpsilonGreedyExploration(
            eps_init=0.5, eps_final=0.1, decay_episodes=100
        )
        for episode in range(1000):
            exploration.should_explore(step=0, episode=episode)
            assert exploration.current_eps >= 0.1

    def test_get_exploration_config(self):
        """Test get_exploration_config returns correct dict."""
        exploration = EpsilonGreedyExploration(
            eps_init=0.8, eps_final=0.02, decay_episodes=500
        )
        config = exploration.get_exploration_config()
        assert isinstance(config, dict)
        assert config["eps_init"] == 0.8
        assert config["eps_final"] == 0.02
        assert config["decay_episodes"] == 500
        assert "current_eps" in config

    def test_min_eps_floor(self):
        """Test min_eps parameter sets a floor."""
        exploration = EpsilonGreedyExploration(
            eps_init=0.5, eps_final=0.01, decay_episodes=100, min_eps=0.05
        )
        # eps_final should be clamped to min_eps
        assert exploration.eps_final == 0.05
        for episode in range(1000):
            exploration.should_explore(step=0, episode=episode)
            assert exploration.current_eps >= 0.05


class TestBaseExplorationModule:
    """Test BaseExplorationModule abstract class."""

    def test_cannot_instantiate_abstract(self):
        """Test that BaseExplorationModule cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseExplorationModule()  # type: ignore

    def test_custom_subclass_requires_methods(self):
        """Test that custom subclasses must implement abstract methods."""

        class IncompleteExploration(BaseExplorationModule):
            def should_explore(self, step: int, episode: int, **kwargs) -> bool:
                return True

        with pytest.raises(TypeError):
            IncompleteExploration()  # type: ignore

    def test_custom_subclass_with_all_methods(self):
        """Test that custom subclass with all methods can be instantiated."""

        class CompleteExploration(BaseExplorationModule):
            def should_explore(self, step: int, episode: int, **kwargs) -> bool:
                return True

            def get_exploration_config(self) -> dict:
                return {"strategy": "test"}

        exploration = CompleteExploration()
        assert exploration.should_explore(0, 0) is True
        assert exploration.get_exploration_config() == {"strategy": "test"}
        exploration.reset(0)  # Should not raise


class TestExplorationIntegration:
    """Test exploration module integration with BasePolicyTrainer."""

    def test_trainer_auto_instantiates_random_exploration(self):
        """Test that trainer auto-instantiates RandomExploration when None."""
        from aap_core.policy_trainer import ReinforcePPTrainer

        action_space = spaces.Discrete(5)
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = SimplePolicy(action_space, observation_space)
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=5,
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # No exploration module provided
        trainer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=10,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            exploration_module=None,
        )

        assert isinstance(trainer.exploration_module, RandomExploration)
        assert trainer.exploration_module.explore_ratio == 0.0

    def test_trainer_uses_provided_exploration_module(self):
        """Test that trainer uses provided exploration module."""
        from aap_core.policy_trainer import ReinforcePPTrainer

        action_space = spaces.Discrete(5)
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = SimplePolicy(action_space, observation_space)
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=5,
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        custom_exploration = EpsilonGreedyExploration(eps_init=0.8, eps_final=0.1)
        trainer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=10,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            exploration_module=custom_exploration,
        )

        assert trainer.exploration_module is custom_exploration
        assert isinstance(trainer.exploration_module, EpsilonGreedyExploration)

    def test_exploration_module_reset_called(self):
        """Test that reset is called at episode start."""
        from aap_core.policy_trainer import ReinforcePPTrainer

        action_space = spaces.Discrete(1)  # IdentityPromptAugmenter = 1 action
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = SimplePolicy(action_space, observation_space)
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=3,
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        class TrackingExploration(RandomExploration):
            """Exploration module that tracks reset calls."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.reset_calls = []

            def reset(self, episode: int) -> None:
                self.reset_calls.append(episode)

        tracking_exploration = TrackingExploration(explore_ratio=0.0)
        trainer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=3,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            exploration_module=tracking_exploration,
        )

        # Run a few episodes
        trainer.fit(checkpoint_every=100, earlystop_last=10, use_wandb=False)

        # Check that reset was called for each episode
        assert len(tracking_exploration.reset_calls) == 3
        assert tracking_exploration.reset_calls == [0, 1, 2]

    def test_exploration_affects_action_selection(self):
        """Test that exploration module affects action selection in rollout."""
        from aap_core.policy_trainer import ReinforcePPTrainer

        action_space = spaces.Discrete(1)  # IdentityPromptAugmenter = 1 action
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Create a policy that outputs uniform logits (all actions equally likely)
        class UniformPolicy(BasePolicy):
            def __init__(self, action_space, observation_space):
                super().__init__(action_space, observation_space)
                self.action_dim = int(action_space.n)
                self.obs_dim = observation_space.shape[0]
                # Add a learnable parameter so optimizer can be created
                self.dummy_param = nn.Parameter(torch.ones(1))

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                batch_size, seq_len, _ = obs.shape
                # Use dummy_param so gradients flow through the model
                return (
                    torch.zeros(batch_size, seq_len, self.action_dim)
                    + self.dummy_param * 0
                )

            def evaluate_actions(
                self, obs: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor
            ):
                if actions.ndim == 1:
                    actions = actions.unsqueeze(-1)
                if masks.ndim == 1:
                    masks = masks.unsqueeze(-1)
                logits = self.forward(obs)
                batch_size, seq_len, _ = logits.shape
                log_probs = F.log_softmax(logits, dim=-1)
                action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(
                    -1
                )
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
                values = torch.zeros(batch_size, seq_len, device=obs.device)
                return values, action_log_probs, entropy, logits

        policy = UniformPolicy(action_space, observation_space)
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=3,
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # With 0% exploration
        torch.manual_seed(42)
        exploration = RandomExploration(explore_ratio=0.0)
        trainer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            exploration_module=exploration,
        )

        # Run training - should complete without error
        trainer.fit(checkpoint_every=100, earlystop_last=10, use_wandb=False)

        # If we got here, the exploration module worked correctly
        assert trainer.max_episodes == 1
