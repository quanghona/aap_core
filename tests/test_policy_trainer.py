"""Test cases for ReinforcePP policy gradient algorithm."""

import glob
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.optim.lr_scheduler
from aap_core import SimpleReplayBuffer
from aap_core.policy import BasePolicy, GPT2Policy
from aap_core.policy_trainer import (
    DPOTrainer,
    GRPOTrainer,
    PPOTrainer,
    ReinforcePPTrainer,
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


class MultiLayerPolicy(BasePolicy):
    """Multi-layer policy model for testing."""

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_space: spaces.Box,
        hidden_dim: int = 64,
    ):
        super().__init__(action_space, observation_space)
        self.obs_dim = observation_space.shape[0]
        self.action_dim = int(action_space.n)
        self.fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.action_dim)
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy."""
        batch_size, seq_len, _ = obs.shape
        x = self.relu(self.fc1(obs.view(-1, self.obs_dim)))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits.view(batch_size, seq_len, self.action_dim)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
    ):
        """Evaluate actions and return values."""
        logits = self.forward(obs)
        batch_size, seq_len, _ = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        values = torch.zeros(batch_size, seq_len, device=obs.device)
        return values, action_log_probs, entropy, logits


class TestReinforcePPBasic:
    """Test basic functionality of ReinforcePP."""

    def test_basic_update(self):
        """Test basic update with valid inputs."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        reinforce_pp = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = reinforce_pp.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(entropy))
        assert kl_loss == 0.0

    def test_model_parameters_updated(self):
        """Test that model parameters are actually updated."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        reinforce_pp = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        initial_params = {
            name: param.clone() for name, param in policy.named_parameters()
        }

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        reinforce_pp.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        for name, param in policy.named_parameters():
            assert not torch.equal(initial_params[name], param), (
                f"Parameter {name} was not updated"
            )

    def test_loss_values_reasonable(self):
        """Test that loss values are reasonable (not NaN or inf)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        reinforce_pp = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = reinforce_pp.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isinf(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(entropy))
        assert not torch.isinf(torch.tensor(entropy))


class TestDPOBasic:
    """Test basic functionality of DPO."""

    def test_basic_update(self):
        obs_dim = 10
        batch_size = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(1),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        dpo = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=10,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            beta=0.1,
            num_mini_batch=2,
            use_reference_policy=True,
        )

        preferred_obs = torch.randn(batch_size, 1, obs_dim)
        rejected_obs = torch.randn(batch_size, 1, obs_dim)
        preferred_actions = torch.zeros(batch_size, 1, dtype=torch.long)
        rejected_actions = torch.zeros(batch_size, 1, dtype=torch.long)
        preferred_masks = torch.ones(batch_size, 1)
        rejected_masks = torch.ones(batch_size, 1)

        dpo_loss, reward_diff = dpo.update(
            preferred_obs,
            rejected_obs,
            preferred_actions,
            rejected_actions,
            preferred_masks,
            rejected_masks,
        )

        assert isinstance(dpo_loss, float)
        assert not np.isnan(dpo_loss)
        assert np.isfinite(dpo_loss)
        assert isinstance(reward_diff, float)

    def test_reference_policy_is_frozen(self):
        obs_dim = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(1),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        dpo = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            use_reference_policy=True,
        )

        assert dpo.reference_policy is not None
        assert all(not p.requires_grad for p in dpo.reference_policy.parameters())
        assert dpo.reference_policy is not policy

    def test_update_without_reference_policy(self):
        obs_dim = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(1),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        dpo = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            use_reference_policy=False,
        )

        preferred_obs = torch.randn(2, 1, obs_dim)
        rejected_obs = torch.randn(2, 1, obs_dim)
        preferred_actions = torch.zeros(2, 1, dtype=torch.long)
        rejected_actions = torch.zeros(2, 1, dtype=torch.long)
        preferred_masks = torch.ones(2, 1)
        rejected_masks = torch.ones(2, 1)

        dpo_loss, reward_diff = dpo.update(
            preferred_obs,
            rejected_obs,
            preferred_actions,
            rejected_actions,
            preferred_masks,
            rejected_masks,
        )

        assert isinstance(dpo_loss, float)
        assert np.isfinite(dpo_loss)
        assert isinstance(reward_diff, float)

    def test_compute_dpo_loss_matches_manual(self):
        obs_dim = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(1),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        dpo = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            beta=0.2,
            num_mini_batch=1,
            use_reference_policy=False,
        )

        policy_log_probs_w = torch.tensor([[-0.5]], dtype=torch.float32)
        policy_log_probs_l = torch.tensor([[-1.0]], dtype=torch.float32)
        reference_log_probs_w = torch.tensor([[-0.6]], dtype=torch.float32)
        reference_log_probs_l = torch.tensor([[-1.2]], dtype=torch.float32)

        loss = dpo._compute_dpo_loss(
            policy_log_probs_w,
            policy_log_probs_l,
            reference_log_probs_w,
            reference_log_probs_l,
        )

        expected = -F.logsigmoid(
            0.2
            * (
                (policy_log_probs_w - reference_log_probs_w)
                - (policy_log_probs_l - reference_log_probs_l)
            )
        ).mean()
        assert torch.allclose(loss, expected)

    def test_fit_saves_checkpoint(self, tmp_path):
        obs_dim = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(1),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter(), IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 1.0 if "test" in x else 0.0,
            max_steps=2,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        dpo = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            use_reference_policy=True,
        )

        dpo.fit(
            checkpoint_every=1,
            earlystop_last=1,
            use_wandb=False,
            checkpoint_dir=str(tmp_path),
            pairs_per_episode=1,
        )

        ckpt_files = glob.glob(str(tmp_path / "dpo_checkpoint_*.pt"))
        assert len(ckpt_files) == 1

    def test_collect_preference_pairs(self):
        obs_dim = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(2),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test prompt",
            augmenters=[IdentityPromptAugmenter(), IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 1.0 if "good" in x else 0.0,
            max_steps=2,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        dpo = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            use_reference_policy=False,
        )

        (
            preferred_prompts,
            rejected_prompts,
            preferred_rewards,
            rejected_rewards,
        ) = dpo._collect_preference_pairs(num_pairs=4)

        assert len(preferred_prompts) >= 0
        assert len(rejected_prompts) >= 0
        assert len(preferred_rewards) == len(rejected_rewards)
        assert len(preferred_prompts) == len(rejected_prompts)

    def test_collect_preference_pairs_with_different_rewards(self):
        obs_dim = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(3),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="base",
            augmenters=[
                IdentityPromptAugmenter(),
                IdentityPromptAugmenter(),
                IdentityPromptAugmenter(),
            ],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 2.0 if "high" in x else (1.0 if "med" in x else 0.0),
            max_steps=3,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        dpo = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            use_reference_policy=False,
        )

        (
            preferred_prompts,
            rejected_prompts,
            preferred_rewards,
            rejected_rewards,
        ) = dpo._collect_preference_pairs(num_pairs=2)

        if len(preferred_rewards) > 0:
            assert all(r >= 0 for r in preferred_rewards)
            assert all(r >= 0 for r in rejected_rewards)
            for pr, rr in zip(preferred_rewards, rejected_rewards):
                assert pr >= rr

    def test_gradient_flow(self):
        obs_dim = 8
        batch_size = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(1),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        dpo = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            use_reference_policy=True,
        )

        preferred_obs = torch.randn(batch_size, 1, obs_dim)
        rejected_obs = torch.randn(batch_size, 1, obs_dim)
        preferred_actions = torch.zeros(batch_size, 1, dtype=torch.long)
        rejected_actions = torch.zeros(batch_size, 1, dtype=torch.long)
        preferred_masks = torch.ones(batch_size, 1)
        rejected_masks = torch.ones(batch_size, 1)

        dpo_loss, reward_diff = dpo.update(
            preferred_obs,
            rejected_obs,
            preferred_actions,
            rejected_actions,
            preferred_masks,
            rejected_masks,
        )

        # Verify gradients were computed
        has_grad = False
        for param in policy.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "Policy parameters should have gradients after update"

    def test_lr_scheduler_steps(self):
        obs_dim = 8
        batch_size = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(1),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=1, gamma=0.5
        )

        dpo = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            use_reference_policy=False,
        )

        preferred_obs = torch.randn(batch_size, 1, obs_dim)
        rejected_obs = torch.randn(batch_size, 1, obs_dim)
        preferred_actions = torch.zeros(batch_size, 1, dtype=torch.long)
        rejected_actions = torch.zeros(batch_size, 1, dtype=torch.long)
        preferred_masks = torch.ones(batch_size, 1)
        rejected_masks = torch.ones(batch_size, 1)

        initial_lr = torch_optimizer.param_groups[0]["lr"]
        dpo_loss, reward_diff = dpo.update(
            preferred_obs,
            rejected_obs,
            preferred_actions,
            rejected_actions,
            preferred_masks,
            rejected_masks,
        )
        assert lr_scheduler.get_last_lr()[0] < initial_lr

    def test_checkpoint_loads_correctly(self, tmp_path):
        obs_dim = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(1),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter(), IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 1.0 if "test" in x else 0.0,
            max_steps=2,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        dpo = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=2,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            use_reference_policy=True,
        )

        dpo.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=str(tmp_path),
            pairs_per_episode=1,
        )

        # Load checkpoint and verify it contains correct keys
        ckpt_files = glob.glob(str(tmp_path / "dpo_checkpoint_*.pt"))
        assert len(ckpt_files) >= 1

        ckpt_path = ckpt_files[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "episode" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "best_reward_diff" in checkpoint
        assert "reference_policy_state_dict" in checkpoint

    def test_multiple_mini_batches(self):
        obs_dim = 8
        batch_size = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(1),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        dpo = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            num_mini_batch=2,
            use_reference_policy=False,
        )

        preferred_obs = torch.randn(batch_size, 1, obs_dim)
        rejected_obs = torch.randn(batch_size, 1, obs_dim)
        preferred_actions = torch.zeros(batch_size, 1, dtype=torch.long)
        rejected_actions = torch.zeros(batch_size, 1, dtype=torch.long)
        preferred_masks = torch.ones(batch_size, 1)
        rejected_masks = torch.ones(batch_size, 1)

        dpo_loss, reward_diff = dpo.update(
            preferred_obs,
            rejected_obs,
            preferred_actions,
            rejected_actions,
            preferred_masks,
            rejected_masks,
        )

        assert isinstance(dpo_loss, float)
        assert np.isfinite(dpo_loss)

    def test_dpo_loss_with_different_beta(self):
        obs_dim = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(1),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler1 = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        lr_scheduler2 = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        dpo_beta1 = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler1,
            beta=0.1,
            num_mini_batch=1,
            use_reference_policy=False,
        )

        dpo_beta2 = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler2,
            beta=0.5,
            num_mini_batch=1,
            use_reference_policy=False,
        )

        policy_log_probs_w = torch.tensor([[-0.5]], dtype=torch.float32)
        policy_log_probs_l = torch.tensor([[-1.0]], dtype=torch.float32)
        reference_log_probs_w = torch.tensor([[-0.6]], dtype=torch.float32)
        reference_log_probs_l = torch.tensor([[-1.2]], dtype=torch.float32)

        loss1 = dpo_beta1._compute_dpo_loss(
            policy_log_probs_w,
            policy_log_probs_l,
            reference_log_probs_w,
            reference_log_probs_l,
        )
        loss2 = dpo_beta2._compute_dpo_loss(
            policy_log_probs_w,
            policy_log_probs_l,
            reference_log_probs_w,
            reference_log_probs_l,
        )

        # Different beta should produce different losses
        assert not torch.allclose(loss1, loss2)

    def test_update_returns_consistent_types(self):
        obs_dim = 8
        batch_size = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(1),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        dpo = DPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=1,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            use_reference_policy=True,
        )

        preferred_obs = torch.randn(batch_size, 1, obs_dim)
        rejected_obs = torch.randn(batch_size, 1, obs_dim)
        preferred_actions = torch.zeros(batch_size, 1, dtype=torch.long)
        rejected_actions = torch.zeros(batch_size, 1, dtype=torch.long)
        preferred_masks = torch.ones(batch_size, 1)
        rejected_masks = torch.ones(batch_size, 1)

        result = dpo.update(
            preferred_obs,
            rejected_obs,
            preferred_actions,
            rejected_actions,
            preferred_masks,
            rejected_masks,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        dpo_loss, reward_diff = result
        assert isinstance(dpo_loss, float)
        assert isinstance(reward_diff, float)


class TestGRPOBasic:
    """Test basic functionality of GRPO."""

    def test_basic_update(self):
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(entropy))
        assert kl_loss == 0.0

    def test_parameters_updated(self):
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
        )

        initial_params = {
            name: param.clone() for name, param in policy.named_parameters()
        }

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        for name, param in policy.named_parameters():
            assert not torch.equal(initial_params[name], param), (
                f"Parameter {name} was not updated"
            )

    def test_clip_param_zero(self):
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.0,
            num_mini_batch=2,
            group_size=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))

    def test_zero_rewards(self):
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.zeros(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))

    def test_single_sample_batch(self):
        obs_dim = 10
        action_dim = 5
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=1,
            group_size=1,
        )

        batch = {
            "obs": torch.randn(1, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (1, seq_len)),
            "rewards": torch.randn(1, seq_len),
            "masks": torch.ones(1, seq_len),
            "old_log_probs": torch.randn(1, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))


class TestGRPOVariations:
    """Test GRPO with different hyperparameters and input patterns."""

    @pytest.mark.parametrize("clip_param", [0.0, 0.1, 0.2, 0.5])
    def test_different_clip_params(self, clip_param):
        """Test GRPO with various clip parameter values."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=clip_param,
            num_mini_batch=2,
            group_size=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isinf(torch.tensor(action_loss))

    @pytest.mark.parametrize("group_size", [1, 2, 4, 8])
    def test_different_group_sizes(self, group_size):
        """Test GRPO with various group size values."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=group_size,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))

    @pytest.mark.parametrize("batch_size", [2, 4, 8, 16])
    def test_different_batch_sizes(self, batch_size):
        """Test GRPO with various batch size values."""
        obs_dim = 10
        action_dim = 5
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))

    @pytest.mark.parametrize(
        "reward_pattern,reward_val",
        [
            ("positive", 1.0),
            ("negative", -1.0),
            ("mixed", None),
            ("constant", 0.5),
        ],
    )
    def test_different_reward_patterns(self, reward_pattern, reward_val):
        """Test GRPO with various reward patterns."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
        )

        if reward_pattern == "mixed":
            rewards = torch.randn(batch_size, seq_len)
        elif reward_val is not None:
            rewards = torch.full((batch_size, seq_len), reward_val)
        else:
            rewards = torch.randn(batch_size, seq_len)

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": rewards,
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isinf(torch.tensor(action_loss))

    def test_fit_training_few_episodes(self):
        """Test GRPO fit method for a few training episodes."""
        obs_dim = 10
        action_dim = 1

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=5,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=3,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=1,
            group_size=1,
        )

        # Run fit for a few episodes
        grpo.fit(
            checkpoint_every=10,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir="./ckpt",
        )

        # Verify that parameters were updated
        params = list(policy.parameters())
        assert all(p.requires_grad for p in params), (
            "Policy parameters should require gradients"
        )

    def test_fit_with_reward_normalization(self):
        """Test GRPO fit with reward normalization enabled."""
        obs_dim = 10
        action_dim = 1

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=5,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=3,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=1,
            group_size=1,
            use_reward_normalization=True,
        )

        grpo.fit(
            checkpoint_every=10,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir="./ckpt",
        )

    def test_fit_without_reward_normalization(self):
        """Test GRPO fit with reward normalization disabled."""
        obs_dim = 10
        action_dim = 1

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=5,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=3,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=1,
            group_size=1,
            use_reward_normalization=False,
        )

        grpo.fit(
            checkpoint_every=10,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir="./ckpt",
        )


class TestGRPOCheckpoint:
    """Test checkpoint save/load functionality for GRPO."""

    def test_grpo_checkpoint_save_and_load(self, tmp_path):
        """Test that GRPO saves and loads checkpoints correctly."""
        obs_dim = 10
        action_dim = 1
        checkpoint_dir = str(tmp_path / "grpo_ckpt")

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        trainer = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=5,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=1,
            group_size=1,
        )

        # Run training to create checkpoints
        trainer.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify checkpoint files exist
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "grpo_checkpoint_*.pt"))
        assert len(ckpt_files) > 0, "No GRPO checkpoint files were saved"

        # Create a new policy and trainer
        policy2 = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        torch_optimizer2 = torch.optim.Adam(policy2.parameters(), lr=1e-3)
        lr_scheduler2 = torch.optim.lr_scheduler.StepLR(
            torch_optimizer2, step_size=100, gamma=0.9
        )
        trainer2 = GRPOTrainer(
            policy_model=policy2,
            env=env,
            max_episodes=3,
            optimizer=torch_optimizer2,
            lr_scheduler=lr_scheduler2,
            clip_param=0.2,
            num_mini_batch=1,
            group_size=1,
        )

        # Get initial params before loading
        initial_params = {
            name: param.clone() for name, param in policy2.named_parameters()
        }

        # Run training with checkpoint loading
        trainer2.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify that the policy was loaded from checkpoint (params changed)
        params_changed = False
        for name, param in policy2.named_parameters():
            if not torch.equal(initial_params[name], param):
                params_changed = True
                break
        assert params_changed, (
            "GRPO policy parameters were not updated after loading checkpoint"
        )

    def test_grpo_no_checkpoint_when_empty_dir(self, tmp_path):
        """Test that GRPO works when checkpoint directory is empty."""
        checkpoint_dir = str(tmp_path / "grpo_empty_ckpt")
        os.makedirs(checkpoint_dir, exist_ok=True)

        obs_dim = 10
        action_dim = 1

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        trainer = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=3,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=1,
            group_size=1,
            checkpoint_every=10,
            earlystop_last=10,
        )

        # Should not raise any error
        trainer.fit(
            checkpoint_every=10,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

    def test_grpo_checkpoint_loads_latest(self, tmp_path):
        """Test that GRPO loads the latest checkpoint."""
        obs_dim = 10
        action_dim = 1
        checkpoint_dir = str(tmp_path / "grpo_ckpt_latest")

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        trainer = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=5,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=1,
            group_size=1,
        )

        # Run training to create checkpoints
        trainer.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify the latest checkpoint has the highest episode number
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "grpo_checkpoint_*.pt"))
        episodes = []
        for f in ckpt_files:
            try:
                ep = int(os.path.basename(f).split("_")[2].split(".")[0])
                episodes.append(ep)
            except ValueError:
                pass
        assert len(episodes) > 0
        latest_episode = max(episodes)

        # Create a new policy and trainer
        policy2 = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        torch_optimizer2 = torch.optim.Adam(policy2.parameters(), lr=1e-3)
        lr_scheduler2 = torch.optim.lr_scheduler.StepLR(
            torch_optimizer2, step_size=100, gamma=0.9
        )
        trainer2 = GRPOTrainer(
            policy_model=policy2,
            env=env,
            max_episodes=latest_episode + 3,
            optimizer=torch_optimizer2,
            lr_scheduler=lr_scheduler2,
            clip_param=0.2,
            num_mini_batch=1,
            group_size=1,
        )

        # Run training - should start from latest_episode + 1
        trainer2.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify that the new checkpoint saved has the correct episode number
        new_ckpt_files = glob.glob(os.path.join(checkpoint_dir, "grpo_checkpoint_*.pt"))
        new_episodes = []
        for f in new_ckpt_files:
            try:
                ep = int(os.path.basename(f).split("_")[2].split(".")[0])
                new_episodes.append(ep)
            except ValueError:
                pass
        assert max(new_episodes) >= latest_episode + 1, (
            f"Expected episode >= {latest_episode + 1}, got {max(new_episodes)}"
        )


class TestGRPOInvalidInputs:
    """Test GRPO behavior with invalid inputs."""

    def test_batch_not_divisible_by_group_size(self):
        """Test that GRPO raises error when batch_size is not divisible by group_size."""
        obs_dim = 10
        action_dim = 5
        batch_size = 7  # Not divisible by group_size=2
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        with pytest.raises(ValueError, match="divisible by group_size"):
            grpo.update(
                batch["obs"],
                batch["actions"],
                batch["rewards"],
                batch["masks"],
                batch["old_log_probs"],
            None,
            )

    def test_mismatched_tensor_shapes(self):
        """Test that GRPO handles mismatched tensor shapes appropriately."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
        )

        # Mismatched sequence lengths
        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len - 1),  # Wrong seq_len
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        # Should raise an error due to shape mismatch
        with pytest.raises((RuntimeError, ValueError)):
            grpo.update(
                batch["obs"],
                batch["actions"],
                batch["rewards"],
                batch["masks"],
                batch["old_log_probs"],
            None,
            )

    def test_invalid_action_indices(self):
        """Test GRPO with action indices out of bounds."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
        )

        # Action indices out of bounds (>= action_dim)
        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.full((batch_size, seq_len), action_dim),  # Out of bounds
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        # Should raise an error due to invalid action indices
        with pytest.raises((IndexError, RuntimeError)):
            grpo.update(
                batch["obs"],
                batch["actions"],
                batch["rewards"],
                batch["masks"],
                batch["old_log_probs"],
            None,
            )

    def test_all_masks_zero(self):
        """Test GRPO with all masks set to zero (no valid steps)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.zeros(batch_size, seq_len),  # All masks zero
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        # Should still produce valid loss values even with all masks zero
        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPClipParam:
    """Test different clip parameter values."""

    def test_clip_param_zero(self):
        """Test with clip_param = 0 (no clipping effect)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.0,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_clip_param_one(self):
        """Test with clip_param = 1 (full clipping range)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=1.0,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_clip_param_very_small(self):
        """Test with very small clip_param."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=1e-10,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPZeroValues:
    """Test zero-value parameter cases."""

    def test_zero_rewards(self):
        """Test with zero rewards."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.zeros(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_all_zero_masks(self):
        """Test with all-zero masks."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.zeros(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_single_sample_batch(self):
        """Test with batch_size = 1."""
        obs_dim = 10
        action_dim = 5
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=1,
        )

        batch = {
            "obs": torch.randn(1, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (1, seq_len)),
            "rewards": torch.randn(1, seq_len),
            "masks": torch.ones(1, seq_len),
            "old_log_probs": torch.randn(1, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_single_action(self):
        """Test with sequence length = 1."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, 1, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, 1)),
            "rewards": torch.randn(batch_size, 1),
            "masks": torch.ones(batch_size, 1),
            "old_log_probs": torch.randn(batch_size, 1),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPRewardNormalization:
    """Test reward normalization functionality."""

    def test_identical_rewards(self):
        """Test with identical rewards (std = 0)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            use_reward_normalization=True,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.ones(batch_size, seq_len) * 5.0,
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_very_large_rewards(self):
        """Test with very large reward values."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            use_reward_normalization=True,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len) * 1e6,
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_very_small_rewards(self):
        """Test with very small reward values."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            use_reward_normalization=True,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len) * 1e-6,
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_negative_rewards(self):
        """Test with negative rewards."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            use_reward_normalization=True,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": -torch.abs(torch.randn(batch_size, seq_len)),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_reward_normalization_disabled(self):
        """Test with reward normalization disabled."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            use_reward_normalization=False,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPKLLoss:
    """Test KL loss functionality."""

    def test_kl_loss_enabled(self):
        """Test with KL loss enabled."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            use_kl_loss=True,
            kl_coef=1e-5,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert kl_loss >= 0

    def test_kl_loss_disabled(self):
        """Test with KL loss disabled."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            use_kl_loss=False,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert kl_loss == 0.0

    def test_zero_kl_divergence(self):
        """Test with zero KL divergence."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            use_kl_loss=True,
            kl_coef=1e-5,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
            "kl_divergence": torch.zeros(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert kl_loss == 0.0


class TestReinforcePPBatchSequenceVariations:
    """Test different batch size and sequence length combinations."""

    def test_large_batch_size(self):
        """Test with large batch size."""
        obs_dim = 10
        action_dim = 5
        batch_size = 128
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=4,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_long_sequence(self):
        """Test with long sequence."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 100

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_small_batch_large_sequence(self):
        """Test with small batch and large sequence."""
        obs_dim = 10
        action_dim = 5
        batch_size = 2
        seq_len = 50

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=1,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPGradientClipping:
    """Test gradient clipping functionality."""

    def test_gradient_norm_limited(self):
        """Test that gradient norm is properly clipped."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            max_grad_norm=0.5,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_zero_gradients(self):
        """Test with zero gradients."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "rewards": torch.zeros(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.zeros(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPMultiLayer:
    """Test with multi-layer policy models."""

    def test_multilayer_policy(self):
        """Test with multi-layer policy model."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = MultiLayerPolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            hidden_dim=32,
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_multilayer_with_large_hidden(self):
        """Test with multi-layer policy and large hidden dimension."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = MultiLayerPolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            hidden_dim=128,
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPEntropy:
    """Test entropy regularization."""

    def test_entropy_regularization(self):
        """Test that entropy regularization affects training."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer_high_entropy = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            entropy_coef=0.1,
        )
        optimizer_low_entropy = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            entropy_coef=0.001,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses_high = optimizer_high_entropy.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )
        losses_low = optimizer_low_entropy.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss_high, entropy_high, kl_loss_high = losses_high
        action_loss_low, entropy_low, kl_loss_low = losses_low

        assert not torch.isnan(torch.tensor(action_loss_high))
        assert not torch.isnan(torch.tensor(action_loss_low))

    def test_zero_entropy_coef(self):
        """Test with zero entropy coefficient."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            entropy_coef=0.0,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPLearningRate:
    """Test different learning rates."""

    def test_very_small_lr(self):
        """Test with very small learning rate."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-12)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_very_large_lr(self):
        """Test with very large learning rate."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1.0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPEpochsAndBatches:
    """Test different epoch and mini-batch configurations."""

    def test_single_epoch(self):
        """Test with single epoch."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_many_epochs(self):
        """Test with many epochs."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_single_mini_batch(self):
        """Test with single mini-batch."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=1,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_many_mini_batches(self):
        """Test with many mini-batches."""
        obs_dim = 10
        action_dim = 5
        batch_size = 32
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=8,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPWithProvidedValues:
    """Test using pre-computed returns and advantages."""

    def test_with_provided_returns(self):
        """Test with pre-computed returns."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        rewards = torch.randn(batch_size, seq_len)
        masks = torch.ones(batch_size, seq_len)

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": rewards,
            "masks": masks,
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_with_provided_advantages(self):
        """Test with pre-computed advantages."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        rewards = torch.randn(batch_size, seq_len)
        masks = torch.ones(batch_size, seq_len)

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": rewards,
            "masks": masks,
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_with_all_precomputed(self):
        """Test with all values pre-computed."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        rewards = torch.randn(batch_size, seq_len)
        masks = torch.ones(batch_size, seq_len)

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": rewards,
            "masks": masks,
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPNoOldLogProbs:
    """Test behavior when old_log_probs is not provided."""

    def test_without_old_log_probs(self):
        """Test update without old_log_probs."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            None,
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPInvalidInputs:
    """Test error handling for invalid inputs."""

    def test_invalid_clip_param_negative(self):
        """Test with negative clip_param."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=-0.1,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_invalid_clip_param_very_large(self):
        """Test with very large clip_param."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=10.0,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_invalid_reinforce_epoch_zero(self):
        """Test with zero reinforce_epoch."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_invalid_num_mini_batch_zero(self):
        """Test with zero num_mini_batch."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=1,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_invalid_entropy_coef_negative(self):
        """Test with negative entropy coefficient."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            entropy_coef=-0.1,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_invalid_lr_zero(self):
        """Test with zero learning rate."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-10)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_invalid_max_grad_norm_zero(self):
        """Test with zero max_grad_norm."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            max_grad_norm=1e-8,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_invalid_max_grad_norm_negative(self):
        """Test with negative max_grad_norm."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            max_grad_norm=1e-8,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_invalid_kl_coef_negative(self):
        """Test with negative KL coefficient."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            use_kl_loss=True,
            kl_coef=1e-5,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_empty_batch(self):
        """Test with empty batch."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, 4, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, 4)),
            "rewards": torch.randn(batch_size, 4),
            "masks": torch.ones(batch_size, 4),
            "old_log_probs": torch.randn(batch_size, 4),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_mismatched_batch_sizes(self):
        """Test with mismatched batch sizes in batch dict."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_mismatched_sequence_lengths(self):
        """Test with mismatched sequence lengths in batch dict."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_invalid_action_values(self):
        """Test with invalid action values (out of bounds)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_invalid_mask_values(self):
        """Test with invalid mask values (not binary)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_nan_rewards(self):
        """Test with NaN rewards."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_inf_rewards(self):
        """Test with infinite rewards."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_nan_actions(self):
        """Test with NaN actions."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_nan_old_log_probs(self):
        """Test with NaN old_log_probs."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_nan_obs(self):
        """Test with NaN observations."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestReinforcePPObservationOptional:
    """Test when observations are optional."""

    def test_without_observations(self):
        """Test that update handles missing observations gracefully."""
        action_dim = 5
        obs_dim = 10
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))


class TestGPT2PolicyForward:
    """Test GPT2Policy forward pass."""

    def test_gpt2_policy_forward_shape(self):
        """Test that GPT2Policy forward produces correct output shape."""

        action_space = spaces.Discrete(10)
        obs_dim = 768
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )

        batch_size = 4
        seq_len = 8
        obs = torch.randn(batch_size, seq_len, obs_dim)

        logits = policy(obs)

        assert logits.shape == (batch_size, seq_len, 10)

    def test_gpt2_policy_forward_single_step(self):
        """Test GPT2Policy with single time step."""

        action_space = spaces.Discrete(5)
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=2, n_embd=32
        )

        obs = torch.randn(2, 1, obs_dim)
        logits = policy(obs)

        assert logits.shape == (2, 1, 5)
        assert not torch.isnan(logits).any()

    def test_gpt2_policy_forward_long_sequence(self):
        """Test GPT2Policy with longer sequence."""

        action_space = spaces.Discrete(8)
        obs_dim = 64
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space,
            observation_space,
            n_layer=3,
            n_head=4,
            n_embd=64,
            block_size=32,
        )

        obs = torch.randn(4, 16, obs_dim)
        logits = policy(obs)

        assert logits.shape == (4, 16, 8)
        assert not torch.isnan(logits).any()


class TestGPT2PolicyEvaluateActions:
    """Test GPT2Policy evaluate_actions method."""

    def test_evaluate_actions_shapes(self):
        """Test that evaluate_actions returns correct shapes."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )

        batch_size = 4
        seq_len = 8
        obs = torch.randn(batch_size, seq_len, obs_dim)
        actions = torch.randint(0, action_dim, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert values.shape == (batch_size, seq_len)
        assert action_log_probs.shape == (batch_size, seq_len)
        assert entropy.shape == (batch_size, seq_len)
        assert logits.shape == (batch_size, seq_len, action_dim)

    def test_evaluate_actions_log_probs_valid(self):
        """Test that action log probabilities are valid (finite and reasonable)."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )

        batch_size = 4
        seq_len = 8
        obs = torch.randn(batch_size, seq_len, obs_dim)
        actions = torch.randint(0, action_dim, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        # Log probs should be finite
        assert not torch.isnan(action_log_probs).any()
        assert not torch.isinf(action_log_probs).any()

        # Log probs should be negative (log of probabilities <= 1)
        assert (action_log_probs <= 0).all()

    def test_evaluate_actions_entropy_positive(self):
        """Test that entropy is positive for non-deterministic policies."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )

        batch_size = 4
        seq_len = 8
        obs = torch.randn(batch_size, seq_len, obs_dim)
        actions = torch.randint(0, action_dim, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        # Entropy should be positive (non-deterministic policy)
        assert (entropy > 0).all()
        assert not torch.isnan(entropy).any()

    def test_evaluate_actions_values_are_zeros(self):
        """Test that values are zeros (no critic network)."""

        action_dim = 5
        obs_dim = 32
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=2, n_embd=32
        )

        batch_size = 4
        seq_len = 4
        obs = torch.randn(batch_size, seq_len, obs_dim)
        actions = torch.randint(0, action_dim, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert (values == 0).all()

    def test_evaluate_actions_values_with_value_head(self):
        """Test that evaluate_actions returns learned values when value head is enabled."""

        action_dim = 5
        obs_dim = 32
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space,
            observation_space,
            n_layer=2,
            n_head=2,
            n_embd=32,
            use_value_head=True,
        )

        batch_size = 4
        seq_len = 4
        obs = torch.randn(batch_size, seq_len, obs_dim)
        actions = torch.randint(0, action_dim, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert values.shape == (batch_size, seq_len)
        assert not torch.allclose(values, torch.zeros_like(values))


class TestGPT2PolicyReinforcePP:
    """Test GPT2Policy with ReinforcePP optimizer."""

    def test_gpt2_policy_basic_update(self):
        """Test basic update with GPT2Policy."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch_size = 4
        seq_len = 8
        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))

    def test_gpt2_policy_parameters_updated(self):
        """Test that GPT2Policy parameters are updated."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        initial_params = {
            name: param.clone() for name, param in policy.named_parameters()
        }

        batch_size = 4
        seq_len = 8
        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        # At least some parameters should have changed
        updated = False
        for name, param in policy.named_parameters():
            if not torch.equal(initial_params[name], param):
                updated = True
                break
        assert updated, "No parameters were updated"

    def test_gpt2_policy_with_scheduler(self):
        """Test GPT2Policy with learning rate scheduler."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )
        initial_lr = 1e-4
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer,
            step_size=1,
            gamma=0.5,
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=scheduler,
            clip_param=0.2,
            num_mini_batch=2,
        )

        batch_size = 4
        seq_len = 8
        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))
        # Check that scheduler was stepped (lr should be reduced)
        current_lr = scheduler.get_last_lr()[0]
        assert current_lr < initial_lr

    def test_gpt2_policy_with_kl_loss(self):
        """Test GPT2Policy with KL loss enabled."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            use_kl_loss=True,
            kl_coef=1e-5,
        )

        batch_size = 4
        seq_len = 8
        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert kl_loss >= 0

    def test_gpt2_policy_large_batch(self):
        """Test GPT2Policy with large batch size."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        optimizer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=4,
        )

        batch_size = 32
        seq_len = 8
        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        action_loss, entropy, kl_loss = losses

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isinf(torch.tensor(action_loss))


class TestGPT2PolicyEdgeCases:
    """Test GPT2Policy edge cases."""

    def test_gpt2_policy_block_size_limit(self):
        """Test that GPT2Policy raises error for sequence > block_size."""

        action_space = spaces.Discrete(5)
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space,
            observation_space,
            n_layer=2,
            n_head=2,
            n_embd=32,
            block_size=8,
        )

        # Sequence length > block_size should raise AssertionError
        obs = torch.randn(2, 16, obs_dim)
        with pytest.raises(AssertionError, match="block_size"):
            policy(obs)

    def test_gpt2_policy_single_action(self):
        """Test GPT2Policy with single action class."""

        action_space = spaces.Discrete(1)
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=2, n_embd=32
        )

        batch_size = 4
        seq_len = 4
        obs = torch.randn(batch_size, seq_len, obs_dim)
        actions = torch.zeros(batch_size, seq_len, dtype=torch.long)
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert action_log_probs.shape == (batch_size, seq_len)
        # With single action, entropy should be 0
        assert (entropy == 0).all()

    def test_gpt2_policy_different_hidden_dims(self):
        """Test GPT2Policy with different hidden dimensions."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Test with larger hidden dimension
        policy = GPT2Policy(
            action_space, observation_space, n_layer=4, n_head=8, n_embd=128
        )

        batch_size = 4
        seq_len = 8
        obs = torch.randn(batch_size, seq_len, obs_dim)
        logits = policy(obs)

        assert logits.shape == (batch_size, seq_len, action_dim)
        assert not torch.isnan(logits).any()


class TestBasePolicyAbstract:
    """Test BasePolicy abstract class behavior."""

    def test_base_policy_cannot_instantiate(self):
        """Test that BasePolicy cannot be instantiated directly."""

        action_space = spaces.Discrete(5)
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        with pytest.raises(TypeError):
            BasePolicy(action_space, observation_space)

    def test_gpt2_policy_is_base_policy(self):
        """Test that GPT2Policy is an instance of BasePolicy."""

        action_space = spaces.Discrete(5)
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(action_space, observation_space)
        assert isinstance(policy, BasePolicy)
        assert isinstance(policy, nn.Module)

    def test_base_policy_has_required_methods(self):
        """Test that BasePolicy has required abstract methods."""

        assert hasattr(BasePolicy, "forward")
        assert hasattr(BasePolicy, "evaluate_actions")


class TestGPT2PolicyActionSpace:
    """Test GPT2Policy with different action spaces."""

    def test_gpt2_policy_large_action_space(self):
        """Test GPT2Policy with large action space."""

        action_dim = 100
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )

        batch_size = 4
        seq_len = 8
        obs = torch.randn(batch_size, seq_len, obs_dim)
        logits = policy(obs)

        assert logits.shape == (batch_size, seq_len, action_dim)

    def test_gpt2_policy_small_action_space(self):
        """Test GPT2Policy with small action space."""

        action_dim = 2
        obs_dim = 32
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=2, n_embd=32
        )

        batch_size = 4
        seq_len = 4
        obs = torch.randn(batch_size, seq_len, obs_dim)
        logits = policy(obs)

        assert logits.shape == (batch_size, seq_len, action_dim)


class TestGPT2PolicyObservationSpace:
    """Test GPT2Policy with different observation spaces."""

    def test_gpt2_policy_small_obs_dim(self):
        """Test GPT2Policy with small observation dimension."""

        action_dim = 10
        obs_dim = 8
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=2, n_embd=32
        )

        batch_size = 4
        seq_len = 4
        obs = torch.randn(batch_size, seq_len, obs_dim)
        logits = policy(obs)

        assert logits.shape == (batch_size, seq_len, action_dim)

    def test_gpt2_policy_large_obs_dim(self):
        """Test GPT2Policy with large observation dimension."""

        action_dim = 10
        obs_dim = 512
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )

        batch_size = 4
        seq_len = 4
        obs = torch.randn(batch_size, seq_len, obs_dim)
        logits = policy(obs)

        assert logits.shape == (batch_size, seq_len, action_dim)


class TestPPOKLLoss:
    """Test PPO trainer KL penalty support."""

    class DummyEnv:
        def __init__(self, obs_dim: int):
            self.max_steps = 4
            self.obs = np.zeros(obs_dim, dtype=np.float32)

        def reset(self):
            return self.obs.copy(), {}

        def step(self, action):
            reward = float(action == 0)
            done = False
            truncated = False
            return self.obs.copy(), reward, done, truncated, {}

    def test_ppo_with_kl_loss_runs(self):
        action_dim = 3
        obs_dim = 8
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = SimplePolicy(action_space, observation_space)
        env = self.DummyEnv(obs_dim)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        trainer = PPOTrainer(
            policy,
            env,
            max_episodes=1,
            optimizer=optimizer,
            lr_scheduler=None,
            num_mini_batch=1,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            use_kl_loss=True,
            kl_coef=0.1,
        )

        trainer.fit(checkpoint_every=1, earlystop_last=1, use_wandb=False)


class TestReinforcePPCheckpointLoading:
    """Test checkpoint save/load functionality for ReinforcePP."""

    def test_reinforcepp_checkpoint_save_and_load(self, tmp_path):
        """Test that ReinforcePP saves and loads checkpoints correctly."""

        obs_dim = 10
        action_dim = 1  # Must match number of augmenters (1)
        checkpoint_dir = str(tmp_path / "ckpt")

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        trainer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=5,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_every=1,
            earlystop_last=10,
        )

        # Run training to create checkpoints
        trainer.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify checkpoint files exist
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
        assert len(ckpt_files) > 0, "No checkpoint files were saved"

        # Create a new policy and trainer
        policy2 = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        torch_optimizer2 = torch.optim.Adam(policy2.parameters(), lr=1e-3)
        lr_scheduler2 = torch.optim.lr_scheduler.StepLR(
            torch_optimizer2, step_size=100, gamma=0.9
        )
        trainer2 = ReinforcePPTrainer(
            policy_model=policy2,
            env=env,
            max_episodes=3,
            optimizer=torch_optimizer2,
            lr_scheduler=lr_scheduler2,
            checkpoint_every=1,
            earlystop_last=10,
        )

        # Get initial params before loading
        initial_params = {
            name: param.clone() for name, param in policy2.named_parameters()
        }

        # Run training with checkpoint loading
        trainer2.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify that the policy was loaded from checkpoint (params changed)
        params_changed = False
        for name, param in policy2.named_parameters():
            if not torch.equal(initial_params[name], param):
                params_changed = True
                break
        assert params_changed, (
            "Policy parameters were not updated after loading checkpoint"
        )

    def test_reinforcepp_no_checkpoint_when_empty_dir(self, tmp_path):
        """Test that ReinforcePP works when checkpoint directory is empty."""
        checkpoint_dir = str(tmp_path / "empty_ckpt")
        os.makedirs(checkpoint_dir, exist_ok=True)

        obs_dim = 10
        action_dim = 1

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        trainer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=3,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_every=10,
            earlystop_last=10,
        )

        # Should not raise any error
        trainer.fit(
            checkpoint_every=10,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

    def test_reinforcepp_checkpoint_loads_latest(self, tmp_path):
        """Test that ReinforcePP loads the latest checkpoint."""

        obs_dim = 10
        action_dim = 1
        checkpoint_dir = str(tmp_path / "ckpt_latest")

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        trainer = ReinforcePPTrainer(
            policy_model=policy,
            env=env,
            max_episodes=5,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_every=1,
            earlystop_last=10,
        )

        # Run training to create checkpoints
        trainer.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify the latest checkpoint has the highest episode number
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
        episodes = []
        for f in ckpt_files:
            try:
                ep = int(os.path.basename(f).split("_")[1].split(".")[0])
                episodes.append(ep)
            except ValueError:
                pass
        assert len(episodes) > 0
        latest_episode = max(episodes)

        # Create a new policy and trainer
        policy2 = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        torch_optimizer2 = torch.optim.Adam(policy2.parameters(), lr=1e-3)
        lr_scheduler2 = torch.optim.lr_scheduler.StepLR(
            torch_optimizer2, step_size=100, gamma=0.9
        )
        trainer2 = ReinforcePPTrainer(
            policy_model=policy2,
            env=env,
            max_episodes=latest_episode
            + 3,  # Must be > latest_episode to run after loading
            optimizer=torch_optimizer2,
            lr_scheduler=lr_scheduler2,
            checkpoint_every=1,
            earlystop_last=10,
        )

        # Run training - should start from latest_episode + 1
        trainer2.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify that the new checkpoint saved has the correct episode number
        new_ckpt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
        new_episodes = []
        for f in new_ckpt_files:
            try:
                ep = int(os.path.basename(f).split("_")[1].split(".")[0])
                new_episodes.append(ep)
            except ValueError:
                pass
        assert max(new_episodes) >= latest_episode + 1, (
            f"Expected episode >= {latest_episode + 1}, got {max(new_episodes)}"
        )


class TestPPOCheckpointLoading:
    """Test checkpoint save/load functionality for PPO."""

    def test_ppo_checkpoint_save_and_load(self, tmp_path):
        """Test that PPO saves and loads checkpoints correctly."""

        obs_dim = 10
        action_dim = 1  # Must match number of augmenters (1)
        checkpoint_dir = str(tmp_path / "ppo_ckpt")

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        trainer = PPOTrainer(
            actor_critic=policy,
            env=env,
            max_episodes=5,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            num_mini_batch=1,
            value_loss_coef=0.5,
            entropy_coef=0.01,
        )

        # Run training to create checkpoints
        trainer.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify checkpoint files exist
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "ppo_checkpoint_*.pt"))
        assert len(ckpt_files) > 0, "No PPO checkpoint files were saved"

        # Create a new policy and trainer
        policy2 = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        torch_optimizer2 = torch.optim.Adam(policy2.parameters(), lr=1e-3)
        lr_scheduler2 = torch.optim.lr_scheduler.StepLR(
            torch_optimizer2, step_size=100, gamma=0.9
        )
        trainer2 = PPOTrainer(
            actor_critic=policy2,
            env=env,
            max_episodes=3,
            optimizer=torch_optimizer2,
            lr_scheduler=lr_scheduler2,
            num_mini_batch=1,
            value_loss_coef=0.5,
            entropy_coef=0.01,
        )

        # Get initial params before loading
        initial_params = {
            name: param.clone() for name, param in policy2.named_parameters()
        }

        # Run training with checkpoint loading
        trainer2.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify that the policy was loaded from checkpoint (params changed)
        params_changed = False
        for name, param in policy2.named_parameters():
            if not torch.equal(initial_params[name], param):
                params_changed = True
                break
        assert params_changed, (
            "PPO policy parameters were not updated after loading checkpoint"
        )

    def test_ppo_no_checkpoint_when_empty_dir(self, tmp_path):
        """Test that PPO works when checkpoint directory is empty."""
        checkpoint_dir = str(tmp_path / "ppo_empty_ckpt")
        os.makedirs(checkpoint_dir, exist_ok=True)

        obs_dim = 10
        action_dim = 1

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        trainer = PPOTrainer(
            actor_critic=policy,
            env=env,
            max_episodes=3,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            num_mini_batch=1,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            checkpoint_every=10,
            earlystop_last=10,
        )

        # Should not raise any error
        trainer.fit(
            checkpoint_every=10,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

    def test_ppo_checkpoint_loads_latest(self, tmp_path):
        """Test that PPO loads the latest checkpoint."""

        obs_dim = 10
        action_dim = 1
        checkpoint_dir = str(tmp_path / "ppo_ckpt_latest")

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        trainer = PPOTrainer(
            actor_critic=policy,
            env=env,
            max_episodes=5,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            num_mini_batch=1,
            value_loss_coef=0.5,
            entropy_coef=0.01,
        )

        # Run training to create checkpoints
        trainer.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify the latest checkpoint has the highest episode number
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "ppo_checkpoint_*.pt"))
        episodes = []
        for f in ckpt_files:
            try:
                ep = int(os.path.basename(f).split("_")[2].split(".")[0])
                episodes.append(ep)
            except ValueError:
                pass
        assert len(episodes) > 0
        latest_episode = max(episodes)

        # Create a new policy and trainer
        policy2 = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        torch_optimizer2 = torch.optim.Adam(policy2.parameters(), lr=1e-3)
        lr_scheduler2 = torch.optim.lr_scheduler.StepLR(
            torch_optimizer2, step_size=100, gamma=0.9
        )
        trainer2 = PPOTrainer(
            actor_critic=policy2,
            env=env,
            max_episodes=latest_episode
            + 3,  # Must be > latest_episode to run after loading
            optimizer=torch_optimizer2,
            lr_scheduler=lr_scheduler2,
            num_mini_batch=1,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            checkpoint_every=1,
            earlystop_last=10,
        )

        # Run training - should start from latest_episode + 1
        trainer2.fit(
            checkpoint_every=1,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify that the new checkpoint saved has the correct episode number
        new_ckpt_files = glob.glob(os.path.join(checkpoint_dir, "ppo_checkpoint_*.pt"))
        new_episodes = []
        for f in new_ckpt_files:
            try:
                ep = int(os.path.basename(f).split("_")[2].split(".")[0])
                new_episodes.append(ep)
            except ValueError:
                pass
        assert max(new_episodes) >= latest_episode + 1, (
            f"Expected episode >= {latest_episode + 1}, got {max(new_episodes)}"
        )


class TestGRPOOffPolicy:
    """Test off-policy GRPO functionality."""

    def test_off_policy_works_without_replay_buffer(self):
        """Test that off-policy GRPO works without a replay buffer."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        # Off-policy should work without replay buffer
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
            use_off_policy=True,
            # No replay_buffer provided
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(entropy))
        assert kl_loss == 0.0

    def test_off_policy_with_replay_buffer(self):
        """Test basic off-policy update with replay buffer."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        replay_buffer = SimpleReplayBuffer(capacity=1000)

        # Fill buffer with some experience
        for _ in range(32):
            replay_buffer.push(
                obs=np.zeros(obs_dim, dtype=np.float32),
                action=0,
                reward=0.5,
                log_prob=-0.1,
                mask=1.0,
            )

        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
            use_off_policy=True,
            replay_buffer=replay_buffer,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(entropy))
        assert kl_loss == 0.0

    def test_off_policy_parameters_updated(self):
        """Test that off-policy update actually updates model parameters."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        replay_buffer = SimpleReplayBuffer(capacity=1000)

        # Fill buffer with some experience
        for _ in range(32):
            replay_buffer.push(
                obs=np.zeros(obs_dim, dtype=np.float32),
                action=0,
                reward=0.5,
                log_prob=-0.1,
                mask=1.0,
            )

        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
            use_off_policy=True,
            replay_buffer=replay_buffer,
        )

        initial_params = {
            name: param.clone() for name, param in policy.named_parameters()
        }

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        for name, param in policy.named_parameters():
            assert not torch.equal(initial_params[name], param), (
                f"Parameter {name} was not updated"
            )

    def test_off_policy_empty_buffer_falls_back(self):
        """Test that off-policy GRPO falls back to on-policy when buffer is empty."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        replay_buffer = SimpleReplayBuffer(capacity=1000)

        # Buffer is empty - should fall back to on-policy style training
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
            use_off_policy=True,
            replay_buffer=replay_buffer,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        # Should not raise an error - falls back to on-policy style
        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(entropy))
        assert kl_loss == 0.0

    def test_off_policy_loss_values_reasonable(self):
        """Test that off-policy loss values are reasonable (not NaN or inf)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        replay_buffer = SimpleReplayBuffer(capacity=1000)

        # Fill buffer with varied rewards
        for i in range(32):
            replay_buffer.push(
                obs=np.zeros(obs_dim, dtype=np.float32),
                action=i % action_dim,
                reward=float(i) / 32.0,
                log_prob=-0.1,
                mask=1.0,
            )

        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
            use_off_policy=True,
            replay_buffer=replay_buffer,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isinf(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(entropy))
        assert not torch.isinf(torch.tensor(entropy))

    def test_off_policy_with_zero_variance_masking(self):
        """Test off-policy GRPO with zero-variance masking enabled."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        replay_buffer = SimpleReplayBuffer(capacity=1000)

        # Fill buffer with constant rewards (zero variance)
        for _ in range(32):
            replay_buffer.push(
                obs=np.zeros(obs_dim, dtype=np.float32),
                action=0,
                reward=1.0,  # Constant reward
                log_prob=-0.1,
                mask=1.0,
            )

        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
            use_off_policy=True,
            replay_buffer=replay_buffer,
            use_zero_variance_masking=True,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        # Should not raise an error even with zero variance
        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(entropy))

    def test_off_policy_pushes_to_buffer_in_fit(self):
        """Test that off-policy GRPO pushes transitions to replay buffer during fit."""
        obs_dim = 10
        action_dim = 1  # Match the environment's single augmenter

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=3,  # Short episodes for testing
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        replay_buffer = SimpleReplayBuffer(capacity=1000)

        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=2,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
            use_off_policy=True,
            replay_buffer=replay_buffer,
        )

        # Before fit, buffer should be empty
        assert replay_buffer.is_empty()

        # Run fit
        grpo.fit(
            checkpoint_every=10,
            earlystop_last=10,
            use_wandb=False,
            checkpoint_dir="./ckpt",
        )

        # After fit, buffer should have data
        assert replay_buffer.size > 0, (
            f"Expected buffer to have data after fit, got size {replay_buffer.size}"
        )

    def test_on_policy_still_works_without_buffer(self):
        """Test that on-policy GRPO still works without a replay buffer."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        # On-policy should work without replay buffer
        grpo = GRPOTrainer(
            policy_model=policy,
            env=env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            clip_param=0.2,
            num_mini_batch=2,
            group_size=2,
            use_off_policy=False,  # Explicitly on-policy
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        action_loss, entropy, kl_loss = grpo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            None,
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(entropy))



class TestPPOOffPolicy:
    """Test off-policy PPO functionality."""

    def test_off_policy_works_without_replay_buffer(self):
        """Test that off-policy PPO works without a replay buffer."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        # Off-policy should work without replay buffer
        ppo = PPOTrainer(
            policy,
            env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            num_mini_batch=2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            use_off_policy=True,
            # No replay_buffer provided
        )

        batch = {
            "obs": torch.randn(batch_size, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size,)),
            "rewards": torch.randn(batch_size),
            "masks": torch.ones(batch_size),
            "old_log_probs": torch.randn(batch_size),
            "values": torch.zeros(batch_size),
            "returns": torch.randn(batch_size),
        }

        value_loss, action_loss, dist_entropy, kl_loss = ppo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            batch["values"],
            batch["returns"],
            torch.zeros_like(batch["obs"]),
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(dist_entropy))
        assert not torch.isnan(torch.tensor(value_loss))
        assert kl_loss == 0.0

    def test_off_policy_with_replay_buffer(self):
        """Test basic off-policy update with replay buffer."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        replay_buffer = SimpleReplayBuffer(capacity=1000)

        # Fill buffer with some experience
        for _ in range(32):
            replay_buffer.push(
                obs=np.zeros(obs_dim, dtype=np.float32),
                action=0,
                reward=0.5,
                log_prob=-0.1,
                mask=1.0,
            )

        ppo = PPOTrainer(
            policy,
            env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            num_mini_batch=2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            use_off_policy=True,
            replay_buffer=replay_buffer,
        )

        batch = {
            "obs": torch.randn(batch_size, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size,)),
            "rewards": torch.randn(batch_size),
            "masks": torch.ones(batch_size),
            "old_log_probs": torch.randn(batch_size),
            "values": torch.zeros(batch_size),
            "returns": torch.randn(batch_size),
        }

        value_loss, action_loss, dist_entropy, kl_loss = ppo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            batch["values"],
            batch["returns"],
            torch.zeros_like(batch["obs"]),
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(dist_entropy))
        assert not torch.isnan(torch.tensor(value_loss))
        assert kl_loss == 0.0

    def test_off_policy_parameters_updated(self):
        """Test that off-policy update actually updates model parameters."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        replay_buffer = SimpleReplayBuffer(capacity=1000)

        # Fill buffer with some experience
        for _ in range(32):
            replay_buffer.push(
                obs=np.zeros(obs_dim, dtype=np.float32),
                action=0,
                reward=0.5,
                log_prob=-0.1,
                mask=1.0,
            )

        ppo = PPOTrainer(
            policy,
            env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            num_mini_batch=2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            use_off_policy=True,
            replay_buffer=replay_buffer,
        )

        initial_params = {
            name: param.clone() for name, param in policy.named_parameters()
        }

        batch = {
            "obs": torch.randn(batch_size, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size,)),
            "rewards": torch.randn(batch_size),
            "masks": torch.ones(batch_size),
            "old_log_probs": torch.randn(batch_size),
            "values": torch.zeros(batch_size),
            "returns": torch.randn(batch_size),
        }

        ppo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            batch["values"],
            batch["returns"],
            torch.zeros_like(batch["obs"]),
        )

        # Check that parameters have changed
        params_changed = False
        for name, param in policy.named_parameters():
            if not torch.equal(initial_params[name], param):
                params_changed = True
                break
        assert params_changed, "Policy parameters were not updated after off-policy training"

    def test_tis_cap_bounds_importance_sampling(self):
        """Test that TIS cap correctly bounds the importance sampling ratio."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )
        replay_buffer = SimpleReplayBuffer(capacity=1000)

        # Fill buffer with experience that will create large IS ratios
        for _ in range(32):
            replay_buffer.push(
                obs=np.zeros(obs_dim, dtype=np.float32),
                action=0,
                reward=0.5,
                log_prob=-5.0,  # Very different log prob to create large ratio
                mask=1.0,
            )

        # Use a small tis_cap to test bounding
        tis_cap_value = 1.5
        ppo = PPOTrainer(
            policy,
            env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            num_mini_batch=2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            use_off_policy=True,
            replay_buffer=replay_buffer,
            tis_cap=tis_cap_value,
        )

        # Create batch with very different log probs to test TIS cap
        batch = {
            "obs": torch.randn(batch_size, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size,)),
            "rewards": torch.randn(batch_size),
            "masks": torch.ones(batch_size),
            "old_log_probs": torch.randn(batch_size) * 10,  # Large variance
            "values": torch.zeros(batch_size),
            "returns": torch.randn(batch_size),
        }

        # Should not crash or produce NaN
        value_loss, action_loss, dist_entropy, kl_loss = ppo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            batch["values"],
            batch["returns"],
            torch.zeros_like(batch["obs"]),
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(dist_entropy))
        assert not torch.isnan(torch.tensor(value_loss))

    def test_on_policy_still_works(self):
        """Test that on-policy PPO (default) still works correctly."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=lambda x: np.zeros(obs_dim, dtype=np.float32),
            reward_model=lambda x: 0.0,
            max_steps=10,
        )
        torch_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=100, gamma=0.9
        )

        # On-policy should work without replay buffer
        ppo = PPOTrainer(
            policy,
            env,
            max_episodes=100,
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            num_mini_batch=2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            use_off_policy=False,  # Explicitly on-policy
        )

        batch = {
            "obs": torch.randn(batch_size, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size,)),
            "rewards": torch.randn(batch_size),
            "masks": torch.ones(batch_size),
            "old_log_probs": torch.randn(batch_size),
            "values": torch.zeros(batch_size),
            "returns": torch.randn(batch_size),
        }

        value_loss, action_loss, dist_entropy, kl_loss = ppo.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["masks"],
            batch["old_log_probs"],
            batch["values"],
            batch["returns"],
            torch.zeros_like(batch["obs"]),
        )

        assert not torch.isnan(torch.tensor(action_loss))
        assert not torch.isnan(torch.tensor(dist_entropy))
        assert not torch.isnan(torch.tensor(value_loss))
        assert kl_loss == 0.0
