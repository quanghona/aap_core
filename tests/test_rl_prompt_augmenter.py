#!/usr/bin/env python3
"""Comprehensive tests for RLPromptAugmenter class.

Tests cover __init__, train, and augment methods with normal inputs,
edge cases, extreme values, and exception scenarios.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

# Ensure the core package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aap_core.policy import BasePolicy, GPT2Policy
from aap_core.prompt_augmenter import (
    IdentityPromptAugmenter,
    PromptOptimizationEnv,
    RLPromptAugmenter,
    SimplePromptAugmenter,
)
from aap_core.types import AgentMessage


class SimplePolicy(BasePolicy):
    """Simple single-layer policy model for testing."""

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
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
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
        logits = self.forward(obs)
        batch_size, seq_len, _ = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        values = torch.zeros(batch_size, seq_len, device=obs.device)
        return values, action_log_probs, entropy, logits


@pytest.fixture
def embedding_model():
    """Return a simple embedding model that produces fixed-dimension vectors."""

    def model(prompt: str) -> np.ndarray:
        return np.random.randn(64).astype(np.float32)

    return model


@pytest.fixture
def reward_model():
    """Return a reward model that gives higher scores for longer prompts."""

    def model(prompt: str) -> float:
        return float(len(prompt)) / 100.0

    return model


@pytest.fixture
def augmenters():
    """Return a list of simple augmenters for the environment."""
    return [
        IdentityPromptAugmenter(),
        SimplePromptAugmenter(format="{query} {data}"),
    ]


@pytest.fixture
def env(embedding_model, reward_model, augmenters):
    """Return a PromptOptimizationEnv with minimal configuration."""
    return PromptOptimizationEnv(
        initial_prompt="test prompt",
        augmenters=augmenters,
        embedding_model=embedding_model,
        reward_model=reward_model,
        max_steps=3,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for model checkpoints."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def policy_model(env):
    """Return a simple policy model for inference tests."""
    return SimplePolicy(env.action_space, env.observation_space)


@pytest.fixture
def rl_augmenter(env, policy_model):
    """Return an RLPromptAugmenter configured for inference."""
    return RLPromptAugmenter(env=env, policy_model=policy_model)


class TestRLPromptAugmenterInit:
    """Tests for RLPromptAugmenter.__init__."""

    def test_init_with_policy_model(self, env, policy_model):
        """Normal initialization with an existing policy model."""
        aug = RLPromptAugmenter(env=env, policy_model=policy_model)
        assert aug.env is env
        assert aug.policy_model is policy_model

    def test_init_requires_policy_model(self, env):
        """RLPromptAugmenter requires a policy model at initialization."""
        with pytest.raises(ValueError):
            RLPromptAugmenter(env=env)


class TestRLPromptAugmenterTrain:
    """Tests to confirm RLPromptAugmenter is inference-only."""

    def test_train_method_does_not_exist(self, rl_augmenter):
        """RLPromptAugmenter should not expose a training API."""
        with pytest.raises(AttributeError):
            rl_augmenter.train()


class TestRLPromptAugmenterAugment:
    """Tests for RLPromptAugmenter.augment()."""

    def test_augment_normal_inference(self, rl_augmenter):
        """Normal inference: augment returns modified message."""
        msg = AgentMessage(query="original prompt")
        result = rl_augmenter.augment(msg)
        assert isinstance(result, AgentMessage)
        assert hasattr(result, "query")

    def test_augment_infer_mode(self, rl_augmenter):
        """Inference path returns a valid AgentMessage."""
        msg = AgentMessage(query="original prompt")
        result = rl_augmenter.augment(msg)
        assert isinstance(result, AgentMessage)
        assert result.query == msg.query or isinstance(result.query, str)

    def test_augment_terminated_early(self, embedding_model, reward_model, augmenters):
        """Augment handles episodes that terminate early."""

        def high_reward(prompt: str) -> float:
            return 1000.0

        env = PromptOptimizationEnv(
            initial_prompt="test prompt",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=high_reward,
            max_steps=10,
            reward_threshold=100.0,
        )
        policy_model = SimplePolicy(env.action_space, env.observation_space)
        aug = RLPromptAugmenter(env=env, policy_model=policy_model)
        msg = AgentMessage(query="test query for augmentation")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_augment_with_different_prompts(self, rl_augmenter):
        """Augment works with different initial prompts."""
        prompts = ["short", "a much longer prompt with more words to test", ""]
        for prompt in prompts:
            msg = AgentMessage(query=prompt)
            result = rl_augmenter.augment(msg)
            assert isinstance(result, AgentMessage)

    def test_augment_returns_same_message_object(self, rl_augmenter):
        """Augment modifies and returns the same message object."""
        msg = AgentMessage(query="original prompt")
        result = rl_augmenter.augment(msg)
        assert result is msg

    def test_augment_with_kwargs_passed_through(self, rl_augmenter):
        """Augment accepts kwargs without failure."""
        msg = AgentMessage(query="test prompt")
        result = rl_augmenter.augment(msg, extra_arg="ignored")
        assert isinstance(result, AgentMessage)

    def test_augment_with_single_step_env(
        self, embedding_model, reward_model, augmenters
    ):
        """Augment with environment that has max_steps=1."""
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=1,
        )
        policy_model = SimplePolicy(env.action_space, env.observation_space)
        aug = RLPromptAugmenter(env=env, policy_model=policy_model)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_augment_preserves_message_attributes(self, rl_augmenter):
        """Augment preserves other message attributes beyond query."""
        msg = AgentMessage(
            query="original",
            origin="test_agent",
            context={"key": "value"},
        )
        result = rl_augmenter.augment(msg)
        assert result.origin == "test_agent"
        assert result.context == {"key": "value"}


class TestRLPromptAugmenterIntegration:
    """Integration and edge case tests for RLPromptAugmenter."""

    def test_infer_with_single_augmenter(self, embedding_model, reward_model):
        """Environment with only one augmenter works with inference."""
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy_model = SimplePolicy(env.action_space, env.observation_space)
        aug = RLPromptAugmenter(env=env, policy_model=policy_model)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_infer_with_many_augmenters(self, embedding_model, reward_model):
        """Environment with many augmenters works with inference."""
        augmenters = [IdentityPromptAugmenter() for _ in range(10)]
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy_model = SimplePolicy(env.action_space, env.observation_space)
        aug = RLPromptAugmenter(env=env, policy_model=policy_model)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_augment_with_empty_query(self, rl_augmenter):
        """Augment with an empty query string."""
        msg = AgentMessage(query="")
        result = rl_augmenter.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_augment_with_unicode_query(self, rl_augmenter):
        """Augment with unicode characters in query."""
        msg = AgentMessage(
            query="Hello 世界 \U0001f30d \u041f\u0440\u0438\u0432\u0435\u0442"
        )
        result = rl_augmenter.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_env_with_zero_augmenters_raises(self, embedding_model, reward_model):
        """Environment with 0 augmenters should raise AssertionError."""
        with pytest.raises(
            AssertionError, match="At least one augmenter must be provided"
        ):
            PromptOptimizationEnv(
                initial_prompt="test",
                augmenters=[],
                embedding_model=embedding_model,
                reward_model=reward_model,
                max_steps=2,
            )


@pytest.fixture
def gpt2_policy_small(env):
    """Return a small GPT2Policy (1 layer) for testing."""
    return GPT2Policy(
        env.action_space,
        env.observation_space,
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=8,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )


@pytest.fixture
def gpt2_policy_default(env):
    """Return a default GPT2Policy (4 layers) for testing."""
    return GPT2Policy(
        env.action_space,
        env.observation_space,
        n_layer=4,
        n_head=4,
        n_embd=128,
    )


@pytest.fixture
def gpt2_rl_augmenter_small(env, gpt2_policy_small):
    """Return an RLPromptAugmenter with small GPT2Policy."""
    return RLPromptAugmenter(env=env, policy_model=gpt2_policy_small)


@pytest.fixture
def gpt2_rl_augmenter_default(env, gpt2_policy_default):
    """Return an RLPromptAugmenter with default GPT2Policy."""
    return RLPromptAugmenter(env=env, policy_model=gpt2_policy_default)


class TestGPT2PolicyIntegration:
    """Tests for RLPromptAugmenter with GPT2Policy."""

    def test_gpt2_policy_inference_small(self, gpt2_rl_augmenter_small):
        """GPT2Policy with 1 layer works for inference."""
        msg = AgentMessage(query="test prompt")
        result = gpt2_rl_augmenter_small.augment(msg)
        assert isinstance(result, AgentMessage)
        assert hasattr(result, "query")

    def test_gpt2_policy_inference_default(self, gpt2_rl_augmenter_default):
        """GPT2Policy with default config works for inference."""
        msg = AgentMessage(query="test prompt")
        result = gpt2_rl_augmenter_default.augment(msg)
        assert isinstance(result, AgentMessage)
        assert hasattr(result, "query")

    def test_gpt2_policy_forward(self, gpt2_policy_small, env):
        """GPT2Policy forward pass produces correct output shape."""
        obs = torch.randn(4, env.observation_space.shape[0])
        logits = gpt2_policy_small(obs)
        assert logits.shape == (4, 1, env.action_space.n)

    def test_gpt2_policy_forward_batched(self, gpt2_policy_small, env):
        """GPT2Policy forward pass with batched input."""
        obs = torch.randn(8, 2, env.observation_space.shape[0])
        logits = gpt2_policy_small(obs)
        assert logits.shape == (8, 2, env.action_space.n)

    def test_gpt2_policy_evaluate_actions(self, gpt2_policy_small, env):
        """GPT2Policy evaluate_actions returns correct shapes."""
        batch_size = 4
        seq_len = 2
        obs = torch.randn(batch_size, seq_len, env.observation_space.shape[0])
        actions = torch.randint(0, env.action_space.n, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = gpt2_policy_small.evaluate_actions(
            obs, actions, masks
        )

        assert values.shape == (batch_size, seq_len)
        assert action_log_probs.shape == (batch_size, seq_len)
        assert entropy.shape == (batch_size, seq_len)
        assert logits.shape == (batch_size, seq_len, env.action_space.n)

    def test_gpt2_policy_evaluate_actions_1d(self, gpt2_policy_small, env):
        """GPT2Policy evaluate_actions with 1D inputs."""
        batch_size = 4
        obs = torch.randn(batch_size, env.observation_space.shape[0])
        actions = torch.randint(0, env.action_space.n, (batch_size,))
        masks = torch.ones(batch_size)

        values, action_log_probs, entropy, logits = gpt2_policy_small.evaluate_actions(
            obs, actions, masks
        )

        assert values.shape == (batch_size, 1)
        assert action_log_probs.shape == (batch_size, 1)
        assert entropy.shape == (batch_size, 1)
        assert logits.shape == (batch_size, 1, env.action_space.n)

    def test_gpt2_policy_with_many_augmenters(self, embedding_model, reward_model):
        """GPT2Policy works with many augmenters (large action space)."""
        augmenters = [IdentityPromptAugmenter() for _ in range(20)]
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        aug = RLPromptAugmenter(env=env, policy_model=policy)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_gpt2_policy_deterministic_action(self, gpt2_policy_small, env):
        """GPT2Policy get_action with deterministic=True returns argmax."""
        obs = torch.randn(4, env.observation_space.shape[0])
        logits = gpt2_policy_small(obs)
        action, log_prob = gpt2_policy_small.get_action(
            logits.squeeze(1), deterministic=True
        )
        expected = torch.argmax(logits.squeeze(1), dim=-1)
        assert torch.equal(action, expected)

    def test_gpt2_policy_stochastic_action(self, gpt2_policy_small, env):
        """GPT2Policy get_action with deterministic=False returns sampled action."""
        torch.manual_seed(42)
        obs = torch.randn(4, env.observation_space.shape[0])
        logits = gpt2_policy_small(obs)
        action, log_prob = gpt2_policy_small.get_action(
            logits.squeeze(1), deterministic=False
        )
        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert all(0 <= a < env.action_space.n for a in action.tolist())

    def test_gpt2_policy_save_load(self, gpt2_policy_small, temp_dir):
        """GPT2Policy can be saved and loaded."""
        path = os.path.join(temp_dir, "policy.pt")
        gpt2_policy_small.save(path)

        new_policy = GPT2Policy(
            gpt2_policy_small.action_space,
            gpt2_policy_small.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
        )
        new_policy.load(path)

        # Compare state dicts directly to avoid device mismatches
        for name, param in gpt2_policy_small.named_parameters():
            assert torch.allclose(
                param,
                new_policy.state_dict()[name],
                atol=1e-6,
            ), f"Mismatch in {name}"

    def test_gpt2_policy_with_value_head(self, env):
        """GPT2Policy with value head produces non-zero values."""
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            use_value_head=True,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        batch_size = 4
        seq_len = 2
        obs = torch.randn(batch_size, seq_len, env.observation_space.shape[0])
        actions = torch.randint(0, env.action_space.n, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert values.shape == (batch_size, seq_len)
        assert not torch.allclose(values, torch.zeros_like(values))

    def test_gpt2_policy_different_embedding_dims(self, reward_model):
        """GPT2Policy works with different embedding dimensions."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(256).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=64,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        obs = torch.randn(2, 256)
        logits = policy(obs)
        assert logits.shape == (2, 1, 1)

    def test_gpt2_policy_inference_with_different_prompts(
        self, gpt2_rl_augmenter_small
    ):
        """GPT2Policy inference works with various prompt types."""
        prompts = ["short", "a longer prompt with more context", ""]
        for prompt in prompts:
            msg = AgentMessage(query=prompt)
            result = gpt2_rl_augmenter_small.augment(msg)
            assert isinstance(result, AgentMessage)

    def test_gpt2_policy_parameter_count(self, gpt2_policy_small):
        """GPT2Policy has reasonable parameter count."""
        param_count = sum(p.numel() for p in gpt2_policy_small.parameters())
        assert param_count > 0
        # Small 1-layer policy should have < 100k params
        assert param_count < 100_000


class TestEmbeddingDimensionIntegration:
    """Tests for RLPromptAugmenter with various embedding dimensions."""

    def test_simple_policy_small_embedding(self, reward_model):
        """SimplePolicy works with small embedding dimension (16)."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(16).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = SimplePolicy(env.action_space, env.observation_space)
        aug = RLPromptAugmenter(env=env, policy_model=policy)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_simple_policy_large_embedding(self, reward_model):
        """SimplePolicy works with large embedding dimension (768)."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(768).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = SimplePolicy(env.action_space, env.observation_space)
        aug = RLPromptAugmenter(env=env, policy_model=policy)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_gpt2_policy_small_embedding(self, reward_model):
        """GPT2Policy works with small embedding dimension (16)."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(16).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        aug = RLPromptAugmenter(env=env, policy_model=policy)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_gpt2_policy_medium_embedding(self, reward_model):
        """GPT2Policy works with medium embedding dimension (128)."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(128).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=64,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        aug = RLPromptAugmenter(env=env, policy_model=policy)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_gpt2_policy_large_embedding(self, reward_model):
        """GPT2Policy works with large embedding dimension (768)."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(768).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=128,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        aug = RLPromptAugmenter(env=env, policy_model=policy)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_gpt2_policy_forward_small_embedding(self, reward_model):
        """GPT2Policy forward pass with small embedding dimension."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(16).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        obs = torch.randn(4, 16)
        logits = policy(obs)
        assert logits.shape == (4, 1, 1)

    def test_gpt2_policy_forward_large_embedding(self, reward_model):
        """GPT2Policy forward pass with large embedding dimension."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(768).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=128,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        obs = torch.randn(4, 768)
        logits = policy(obs)
        assert logits.shape == (4, 1, 1)

    def test_gpt2_policy_evaluate_actions_small_embedding(self, reward_model):
        """GPT2Policy evaluate_actions with small embedding dimension."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(16).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        batch_size = 4
        seq_len = 2
        obs = torch.randn(batch_size, seq_len, 16)
        actions = torch.randint(0, env.action_space.n, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert values.shape == (batch_size, seq_len)
        assert action_log_probs.shape == (batch_size, seq_len)
        assert entropy.shape == (batch_size, seq_len)
        assert logits.shape == (batch_size, seq_len, env.action_space.n)

    def test_gpt2_policy_evaluate_actions_large_embedding(self, reward_model):
        """GPT2Policy evaluate_actions with large embedding dimension."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(768).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=128,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        batch_size = 4
        seq_len = 2
        obs = torch.randn(batch_size, seq_len, 768)
        actions = torch.randint(0, env.action_space.n, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert values.shape == (batch_size, seq_len)
        assert action_log_probs.shape == (batch_size, seq_len)
        assert entropy.shape == (batch_size, seq_len)
        assert logits.shape == (batch_size, seq_len, env.action_space.n)

    def test_gpt2_policy_save_load_small_embedding(self, reward_model, temp_dir):
        """GPT2Policy save/load works with small embedding dimension."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(16).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )

        path = os.path.join(temp_dir, "policy_small_emb.pt")
        policy.save(path)

        new_policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
        )
        new_policy.load(path)

        for name, param in policy.named_parameters():
            assert torch.allclose(
                param,
                new_policy.state_dict()[name],
                atol=1e-6,
            ), f"Mismatch in {name}"

    def test_gpt2_policy_save_load_large_embedding(self, reward_model, temp_dir):
        """GPT2Policy save/load works with large embedding dimension."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(768).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=128,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )

        path = os.path.join(temp_dir, "policy_large_emb.pt")
        policy.save(path)

        new_policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=128,
            block_size=8,
        )
        new_policy.load(path)

        for name, param in policy.named_parameters():
            assert torch.allclose(
                param,
                new_policy.state_dict()[name],
                atol=1e-6,
            ), f"Mismatch in {name}"

    def test_gpt2_policy_with_many_augmenters_small_embedding(self, reward_model):
        """GPT2Policy with many augmenters and small embedding dimension."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(16).astype(np.float32)

        augmenters = [IdentityPromptAugmenter() for _ in range(20)]
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        aug = RLPromptAugmenter(env=env, policy_model=policy)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_gpt2_policy_with_many_augmenters_large_embedding(self, reward_model):
        """GPT2Policy with many augmenters and large embedding dimension."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(768).astype(np.float32)

        augmenters = [IdentityPromptAugmenter() for _ in range(20)]
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=128,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        aug = RLPromptAugmenter(env=env, policy_model=policy)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_gpt2_policy_with_value_head_small_embedding(self, reward_model):
        """GPT2Policy with value head and small embedding dimension."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(16).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            use_value_head=True,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        batch_size = 4
        seq_len = 2
        obs = torch.randn(batch_size, seq_len, 16)
        actions = torch.randint(0, env.action_space.n, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert values.shape == (batch_size, seq_len)
        assert not torch.allclose(values, torch.zeros_like(values))

    def test_gpt2_policy_with_value_head_large_embedding(self, reward_model):
        """GPT2Policy with value head and large embedding dimension."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(768).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=128,
            block_size=8,
            use_value_head=True,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        batch_size = 4
        seq_len = 2
        obs = torch.randn(batch_size, seq_len, 768)
        actions = torch.randint(0, env.action_space.n, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert values.shape == (batch_size, seq_len)
        assert not torch.allclose(values, torch.zeros_like(values))

    def test_gpt2_policy_extreme_small_embedding(self, reward_model):
        """GPT2Policy with extreme small embedding dimension (1)."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(1).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        obs = torch.randn(4, 1)
        logits = policy(obs)
        assert logits.shape == (4, 1, 1)

    def test_simple_policy_extreme_small_embedding(self, reward_model):
        """SimplePolicy with extreme small embedding dimension (1)."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(1).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = SimplePolicy(env.action_space, env.observation_space)
        aug = RLPromptAugmenter(env=env, policy_model=policy)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_gpt2_policy_evaluate_actions_extreme_small_embedding(self, reward_model):
        """GPT2Policy evaluate_actions with extreme small embedding dimension (1)."""

        def embedding_model(prompt: str) -> np.ndarray:
            return np.random.randn(1).astype(np.float32)

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy = GPT2Policy(
            env.action_space,
            env.observation_space,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        batch_size = 4
        seq_len = 2
        obs = torch.randn(batch_size, seq_len, 1)
        actions = torch.randint(0, env.action_space.n, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert values.shape == (batch_size, seq_len)
        assert action_log_probs.shape == (batch_size, seq_len)
        assert entropy.shape == (batch_size, seq_len)
        assert logits.shape == (batch_size, seq_len, env.action_space.n)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
