"""Test cases for PromptOptimizationEnv."""

import numpy as np
import pytest
from aap_core.prompt_augmenter import (
    BasePromptAugmenter,
    IdentityPromptAugmenter,
    PromptOptimizationEnv,
    SimplePromptAugmenter,
)


class DummyAugmenter(BasePromptAugmenter):
    """A dummy augmenter that modifies the prompt in a specific way."""

    suffix: str = ""

    def augment(self, message, **kwargs):
        message.query = message.query + self.suffix
        return message


class TestPromptOptimizationEnv:
    """Test suite for PromptOptimizationEnv."""

    def test_initialization_with_default_values(self):
        """Test environment initialization with default values."""
        augmenters = [IdentityPromptAugmenter()]
        embedding_model = lambda x: np.zeros(768, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        assert len(env._augmenters) == 1
        assert env.action_space.n == 1
        assert env.observation_space.shape == (768,)

    def test_initialization_with_custom_values(self):
        """Test environment initialization with custom values."""
        augmenters = [
            IdentityPromptAugmenter(),
            SimplePromptAugmenter(format="{query} {data}", data_key="context.data"),
        ]
        embedding_model = lambda x: np.ones(128, dtype=np.float32)
        reward_model = lambda x: 1.0

        env = PromptOptimizationEnv(
            initial_prompt="Hello",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=20,
        )

        assert len(env._augmenters) == 2
        assert env.action_space.n == 2
        assert env.observation_space.shape == (128,)
        assert env.max_steps == 20

    def test_action_space_is_discrete(self):
        """Test that action space is Discrete."""
        augmenters = [IdentityPromptAugmenter() for _ in range(5)]
        embedding_model = lambda x: np.zeros(100, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        from gymnasium import spaces

        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 5

    def test_observation_space_is_box(self):
        """Test that observation space is Box."""
        augmenters = [IdentityPromptAugmenter()]
        embedding_model = lambda x: np.zeros(256, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            initial_prompt="test",  # Provide initial prompt to determine embedding dim
        )

        from gymnasium import spaces

        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (256,)
        assert env.observation_space.dtype == np.float32

    def test_reset_returns_observation_and_info(self):
        """Test that reset returns observation and info."""
        augmenters = [IdentityPromptAugmenter()]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            initial_prompt="Test",
        )

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10,)
        assert isinstance(info, dict)
        assert "current_prompt" in info
        assert "step_count" in info
        assert "num_augmenters" in info

    def test_reset_resets_state(self):
        """Test that reset properly resets the environment state."""
        augmenters = [DummyAugmenter(suffix=" test")]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="Test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=5,
        )

        # First reset
        obs, info = env.reset()
        assert info["step_count"] == 0
        assert info["current_prompt"] == "Test"

        # Advance the environment
        env.step(0)
        env.step(0)
        info = env._get_info()
        assert info["step_count"] == 2

        # Reset
        obs, info = env.reset()

        assert info["step_count"] == 0
        assert info["current_prompt"] == "Test"

    def test_step_with_valid_action(self):
        """Test step with a valid action."""
        augmenters = [DummyAugmenter(suffix=" world")]
        embedding_model = lambda x: np.ones(10, dtype=np.float32)
        reward_model = lambda x: 1.0 if "world" in x else 0.0

        env = PromptOptimizationEnv(
            initial_prompt="Hello",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            min_embedding_threshold=0.0,  # Disable embedding distance termination for this test
        )

        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10,)
        assert reward == 1.0
        assert terminated is False
        assert truncated is False
        assert info["current_prompt"] == "Hello world"
        assert info["step_count"] == 1

    def test_step_with_different_augmenters(self):
        """Test step with different augmenters."""
        augmenters = [
            DummyAugmenter(suffix="!"),
            DummyAugmenter(suffix="?"),
            DummyAugmenter(suffix="."),
        ]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: float(len(x))

        env = PromptOptimizationEnv(
            initial_prompt="Hi",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        env.reset()
        # Apply first augmenter
        obs1, reward1, _, _, _ = env.step(0)
        assert env._current_prompt == "Hi!"

        # Apply second augmenter
        obs2, reward2, _, _, _ = env.step(1)
        assert env._current_prompt == "Hi!?"

        # Apply third augmenter
        obs3, reward3, _, _, _ = env.step(2)
        assert env._current_prompt == "Hi!?."

    def test_step_with_invalid_action(self):
        """Test step with an invalid action."""
        augmenters = [IdentityPromptAugmenter()]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(5)

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(-1)

    def test_step_with_augmenter_failure(self):
        """Test step when augmenter fails."""

        # Create a simple augmenter that fails (must NOT inherit from IdentityPromptAugmenter
        # because step() terminates immediately for IdentityPromptAugmenter)
        class FailingAugmenter(BasePromptAugmenter):
            def augment(self, message, **kwargs):
                raise Exception("Augmenter failed")

        augmenters = [FailingAugmenter()]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="Test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        env.reset()
        # Should not raise exception, just warn and keep current prompt
        with pytest.warns(UserWarning, match="Augmenter 0 failed"):
            obs, reward, terminated, truncated, info = env.step(0)

        assert info["current_prompt"] == "Test"

    def test_step_with_max_steps_truncation(self):
        """Test step with max_steps truncation."""
        augmenters = [DummyAugmenter(suffix=" step")]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="Test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=3,
        )

        env.reset()
        # First two steps should not truncate
        env.step(0)
        _, _, _, truncated, _ = env.step(0)
        assert truncated is False

        # Third step should truncate
        _, _, _, truncated, info = env.step(0)
        assert truncated is True
        assert info["step_count"] == 3

    def test_reward_model_called_correctly(self):
        """Test that reward model is called with the correct prompt."""
        augmenters = [DummyAugmenter(suffix=" test")]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)

        reward_calls = []

        def reward_model(prompt):
            reward_calls.append(prompt)
            return 0.0

        env = PromptOptimizationEnv(
            initial_prompt="Hello",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        env.reset()
        env.step(0)
        assert "Hello test" in reward_calls

    def test_identity_augmenter_terminates_episode(self):
        """Test that selecting IdentityPromptAugmenter terminates the episode immediately."""
        augmenters = [IdentityPromptAugmenter()]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="Test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=10,  # Set high so truncation doesn't interfere
        )

        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        # Episode should terminate immediately
        assert terminated is True
        assert truncated is False
        assert reward == 0.0
        assert info["current_prompt"] == "Test"  # Prompt unchanged
        assert info["step_count"] == 1

    def test_identity_augmenter_in_mixed_augmenters_terminates(self):
        """Test that IdentityPromptAugmenter terminates even when mixed with other augmenters."""
        augmenters = [
            DummyAugmenter(suffix="!"),
            IdentityPromptAugmenter(),
            DummyAugmenter(suffix="?"),
        ]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="Test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=10,
            min_embedding_threshold=0.0,  # Disable embedding distance termination
        )

        env.reset()
        # First, apply a non-identity augmenter
        obs1, reward1, terminated1, truncated1, info1 = env.step(0)
        assert terminated1 is False
        assert info1["current_prompt"] == "Test!"

        # Now apply IdentityPromptAugmenter (action=1) - should terminate
        obs2, reward2, terminated2, truncated2, info2 = env.step(1)
        assert terminated2 is True
        assert truncated2 is False
        assert reward2 == 0.0
        assert info2["current_prompt"] == "Test!"  # Prompt unchanged
        assert info2["step_count"] == 2

    def test_identity_augmenter_does_not_call_reward_model(self):
        """Test that reward model is not called when IdentityPromptAugmenter terminates."""
        augmenters = [IdentityPromptAugmenter()]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)

        reward_called = []

        def reward_model(prompt):
            reward_called.append(True)
            return 0.0

        env = PromptOptimizationEnv(
            initial_prompt="Test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        env.reset()
        env.step(0)

        # Reward model should not be called for IdentityPromptAugmenter
        assert len(reward_called) == 0

    def test_identity_augmenter_observation_returns_embedding(self):
        """Test that step with IdentityPromptAugmenter returns the current embedding."""
        augmenters = [IdentityPromptAugmenter()]
        expected_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        embedding_model = lambda x: expected_embedding
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="Test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        env.reset()
        obs, _, terminated, _, _ = env.step(0)

        assert terminated is True
        np.testing.assert_array_equal(obs, expected_embedding)

    def test_embedding_model_called_correctly(self):
        """Test that embedding model is called with the correct prompt."""
        augmenters = [DummyAugmenter(suffix=" test")]

        embedding_calls = []

        def embedding_model(prompt):
            embedding_calls.append(prompt)
            return np.zeros(10, dtype=np.float32)

        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="Hello",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        env.reset()
        env.step(0)

        assert len(embedding_calls) >= 2  # Called during init and reset/step
        assert "Hello" in embedding_calls[0]
        assert "Hello test" in embedding_calls[-1]

    def test_get_observation(self):
        """Test _get_observation method."""
        augmenters = [IdentityPromptAugmenter()]

        def embedding_model(prompt):
            if prompt == "test":
                return np.ones(10, dtype=np.float32)
            return np.zeros(10, dtype=np.float32)

        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        obs, _ = env.reset()
        assert np.array_equal(obs, np.ones(10, dtype=np.float32))

    def test_get_info(self):
        """Test _get_info method."""
        augmenters = [IdentityPromptAugmenter()]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=10,
        )

        env.reset()
        env.step(0)
        env.step(0)

        info = env._get_info()

        assert info["current_prompt"] == "test"
        assert info["step_count"] == 2
        assert info["num_augmenters"] == 1

    def test_metadata(self):
        """Test environment metadata."""
        augmenters = [IdentityPromptAugmenter()]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        # Gymnasium environments always have metadata
        assert "render_modes" in env.metadata

    def test_no_render_mode_parameter(self):
        """Test that render_mode parameter is not used."""
        augmenters = [IdentityPromptAugmenter()]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 0.0

        # render_mode is no longer a parameter
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )
        # No render_mode attribute expected
        assert not hasattr(env, "render_mode") or env.render_mode is None

    def test_seed_reproducibility(self):
        """Test that seeding produces reproducible results."""
        augmenters = [IdentityPromptAugmenter()]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 0.0

        env1 = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )
        env2 = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        assert np.array_equal(obs1, obs2)

    def test_multiple_episodes(self):
        """Test multiple consecutive episodes."""
        augmenters = [DummyAugmenter(suffix="!")]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 1.0 if "!" in x else 0.0

        env = PromptOptimizationEnv(
            initial_prompt="Start",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )

        # First episode
        env.reset()
        env.step(0)
        _, _, _, truncated, _ = env.step(0)
        assert truncated is True

        # Reset and start second episode
        obs, info = env.reset()
        assert info["step_count"] == 0
        assert info["current_prompt"] == "Start"

        # Second episode
        env.step(0)
        _, _, _, truncated, _ = env.step(0)
        assert truncated is True

    def test_embedding_dtype(self):
        """Test that embedding is converted to float32."""
        augmenters = [IdentityPromptAugmenter()]

        def embedding_model(prompt):
            return np.zeros(10, dtype=np.float64)  # Return float64

        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
        )

        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_observation_copy(self):
        """Test that observation is a copy, not a reference."""
        augmenters = [IdentityPromptAugmenter()]
        embedding_model = lambda x: np.zeros(10, dtype=np.float32)
        reward_model = lambda x: 0.0

        env = PromptOptimizationEnv(
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            initial_prompt="test",
        )

        obs1, _ = env.reset()
        obs2 = env._get_observation()

        # Modify obs1
        obs1[0] = 999.0

        # obs2 should not be affected
        assert obs2[0] == 0.0
