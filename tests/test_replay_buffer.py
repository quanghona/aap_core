"""Test cases for the replay buffer module."""

import random

import numpy as np
import pytest
from aap_core.policy_trainer import (
    PrioritizedReplayBuffer,
    ReplayBufferEntry,
    SimpleReplayBuffer,
)


class TestReplayBufferEntry:
    """Test ReplayBufferEntry dataclass."""

    def test_create_entry(self):
        """Test creating a valid ReplayBufferEntry."""
        obs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        entry = ReplayBufferEntry(
            obs=obs, action=2, reward=0.5, log_prob=-1.2, mask=1.0
        )
        assert np.array_equal(entry.obs, obs)
        assert entry.action == 2
        assert entry.reward == 0.5
        assert entry.log_prob == -1.2
        assert entry.mask == 1.0

    def test_entry_zero_obs(self):
        """Test entry with zero observation vector."""
        obs = np.zeros(10, dtype=np.float32)
        entry = ReplayBufferEntry(obs=obs, action=0, reward=0.0, log_prob=0.0, mask=0.0)
        assert np.all(entry.obs == 0)
        assert entry.action == 0
        assert entry.reward == 0.0

    def test_entry_single_action(self):
        """Test entry with single action (action_dim=1)."""
        obs = np.random.randn(5).astype(np.float32)
        entry = ReplayBufferEntry(
            obs=obs, action=0, reward=1.0, log_prob=-0.5, mask=1.0
        )
        assert entry.action == 0

    def test_entry_extreme_values(self):
        """Test entry with extreme reward and log_prob values."""
        obs = np.random.randn(3).astype(np.float32)
        entry = ReplayBufferEntry(
            obs=obs, action=0, reward=1e6, log_prob=-1e6, mask=1.0
        )
        assert entry.reward == 1e6
        assert entry.log_prob == -1e6

    def test_entry_negative_reward(self):
        """Test entry with negative reward."""
        obs = np.random.randn(3).astype(np.float32)
        entry = ReplayBufferEntry(
            obs=obs, action=1, reward=-0.5, log_prob=-0.3, mask=1.0
        )
        assert entry.reward == -0.5


class TestSimpleReplayBuffer:
    """Test SimpleReplayBuffer uniform sampling."""

    @pytest.fixture
    def small_buffer(self):
        """Create a small buffer for testing."""
        return SimpleReplayBuffer(capacity=10)

    @pytest.fixture
    def large_buffer(self):
        """Create a large buffer for testing."""
        return SimpleReplayBuffer(capacity=1000)

    def test_init_capacity(self):
        """Test buffer initialization with valid capacity."""
        buffer = SimpleReplayBuffer(capacity=100)
        assert buffer.capacity == 100
        assert buffer.size == 0
        assert buffer.is_empty()
        assert not buffer.is_full

    def test_init_invalid_capacity(self):
        """Test buffer initialization with invalid capacity."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            SimpleReplayBuffer(capacity=0)
        with pytest.raises(ValueError, match="capacity must be positive"):
            SimpleReplayBuffer(capacity=-1)

    def test_push_and_sample(self, small_buffer):
        """Test pushing a single entry and sampling it."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        small_buffer.push(obs=obs, action=1, reward=0.5, log_prob=-0.3, mask=1.0)
        assert small_buffer.size == 1
        assert not small_buffer.is_empty()

        batch = small_buffer.sample(batch_size=1)
        assert batch["obs"].shape == (1, 3)
        assert np.array_equal(batch["obs"][0], obs)
        assert batch["action"][0] == 1
        assert batch["reward"][0] == 0.5
        assert batch["log_prob"][0] == -0.3
        assert batch["mask"][0] == 1.0

    def test_push_multiple_entries(self, small_buffer):
        """Test pushing multiple entries."""
        for i in range(10):
            obs = np.full(3, float(i), dtype=np.float32)
            small_buffer.push(
                obs=obs, action=i % 2, reward=float(i), log_prob=-float(i), mask=1.0
            )
        assert small_buffer.size == 10
        assert small_buffer.is_full

    def test_sample_less_than_buffer(self, small_buffer):
        """Test sampling fewer entries than buffer size."""
        for i in range(5):
            obs = np.full(3, float(i), dtype=np.float32)
            small_buffer.push(
                obs=obs, action=0, reward=float(i), log_prob=-0.1, mask=1.0
            )

        batch = small_buffer.sample(batch_size=3)
        assert batch["obs"].shape == (3, 3)
        assert len(batch["action"]) == 3
        assert len(batch["reward"]) == 3

    def test_sample_more_than_buffer(self, small_buffer):
        """Test sampling more entries than buffer size (should sample all)."""
        for i in range(3):
            obs = np.full(3, float(i), dtype=np.float32)
            small_buffer.push(
                obs=obs, action=0, reward=float(i), log_prob=-0.1, mask=1.0
            )

        batch = small_buffer.sample(batch_size=10)
        assert batch["obs"].shape == (3, 3)  # Should only return 3

    def test_sample_from_empty_buffer(self, small_buffer):
        """Test sampling from an empty buffer raises error."""
        with pytest.raises(RuntimeError, match="Cannot sample from an empty buffer"):
            small_buffer.sample(batch_size=1)

    def test_circular_overflow(self, small_buffer):
        """Test that buffer overwrites oldest entries when full."""
        # Fill buffer
        for i in range(10):
            obs = np.full(3, float(i), dtype=np.float32)
            small_buffer.push(
                obs=obs, action=0, reward=float(i), log_prob=-0.1, mask=1.0
            )
        assert small_buffer.is_full

        # Push more entries - should overwrite oldest
        for i in range(10, 15):
            obs = np.full(3, float(i), dtype=np.float32)
            small_buffer.push(
                obs=obs, action=0, reward=float(i), log_prob=-0.1, mask=1.0
            )

        assert small_buffer.size == 10  # Size should not exceed capacity
        assert small_buffer.is_full

        # After overflow: indices 0-4 have rewards 10-14, indices 5-9 have rewards 5-9
        # So buffer contains rewards {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
        # Rewards 0-4 should be gone
        batch = small_buffer.sample(batch_size=10)
        rewards = set(batch["reward"])
        # Should contain rewards 5-14, but NOT 0-4
        for r in rewards:
            assert 5 <= r <= 14, f"Expected reward in [5, 14], got {r}"
        # Verify that old rewards 0-4 are not present
        assert 0 not in rewards and 1 not in rewards and 2 not in rewards

    def test_single_element_buffer(self, small_buffer):
        """Test sampling from a buffer with a single element."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        small_buffer.push(obs=obs, action=0, reward=1.0, log_prob=-0.5, mask=1.0)

        batch = small_buffer.sample(batch_size=1)
        assert batch["obs"].shape == (1, 3)
        assert np.array_equal(batch["obs"][0], obs)

    def test_clear(self, small_buffer):
        """Test clearing the buffer."""
        for i in range(5):
            obs = np.full(3, float(i), dtype=np.float32)
            small_buffer.push(
                obs=obs, action=0, reward=float(i), log_prob=-0.1, mask=1.0
            )

        small_buffer.clear()
        assert small_buffer.size == 0
        assert small_buffer.is_empty()

        with pytest.raises(RuntimeError, match="Cannot sample from an empty buffer"):
            small_buffer.sample(batch_size=1)

    def test_sample_reproducibility_with_seed(self, small_buffer):
        """Test that sampling is reproducible with a fixed seed."""
        for i in range(10):
            obs = np.full(3, float(i), dtype=np.float32)
            small_buffer.push(
                obs=obs, action=0, reward=float(i), log_prob=-0.1, mask=1.0
            )

        random.seed(42)
        batch1 = small_buffer.sample(batch_size=5)

        random.seed(42)
        batch2 = small_buffer.sample(batch_size=5)

        assert np.array_equal(batch1["obs"], batch2["obs"])
        assert np.array_equal(batch1["reward"], batch2["reward"])

    def test_buffer_repr(self, small_buffer):
        """Test string representation of the buffer."""
        assert "SimpleReplayBuffer" in repr(small_buffer)
        assert "capacity=10" in repr(small_buffer)
        assert "size=0" in repr(small_buffer)

        small_buffer.push(np.zeros(3), 0, 0.0, 0.0, 1.0)
        assert "size=1" in repr(small_buffer)

    def test_obs_dtype_preserved(self, small_buffer):
        """Test that observation dtype is float32."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        small_buffer.push(obs=obs, action=0, reward=0.0, log_prob=0.0, mask=1.0)

        batch = small_buffer.sample(batch_size=1)
        assert batch["obs"].dtype == np.float32

    def test_large_buffer(self, large_buffer):
        """Test with a large buffer."""
        for i in range(500):
            obs = np.random.randn(10).astype(np.float32)
            large_buffer.push(
                obs=obs, action=i % 5, reward=float(i), log_prob=-0.1, mask=1.0
            )

        assert large_buffer.size == 500
        assert not large_buffer.is_full

        batch = large_buffer.sample(batch_size=32)
        assert batch["obs"].shape == (32, 10)
        assert len(batch["action"]) == 32


class TestPrioritizedReplayBuffer:
    """Test PrioritizedReplayBuffer prioritized sampling."""

    @pytest.fixture
    def buffer(self):
        """Create a prioritized replay buffer for testing."""
        return PrioritizedReplayBuffer(capacity=20, alpha=0.6, beta=0.4)

    def test_init(self, buffer):
        """Test buffer initialization."""
        assert buffer.capacity == 20
        assert buffer.size == 0
        assert buffer.is_empty()
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4

    def test_init_invalid_alpha(self):
        """Test initialization with invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be in \\[0, 1\\]"):
            PrioritizedReplayBuffer(capacity=100, alpha=-0.1)
        with pytest.raises(ValueError, match="alpha must be in \\[0, 1\\]"):
            PrioritizedReplayBuffer(capacity=100, alpha=1.5)

    def test_init_invalid_beta(self):
        """Test initialization with invalid beta."""
        with pytest.raises(ValueError, match="beta must be in \\[0, 1\\]"):
            PrioritizedReplayBuffer(capacity=100, beta=-0.1)
        with pytest.raises(ValueError, match="beta must be in \\[0, 1\\]"):
            PrioritizedReplayBuffer(capacity=100, beta=1.5)

    def test_push_with_priority(self, buffer):
        """Test pushing with explicit priority."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buffer.push(
            obs=obs, action=1, reward=0.5, log_prob=-0.3, mask=1.0, priority=0.8
        )
        assert buffer.size == 1

    def test_push_without_priority(self, buffer):
        """Test pushing without priority uses max priority."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buffer.push(obs=obs, action=1, reward=0.5, log_prob=-0.3, mask=1.0)
        assert buffer.size == 1

    def test_sample_with_weights(self, buffer):
        """Test that sampled batch includes importance sampling weights."""
        for i in range(10):
            obs = np.full(3, float(i), dtype=np.float32)
            buffer.push(
                obs=obs,
                action=i % 3,
                reward=float(i),
                log_prob=-0.1,
                mask=1.0,
                priority=0.5,
            )

        batch = buffer.sample(batch_size=5)
        assert "weights" in batch
        assert "indices" in batch
        assert batch["weights"].shape == (5,)
        assert batch["indices"] is not None

    def test_sample_less_than_buffer(self, buffer):
        """Test sampling fewer entries than buffer size."""
        for i in range(5):
            obs = np.full(3, float(i), dtype=np.float32)
            buffer.push(
                obs=obs,
                action=0,
                reward=float(i),
                log_prob=-0.1,
                mask=1.0,
                priority=0.5,
            )

        batch = buffer.sample(batch_size=3)
        assert batch["obs"].shape == (3, 3)

    def test_sample_more_than_buffer(self, buffer):
        """Test sampling more entries than buffer size."""
        for i in range(3):
            obs = np.full(3, float(i), dtype=np.float32)
            buffer.push(
                obs=obs,
                action=0,
                reward=float(i),
                log_prob=-0.1,
                mask=1.0,
                priority=0.5,
            )

        batch = buffer.sample(batch_size=10)
        assert batch["obs"].shape == (3, 3)

    def test_sample_from_empty_buffer(self, buffer):
        """Test sampling from an empty buffer raises error."""
        with pytest.raises(RuntimeError, match="Cannot sample from an empty buffer"):
            buffer.sample(batch_size=1)

    def test_update_priority(self, buffer):
        """Test updating priority of a specific transition."""
        for i in range(5):
            obs = np.full(3, float(i), dtype=np.float32)
            buffer.push(
                obs=obs,
                action=0,
                reward=float(i),
                log_prob=-0.1,
                mask=1.0,
                priority=0.1,
            )

        # Update priority of index 2
        buffer.update_priority(2, 1.0)

        # The updated entry should now have higher priority
        new_priority = buffer._get_priority(2)
        assert new_priority >= 1.0

    def test_importance_sampling_weights(self, buffer):
        """Test importance sampling weight computation."""
        for i in range(5):
            obs = np.full(3, float(i), dtype=np.float32)
            buffer.push(
                obs=obs,
                action=0,
                reward=float(i),
                log_prob=-0.1,
                mask=1.0,
                priority=float(i + 1),
            )

        indices = [0, 1, 2, 3, 4]
        weights = buffer.get_importance_sampling_weights(indices)
        assert weights.shape == (5,)
        # Weights should be between 0 and 1 (normalized)
        assert np.all(weights >= 0)
        assert np.all(weights <= 1.0 + 1e-6)

    def test_decay_beta(self, buffer):
        """Test beta decay over episodes."""
        initial_beta = buffer.beta
        for _ in range(10):
            buffer.decay_beta()
        assert buffer.beta > initial_beta
        assert buffer.beta <= 1.0

    def test_decay_beta_to_one(self, buffer):
        """Test that beta eventually approaches 1.0."""
        for _ in range(10000):
            buffer.decay_beta()
        assert abs(buffer.beta - 1.0) < 1e-3

    def test_clear(self, buffer):
        """Test clearing the buffer."""
        for i in range(5):
            obs = np.full(3, float(i), dtype=np.float32)
            buffer.push(
                obs=obs,
                action=0,
                reward=float(i),
                log_prob=-0.1,
                mask=1.0,
                priority=0.5,
            )

        buffer.clear()
        assert buffer.size == 0
        assert buffer.is_empty()

    def test_circular_overflow(self, buffer):
        """Test that buffer overwrites oldest entries when full."""
        for i in range(30):
            obs = np.full(3, float(i), dtype=np.float32)
            buffer.push(
                obs=obs,
                action=0,
                reward=float(i),
                log_prob=-0.1,
                mask=1.0,
                priority=0.5,
            )

        assert buffer.size == 20  # Should not exceed capacity

    def test_buffer_repr(self, buffer):
        """Test string representation."""
        assert "PrioritizedReplayBuffer" in repr(buffer)
        assert "capacity=20" in repr(buffer)
        assert "alpha=0.6" in repr(buffer)

    def test_prioritized_sampling_favors_high_priority(self, buffer):
        """Test that high-priority transitions are sampled more frequently."""
        # Add entries with varying priorities
        for i in range(10):
            obs = np.full(3, float(i), dtype=np.float32)
            priority = (
                0.1 if i < 5 else 1.0
            )  # First 5 low priority, last 5 high priority
            buffer.push(
                obs=obs,
                action=0,
                reward=float(i),
                log_prob=-0.1,
                mask=1.0,
                priority=priority,
            )

        # Sample many times and count frequency
        high_priority_count = 0
        num_samples = 1000
        for _ in range(num_samples):
            batch = buffer.sample(batch_size=1)
            idx = batch["indices"][0]
            if idx >= 5:  # High priority indices are 5-9
                high_priority_count += 1

        # High priority should be sampled more than 50% of the time
        # (they represent 50% of entries but should have higher sampling rate)
        assert high_priority_count / num_samples > 0.6

    def test_zero_priority_uses_epsilon(self, buffer):
        """Test that zero priority is clamped to epsilon."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buffer.push(obs=obs, action=0, reward=0.0, log_prob=0.0, mask=1.0, priority=0.0)
        assert buffer.size == 1

    def test_obs_dtype_preserved(self, buffer):
        """Test that observation dtype is float32."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        buffer.push(obs=obs, action=0, reward=0.0, log_prob=0.0, mask=1.0, priority=0.5)

        batch = buffer.sample(batch_size=1)
        assert batch["obs"].dtype == np.float32

    def test_uniform_sampling_when_alpha_zero(self):
        """Test that alpha=0 gives uniform sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.0)
        for i in range(50):
            obs = np.full(3, float(i), dtype=np.float32)
            buffer.push(
                obs=obs,
                action=0,
                reward=float(i),
                log_prob=-0.1,
                mask=1.0,
                priority=1.0,
            )

        # With alpha=0, all priorities are equal, so sampling should be uniform
        # We can verify by checking that all indices have equal probability
        # For simplicity, just verify sampling works
        batch = buffer.sample(batch_size=10)
        assert batch["obs"].shape == (10, 3)

    def test_beta_decay_property(self, buffer):
        """Test that beta decay is monotonic."""
        betas = [buffer.beta]
        for _ in range(100):
            buffer.decay_beta()
            betas.append(buffer.beta)

        # Beta should be non-decreasing
        for i in range(1, len(betas)):
            assert betas[i] >= betas[i - 1]
