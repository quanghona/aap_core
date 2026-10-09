import gymnasium
import numpy as np
import pytest
import torch
from aap_core.policy import (
    GPT2Policy,
    GPT2RoPEGQAPolicy,
    GQABlock,
    GroupedQuerySelfAttention,
    RotaryPositionalEmbeddings,
)


def test_gpt2_policy_save_load(tmp_path):
    # Setup environment parameters
    action_space = gymnasium.spaces.Discrete(10)
    observation_space = gymnasium.spaces.Box(
        low=-1.0, high=1.0, shape=(768,), dtype=np.float32
    )

    # Initialize policy
    policy = GPT2Policy(
        action_space=action_space,
        observation_space=observation_space,
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=16,
    )
    policy.eval()

    # Create dummy input
    batch_size = 4
    obs = torch.randn(batch_size, 768)

    # Get original output
    with torch.no_grad():
        original_logits = policy(obs)

    # Save policy
    save_path = tmp_path / "test_policy.pth"
    print(f"Saving policy to {save_path}")
    policy.save(str(save_path))

    # Create a new policy instance with same configuration
    new_policy = GPT2Policy(
        action_space=action_space,
        observation_space=observation_space,
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=16,
    )
    new_policy.eval()

    # Load policy
    print(f"Loading policy from {save_path}")
    new_policy.load(str(save_path))

    # Get new output
    with torch.no_grad():
        loaded_logits = new_policy(obs)

    # Compare outputs
    diff = torch.abs(original_logits - loaded_logits).max().item()
    print(f"Max difference between original and loaded logits: {diff}")

    assert diff < 1e-6, f"Difference too large: {diff}"


@pytest.fixture
def action_space():
    return gymnasium.spaces.Discrete(10)


@pytest.fixture
def observation_space():
    return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(768,), dtype=np.float32)


@pytest.fixture
def small_policy(action_space, observation_space):
    """Small GPT2RoPEGQAPolicy for fast tests."""
    return GPT2RoPEGQAPolicy(
        action_space=action_space,
        observation_space=observation_space,
        n_layer=2,
        n_head=4,
        n_kv_groups=2,
        n_embd=32,
        block_size=16,
    )


class TestRotaryPositionalEmbeddings:
    """Tests for Rotary Positional Embeddings."""

    def test_rope_output_shape(self):
        """RoPE should preserve input shape."""
        rope = RotaryPositionalEmbeddings(dim=64)
        x = torch.randn(2, 4, 16, 64)  # (batch, heads, seq_len, dim)
        out = rope(x)
        assert out.shape == x.shape

    def test_rope_partial_embedding(self):
        """RoPE should only transform specified fraction of features."""
        rope = RotaryPositionalEmbeddings(dim=64, rope_percentage=0.5)
        x = torch.randn(2, 4, 16, 64)
        out = rope(x)
        # First half should be transformed, second half should be unchanged
        assert out.shape == x.shape
        assert not torch.allclose(out[..., :32], x[..., :32])  # transformed
        assert torch.allclose(out[..., 32:], x[..., 32:])  # unchanged

    def test_rope_cache_reuse(self):
        """RoPE should reuse cached cos/sin for shorter sequences."""
        rope = RotaryPositionalEmbeddings(dim=64)
        x_long = torch.randn(2, 4, 32, 64)
        _ = rope(x_long)
        # Cache should exist
        assert rope.cos_cached is not None
        assert rope.sin_cached is not None
        # Shorter sequence should reuse cache
        x_short = torch.randn(2, 4, 16, 64)
        out_short = rope(x_short)
        assert out_short.shape == x_short.shape

    def test_rope_different_dtypes(self):
        """RoPE should handle different input dtypes."""
        rope = RotaryPositionalEmbeddings(dim=64)
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(2, 4, 16, 64, dtype=dtype)
            out = rope(x)
            assert out.dtype == dtype

    def test_rope_invalid_dim(self):
        """RoPE should raise error for odd dimensions."""
        with pytest.raises(AssertionError, match="must be even"):
            RotaryPositionalEmbeddings(dim=63)

    def test_rope_relative_position(self):
        """RoPE should encode relative position information."""
        rope = RotaryPositionalEmbeddings(dim=64)
        # Same vector at different positions should produce different outputs
        x = torch.ones(1, 1, 1, 64)
        out_0 = rope(x)
        x_pos1 = torch.ones(1, 1, 2, 64)
        out_1 = rope(x_pos1)
        # The second position should differ from the first
        assert not torch.allclose(out_0, out_1[..., 1:, :])


class TestGroupedQuerySelfAttention:
    """Tests for Grouped-Query Attention."""

    def test_gqa_output_shape(self):
        """GQA should preserve input shape."""
        attn = GroupedQuerySelfAttention(
            n_embd=128, n_head=8, n_kv_groups=4, attn_pdrop=0.0
        )
        x = torch.randn(2, 16, 128)
        out = attn(x)
        assert out.shape == x.shape

    def test_gqa_with_rope(self):
        """GQA with RoPE should work correctly."""
        attn = GroupedQuerySelfAttention(
            n_embd=64, n_head=4, n_kv_groups=2, attn_pdrop=0.0, use_rope=True
        )
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_gqa_without_rope(self):
        """GQA without RoPE should work correctly."""
        attn = GroupedQuerySelfAttention(
            n_embd=64, n_head=4, n_kv_groups=2, attn_pdrop=0.0, use_rope=False
        )
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_gqa_causal_masking(self):
        """GQA should apply causal mask (each position attends to previous only)."""
        attn = GroupedQuerySelfAttention(
            n_embd=64, n_head=4, n_kv_groups=2, attn_pdrop=0.0
        )
        x = torch.ones(1, 4, 64)
        out = attn(x)
        # Output should be finite
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gqa_dropout_training_vs_eval(self):
        """GQA dropout should differ between training and eval modes."""
        attn = GroupedQuerySelfAttention(
            n_embd=64, n_head=4, n_kv_groups=2, attn_pdrop=0.5
        )
        x = torch.randn(2, 8, 64)
        attn.train()
        out1 = attn(x)
        out2 = attn(x)
        # With 50% dropout, outputs should differ
        assert not torch.allclose(out1, out2)
        attn.eval()
        out3 = attn(x)
        out4 = attn(x)
        # In eval mode, outputs should be identical
        assert torch.allclose(out3, out4)

    def test_gqa_invalid_config(self):
        """GQA should raise error for invalid head/group configuration."""
        with pytest.raises(AssertionError):
            GroupedQuerySelfAttention(
                n_embd=64, n_head=5, n_kv_groups=2, attn_pdrop=0.0
            )

    def test_gqa_multi_batch(self):
        """GQA should handle different batch sizes."""
        attn = GroupedQuerySelfAttention(
            n_embd=64, n_head=4, n_kv_groups=2, attn_pdrop=0.0
        )
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 8, 64)
            out = attn(x)
            assert out.shape == x.shape

    def test_gqa_multi_sequence_length(self):
        """GQA should handle different sequence lengths."""
        attn = GroupedQuerySelfAttention(
            n_embd=64, n_head=4, n_kv_groups=2, attn_pdrop=0.0
        )
        for seq_len in [1, 4, 8, 16, 32]:
            x = torch.randn(2, seq_len, 64)
            out = attn(x)
            assert out.shape == x.shape


class TestGQABlock:
    """Tests for GQA transformer block with RMSNorm."""

    def test_block_output_shape(self):
        """GQABlock should preserve input shape."""
        block = GQABlock(
            n_embd=64,
            n_head=4,
            n_kv_groups=2,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_block_with_rope(self):
        """GQABlock with RoPE should work correctly."""
        block = GQABlock(
            n_embd=64,
            n_head=4,
            n_kv_groups=2,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            use_rope=True,
        )
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_block_rmsnorm_pre_norm(self):
        """GQABlock should use RMSNorm for pre-normalization."""
        block = GQABlock(
            n_embd=64,
            n_head=4,
            n_kv_groups=2,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        assert isinstance(block.attn_norm, torch.nn.RMSNorm)
        assert isinstance(block.ff_norm, torch.nn.RMSNorm)

    def test_block_residual_connections(self):
        """GQABlock should have residual connections."""
        block = GQABlock(
            n_embd=64,
            n_head=4,
            n_kv_groups=2,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        x = torch.randn(2, 8, 64)
        out = block(x)
        # Output should be different from input (not just identity)
        assert not torch.allclose(out, x)

    def test_block_dropout(self):
        """GQABlock dropout should differ between training and eval modes."""
        block = GQABlock(
            n_embd=64,
            n_head=4,
            n_kv_groups=2,
            resid_pdrop=0.5,
            attn_pdrop=0.5,
        )
        x = torch.randn(2, 8, 64)
        block.train()
        out1 = block(x)
        out2 = block(x)
        assert not torch.allclose(out1, out2)
        block.eval()
        out3 = block(x)
        out4 = block(x)
        assert torch.allclose(out3, out4)

    def test_block_invalid_config(self):
        """GQABlock should raise error for invalid configuration."""
        with pytest.raises(AssertionError):
            GQABlock(
                n_embd=64,
                n_head=5,  # 64 % 5 != 0
                n_kv_groups=2,
                resid_pdrop=0.0,
                attn_pdrop=0.0,
            )


class TestGPT2RoPEGQAPolicy:
    """Tests for GPT-2 policy with RMSNorm, RoPE, and GQA."""

    def test_forward_single_step(self, small_policy):
        """Forward pass with single-step observation (2D input)."""
        policy = small_policy
        policy.eval()
        obs = torch.randn(4, 768)
        with torch.no_grad():
            logits = policy(obs)
        assert logits.shape == (4, 1, 10)

    def test_forward_sequence(self, small_policy):
        """Forward pass with sequence observation (3D input)."""
        policy = small_policy
        policy.eval()
        obs = torch.randn(4, 8, 768)
        with torch.no_grad():
            logits = policy(obs)
        assert logits.shape == (4, 8, 10)

    def test_evaluate_actions(self, small_policy):
        """evaluate_actions should return correct shapes."""
        policy = small_policy
        policy.train()
        obs = torch.randn(4, 8, 768)
        actions = torch.randint(0, 10, (4, 8))
        masks = torch.ones(4, 8)
        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )
        assert values.shape == (4, 8)
        assert action_log_probs.shape == (4, 8)
        assert entropy.shape == (4, 8)
        assert logits.shape == (4, 8, 10)

    def test_evaluate_actions_1d(self, small_policy):
        """evaluate_actions should handle 1D inputs."""
        policy = small_policy
        policy.train()
        obs = torch.randn(4, 768)
        actions = torch.randint(0, 10, (4,))
        masks = torch.ones(4)
        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )
        assert values.shape == (4, 1)
        assert action_log_probs.shape == (4, 1)
        assert entropy.shape == (4, 1)
        assert logits.shape == (4, 1, 10)

    def test_save_load(self, small_policy, tmp_path):
        """Policy should save and load correctly."""
        policy = small_policy
        policy.eval()
        obs = torch.randn(4, 768)
        with torch.no_grad():
            orig_logits = policy(obs)

        save_path = tmp_path / "policy.pth"
        policy.save(str(save_path))

        new_policy = GPT2RoPEGQAPolicy(
            action_space=gymnasium.spaces.Discrete(10),
            observation_space=gymnasium.spaces.Box(
                low=-1.0, high=1.0, shape=(768,), dtype=np.float32
            ),
            n_layer=2,
            n_head=4,
            n_kv_groups=2,
            n_embd=32,
            block_size=16,
        )
        new_policy.eval()
        new_policy.load(str(save_path))

        with torch.no_grad():
            loaded_logits = new_policy(obs)

        diff = torch.abs(orig_logits - loaded_logits).max().item()
        assert diff < 1e-6, f"Max diff: {diff}"

    def test_block_size_enforcement(self, small_policy):
        """Policy should enforce block_size constraint."""
        policy = small_policy
        policy.eval()
        # This should work (within block_size)
        obs_valid = torch.randn(2, 16, 768)
        with torch.no_grad():
            _ = policy(obs_valid)
        # This should raise assertion error
        obs_invalid = torch.randn(2, 17, 768)
        with pytest.raises(AssertionError, match="block_size"):
            with torch.no_grad():
                _ = policy(obs_invalid)

    def test_value_head(self, action_space, observation_space):
        """Policy with value head should produce non-zero values."""
        policy = GPT2RoPEGQAPolicy(
            action_space=action_space,
            observation_space=observation_space,
            n_layer=2,
            n_head=4,
            n_kv_groups=2,
            n_embd=32,
            block_size=16,
            use_value_head=True,
        )
        policy.train()
        obs = torch.randn(4, 8, 768)
        actions = torch.randint(0, 10, (4, 8))
        masks = torch.ones(4, 8)
        values, _, _, _ = policy.evaluate_actions(obs, actions, masks)
        # Values should not all be zero
        assert not torch.allclose(values, torch.zeros_like(values))

    def test_no_value_head(self, small_policy):
        """Policy without value head should produce zero values."""
        policy = small_policy
        policy.train()
        obs = torch.randn(4, 8, 768)
        actions = torch.randint(0, 10, (4, 8))
        masks = torch.ones(4, 8)
        values, _, _, _ = policy.evaluate_actions(obs, actions, masks)
        assert torch.allclose(values, torch.zeros_like(values))

    def test_deterministic_inference(self, small_policy):
        """Inference should be deterministic in eval mode."""
        policy = small_policy
        policy.eval()
        obs = torch.randn(4, 768)
        with torch.no_grad():
            out1 = policy(obs)
            out2 = policy(obs)
        assert torch.allclose(out1, out2)

    def test_gqa_configurations(self, action_space, observation_space):
        """Test various GQA configurations."""
        configs = [
            {"n_head": 4, "n_kv_groups": 4},  # MHA
            {"n_head": 4, "n_kv_groups": 2},  # GQA
            {"n_head": 8, "n_kv_groups": 2},  # Strong GQA
            {"n_head": 4, "n_kv_groups": 1},  # MQA
        ]
        for config in configs:
            policy = GPT2RoPEGQAPolicy(
                action_space=action_space,
                observation_space=observation_space,
                n_layer=1,
                n_head=config["n_head"],
                n_kv_groups=config["n_kv_groups"],
                n_embd=32,
                block_size=8,
            )
            policy.eval()
            obs = torch.randn(2, 768)
            with torch.no_grad():
                logits = policy(obs)
            assert logits.shape == (2, 1, 10)

    def test_rope_percentage(self, action_space, observation_space):
        """Policy with partial RoPE should work correctly."""
        policy = GPT2RoPEGQAPolicy(
            action_space=action_space,
            observation_space=observation_space,
            n_layer=1,
            n_head=4,
            n_kv_groups=2,
            n_embd=32,
            block_size=8,
            rope_percentage=0.5,
        )
        policy.eval()
        obs = torch.randn(2, 768)
        with torch.no_grad():
            logits = policy(obs)
        assert logits.shape == (2, 1, 10)

    def test_rms_eps(self, action_space, observation_space):
        """Policy with custom RMSNorm epsilon should work."""
        policy = GPT2RoPEGQAPolicy(
            action_space=action_space,
            observation_space=observation_space,
            n_layer=1,
            n_head=4,
            n_kv_groups=2,
            n_embd=32,
            block_size=8,
            rms_eps=1e-8,
        )
        policy.eval()
        obs = torch.randn(2, 768)
        with torch.no_grad():
            logits = policy(obs)
        assert logits.shape == (2, 1, 10)


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_forward_backward(self, small_policy):
        """Test full forward and backward pass."""
        policy = small_policy
        policy.train()
        obs = torch.randn(4, 8, 768)
        actions = torch.randint(0, 10, (4, 8))
        masks = torch.ones(4, 8)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        # Compute a simple loss
        loss = -action_log_probs.mean()
        loss.backward()

        # Check gradients exist
        for name, param in policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_multiple_episodes(self, small_policy):
        """Test multiple forward passes (simulating multiple episodes)."""
        policy = small_policy
        policy.eval()
        for _ in range(5):
            obs = torch.randn(4, 768)
            with torch.no_grad():
                logits = policy(obs)
            assert logits.shape == (4, 1, 10)

    def test_policy_with_different_obs_dims(self, action_space):
        """Test policy with different observation dimensions."""
        for obs_dim in [64, 128, 256, 768]:
            obs_space = gymnasium.spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
            policy = GPT2RoPEGQAPolicy(
                action_space=action_space,
                observation_space=obs_space,
                n_layer=1,
                n_head=4,
                n_kv_groups=2,
                n_embd=32,
                block_size=8,
            )
            policy.eval()
            obs = torch.randn(2, obs_dim)
            with torch.no_grad():
                logits = policy(obs)
            assert logits.shape == (2, 1, 10)
