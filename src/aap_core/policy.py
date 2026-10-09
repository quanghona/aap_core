"""Base policy classes for reinforcement learning.

This module provides abstract and concrete policy model classes that define
the interface for policy-based reinforcement learning algorithms. The policies
are designed to work with the PromptOptimizationEnv and similar gymnasium-based
environments.

Classes:
    BasePolicy: Abstract base class for all policy models
    GPT2Policy: GPT-2 based policy model using minGPT architecture
    RMSNorm: Root Mean Square Layer Normalization (pre-norm)
    RotaryPositionalEmbeddings: RoPE positional encoding
    GroupedQuerySelfAttention: GQA-based causal self-attention
    GPT2RoPEGQAPolicy: Upgraded policy with RMSNorm, RoPE, and GQA
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces


class BasePolicy(nn.Module, ABC):
    """Abstract base class for policy models.

    This class defines the interface for all policy models used in the
    reinforcement learning framework. Subclasses must implement the
    forward and evaluate_actions methods.

    The policy model takes observations as input and outputs action logits,
    which are then used to compute action probabilities and log probabilities
    for the taken actions.

    Args:
        action_space: The action space (typically gymnasium.spaces.Discrete)
        observation_space: The observation space (typically gymnasium.spaces.Box). Typically, the state is the n-dimensional embedding of the prompt

    Example:
        >>> from gymnasium import spaces
        >>> action_space = spaces.Discrete(5)
        >>> observation_space = spaces.Box(low=-1, high=1, shape=(768,))
        >>> policy = MyCustomPolicy(action_space, observation_space)
    """

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_space: spaces.Box,
        weight: Optional[str] = None,
        **kargs,
    ):
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy.

        Args:
            obs: Observations of shape (batch_size, obs_dim)

        Returns:
            Action logits of shape (batch_size, action_dim)
        """
        pass

    def get_action(
        self, logits: torch.Tensor, deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # Tuple of (action, action_log_prob)
        """Extract action from policy output logits.

        This method takes the raw action logits and samples an action according
        to the specified mode (deterministic or stochastic). It also computes
        the log probability of the selected action.

        Args:
            logits: Action logits of shape (batch_size, action_dim)
            deterministic: Whether to select the most likely action (True) or sample from the distribution (False)
        Returns:
            Tuple of (action, action_log_prob)
                - action: Selected action of shape (batch_size,)
                - action_log_prob: Log probability of the selected action of shape (batch_size,)
        """
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action, action_log_prob

    @abstractmethod
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions taken under the policy.

        This method computes the log probabilities of the taken actions,
        the entropy of the action distribution, and optionally value estimates.

        Args:
            obs: Observations of shape (batch_size, obs_dim)
            actions: Actions taken of shape (batch_size,)
            masks: Binary masks of shape (batch_size,) indicating
                   which positions are valid

        Returns:
            Tuple of (values, action_log_probs, entropy, logits):
                - values: Value predictions of shape (batch_size,)
                          (can be zeros if no critic network)
                - action_log_probs: Log probabilities of taken actions
                                    of shape (batch_size,)
                - entropy: Per-position entropy of shape (batch_size,)
                - logits: Action logits of shape (batch_size, action_dim)
        """
        pass

    def save(self, path: str):
        """Save the policy model to a file.

        Args:
            path: Path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load the policy model from a file.

        Args:
            path: Path to load the model from.
        """
        self.load_state_dict(torch.load(path))


class GPT2Policy(BasePolicy):
    """GPT-2 based policy model for reinforcement learning.

    This policy uses a GPT-2 architecture (based on minGPT) to process
    continuous observations and output action probabilities. The observations
    are projected to the embedding dimension and processed through transformer
    blocks with causal self-attention.

    The architecture follows the minGPT implementation from
    https://github.com/karpathy/minGPT/blob/master/mingpt/model.py, with
    modifications to accept continuous observations instead of discrete tokens.

    Args:
        action_space: The action space (gymnasium.spaces.Discrete)
        observation_space: The observation space (gymnasium.spaces.Box)
        n_layer: Number of transformer layers. Default: 4
        n_head: Number of attention heads. Default: 4
        n_embd: Embedding dimension. Default: 128
        block_size: Maximum sequence length. Default: 64
        embd_pdrop: Embedding dropout rate. Default: 0.1
        resid_pdrop: Residual dropout rate. Default: 0.1
        attn_pdrop: Attention dropout rate. Default: 0.1

    Example:
        >>> from gymnasium import spaces
        >>> action_space = spaces.Discrete(10)
        >>> observation_space = spaces.Box(low=-1, high=1, shape=(768,))
        >>> policy = GPT2Policy(action_space, observation_space, n_layer=6, n_embd=256)
        >>> obs = torch.randn(8, 768)  # batch=8, dim=768
        >>> logits = policy(obs)  # (8, 10)
    """

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_space: spaces.Box,
        weight: Optional[str] = None,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        block_size: int = 64,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        use_value_head: bool = False,
    ):
        super().__init__(action_space, observation_space)

        self.n_embd = n_embd
        self.block_size = block_size
        self.action_dim = int(action_space.n)
        self.use_value_head = use_value_head

        # Project observations to embedding dimension
        obs_dim = observation_space.shape[0]
        self.obs_projection = nn.Linear(obs_dim, n_embd)

        # Positional embeddings (max block_size positions)
        self.pos_embeddings = nn.Embedding(block_size, n_embd)

        # Dropout
        self.drop = nn.Dropout(embd_pdrop)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(n_embd, n_head, embd_pdrop, resid_pdrop, attn_pdrop)
                for _ in range(n_layer)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embd)

        # Action head - maps embeddings to action logits
        self.action_head = nn.Linear(n_embd, self.action_dim)

        # Optional value head - maps embeddings to scalar state values
        self.value_head = nn.Linear(n_embd, 1) if use_value_head else None

        # Initialize weights following GPT-2 conventions
        if weight is None:
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith("c_proj.weight"):
                    torch.nn.init.normal_(
                        p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer)
                    )
        else:
            self.load_state_dict(torch.load(weight))

    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 paper conventions.

        - Linear layers: normal initialization with std=0.02
        - Embedding layers: normal initialization with std=0.02
        - LayerNorm layers: bias=0, weight=1
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _get_hidden_states(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute shared transformer hidden states for observations.

        Args:
            obs: Observations of shape (batch_size, seq_len, obs_dim) or
                 (batch_size, obs_dim) for single-step inference.

        Returns:
            Hidden state tensor of shape (batch_size, seq_len, n_embd)
        """
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)

        batch_size, seq_len, obs_dim = obs.shape
        assert seq_len <= self.block_size, (
            f"Cannot forward sequence of length {seq_len}, "
            f"block_size is only {self.block_size}"
        )

        x = self.obs_projection(obs)

        device = obs.device
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_embeddings(pos)
        x = x + pos_emb

        x = self.drop(x)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return x

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GPT-2 policy.

        Args:
            obs: Observations of shape (batch_size, seq_len, obs_dim) or
                 (batch_size, obs_dim).

        Returns:
            Action logits of shape (batch_size, seq_len, action_dim)
        """
        hidden_states = self._get_hidden_states(obs)
        logits = self.action_head(hidden_states)
        return logits

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions taken under the policy.

        Args:
            obs: Observations of shape (batch_size, seq_len, obs_dim) or
                 (batch_size, obs_dim)
            actions: Actions taken of shape (batch_size, seq_len) or
                     (batch_size,)
            masks: Masks of shape (batch_size, seq_len) or (batch_size,)

        Returns:
            Tuple of (values, action_log_probs, entropy, logits):
                - values: Value predictions of shape (batch_size, seq_len)
                - action_log_probs: Log probs of taken actions of shape (batch_size, seq_len)
                - entropy: Per-position entropy of shape (batch_size, seq_len)
                - logits: Action logits of shape (batch_size, seq_len, action_dim)
        """
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        if masks.ndim == 1:
            masks = masks.unsqueeze(-1)

        hidden_states = self._get_hidden_states(obs)
        logits = self.action_head(hidden_states)

        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        if self.use_value_head and self.value_head is not None:
            values = self.value_head(hidden_states).squeeze(-1)
            values = values * masks
        else:
            values = torch.zeros_like(masks, dtype=torch.float32, device=obs.device)

        return values, action_log_probs, entropy, logits


class Block(nn.Module):
    """A transformer block with causal self-attention and MLP.

    This is based on the Block class from minGPT. It consists of:
    1. LayerNorm + Causal Self-Attention + residual connection
    2. LayerNorm + MLP (GELU activation) + residual connection

    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        embd_pdrop: Dropout for attention output
        resid_pdrop: Dropout for residual connections
        attn_pdrop: Dropout for attention weights
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        embd_pdrop: float,
        resid_pdrop: float,
        attn_pdrop: float,
    ):
        super().__init__()
        assert n_embd % n_head == 0

        # Causal self-attention
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop)

        # Layer norms
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

        # MLP with SiLU activation, note: original minGPT uses GELU
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=torch.nn.SiLU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Causal (masked) self-attention layer.

    This implements multi-head self-attention with a causal mask to ensure
    that each position can only attend to previous positions (leftward).
    Based on the implementation from minGPT.

    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        attn_pdrop: Dropout rate for attention weights
    """

    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float):
        super().__init__()
        assert n_embd % n_head == 0

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(attn_pdrop)
        # Number of heads and embedding dim per head
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through causal self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimension

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Attend to all preceding positions (causal mask)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask
        causal_mask = self._get_causal_mask(T, att.device, att.dtype)
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Aggregate values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    def _get_causal_mask(self, t: int, device: torch.device, dtype: torch.dtype):
        """Get causal mask for the current sequence length.

        Args:
            t: Current sequence length
            device: Device for the mask
            dtype: Data type for the mask

        Returns:
            Causal mask tensor of shape (1, 1, t, t)
        """
        # Create a triangular mask: 1 where i >= j (can attend), 0 otherwise
        mask = torch.tril(torch.ones(t, t, device=device, dtype=dtype))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, t, t)


class RotaryPositionalEmbeddings(nn.Module):
    """Rotary Positional Embeddings (RoPE).

    Encodes position information by rotating feature pairs in the 2D plane
    by position-dependent angles. This naturally incorporates explicit relative
    position dependency in dot-product attention.

    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021). arXiv:2104.09864

    For a pair of features (x_m^(1), x_m^(2)) at position m:
        RoPE(x_m, m) = [[cos(m*theta), -sin(m*theta)],
                         [sin(m*theta),  cos(m*theta)]] @ [x_m^(1), x_m^(2)]^T

    The dot-product between two positions m and n becomes:
        <RoPE(x_m, m), RoPE(x_n, n)> = <RoPE(x_m, m-n), RoPE(x_n, 0)>

    This shows that RoPE gives relative attention naturally.

    Args:
        dim: Number of features to apply rotary embedding to (must be even)
        base: Base frequency constant (default: 10000)
        rope_percentage: Fraction of features to apply RoPE to (default: 1.0)

    Example:
        >>> rope = RotaryPositionalEmbeddings(64)
        >>> q = torch.randn(4, 8, 16, 64)  # (batch, heads, seq_len, dim)
        >>> k = torch.randn(4, 8, 16, 64)
        >>> q_rope = rope(q)
        >>> k_rope = rope(k)
    """

    def __init__(self, dim: int, base: int = 10000, rope_percentage: float = 1.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim = dim
        self.base = base
        self.rotary_dim = int(dim * rope_percentage)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        """Build cached cos/sin values for the given sequence length."""
        seq_len = x.shape[-2]  # sequence length
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[-2]:
            return

        # theta_i = 10000^(-2*(i-1)/dim) for i = 1, 2, ..., dim/2
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        ).to(x.device)

        # Position indices
        seq_idx = torch.arange(seq_len, device=x.device).float()

        # idx_theta: (seq_len, rotary_dim//2)
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # Concatenate for pairing: (seq_len, rotary_dim)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)

        # Cache cos and sin: (1, 1, seq_len, rotary_dim) for proper broadcasting
        self.cos_cached = idx_theta2.cos()[None, None, :, :].to(x.dtype)
        self.sin_cached = idx_theta2.sin()[None, None, :, :].to(x.dtype)

    @staticmethod
    def _neg_half(x: torch.Tensor) -> torch.Tensor:
        """Compute [-x_{d/2+1}, ..., -x_d, x_1, ..., x_{d/2}]."""
        d_2 = x.shape[-1] // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embeddings.

        Args:
            x: Input tensor of shape (..., seq_len, dim) or
               (batch, heads, seq_len, dim)

        Returns:
            Rotated tensor of the same shape
        """
        self._build_cache(x)
        seq_len = x.shape[-2]

        # Split into rotary and pass-through features
        x_rope, x_pass = x[..., : self.rotary_dim], x[..., self.rotary_dim :]

        # Apply rotation: x_rope * cos + neg_half(x_rope) * sin
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[:, :, :seq_len, :]) + (
            neg_half_x * self.sin_cached[:, :, :seq_len, :]
        )

        return torch.cat((x_rope, x_pass), dim=-1)


class GroupedQuerySelfAttention(nn.Module):
    """Causal self-attention with Grouped-Query Attention (GQA).

    GQA reduces the number of key/value heads by grouping multiple query heads
    to share the same key and value projections. This reduces KV cache memory
    and improves inference efficiency while maintaining modeling quality.

    Compared to Multi-Head Attention (MHA):
    - MHA: n_query_heads == n_kv_heads (each head has its own K, V)
    - GQA: n_query_heads > n_kv_heads (groups share K, V)
    - MQA: n_kv_heads == 1 (all heads share single K, V)

    Used in LLaMA, Gemma, Qwen, and other modern LLMs.

    Args:
        n_embd: Embedding dimension
        n_head: Number of query attention heads
        n_kv_groups: Number of key-value groups (n_kv_heads)
        attn_pdrop: Dropout rate for attention weights
        use_rope: Whether to apply RoPE (default: True)
        rope_base: Base for RoPE frequency calculation (default: 10000)
        rope_percentage: Fraction of features for RoPE (default: 1.0)

    Example:
        >>> # 8 query heads, 2 KV groups (4x memory savings)
        >>> attn = GroupedQuerySelfAttention(n_embd=128, n_head=8, n_kv_groups=2)
        >>> x = torch.randn(4, 16, 128)
        >>> out = attn(x)  # (4, 16, 128)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_groups: int,
        attn_pdrop: float,
        use_rope: bool = True,
        rope_base: int = 10000,
        rope_percentage: float = 1.0,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_groups == 0, "n_head must be divisible by n_kv_groups"

        self.n_head = n_head
        self.n_kv_groups = n_kv_groups
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.use_rope = use_rope

        # Query projection: n_embd -> n_embd
        # K/V projections: n_embd -> n_kv_groups * head_dim (GQA optimization)
        self.c_attn = nn.Linear(n_embd, n_embd + 2 * self.n_kv_groups * self.head_dim)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(attn_pdrop)

        # RoPE for positional encoding (applied per-head, so use head_dim)
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(
                dim=self.head_dim, base=rope_base, rope_percentage=rope_percentage
            )
        else:
            self.rope = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GQA.

        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        B, T, C = x.size()

        # Calculate query, key, values for all heads in batch
        # q: n_embd, k: n_kv_groups * head_dim, v: n_kv_groups * head_dim
        q, k, v = self.c_attn(x).split(
            [
                self.n_embd,
                self.n_kv_groups * self.head_dim,
                self.n_kv_groups * self.head_dim,
            ],
            dim=2,
        )

        # Reshape for multi-head / multi-group
        # q: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # k, v: (B, T, n_kv_groups, head_dim) -> (B, n_kv_groups, T, head_dim)
        k = k.view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        if self.use_rope and self.rope is not None:
            q = self.rope(q)  # (B, n_head, T, head_dim)
            k = self.rope(k)  # (B, n_kv_groups, T, head_dim)

        # Expand K and V to match number of query heads
        # Each KV group is repeated to cover its assigned query heads
        n_repeats = self.n_head // self.n_kv_groups
        k = k.unsqueeze(1).expand(B, n_repeats, self.n_kv_groups, T, self.head_dim)
        v = v.unsqueeze(1).expand(B, n_repeats, self.n_kv_groups, T, self.head_dim)
        k = k.reshape(B, self.n_head, T, self.head_dim)
        v = v.reshape(B, self.n_head, T, self.head_dim)

        # Compute attention using SDPA for efficiency
        # SDPA handles causal masking internally
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,  # causal mask handled by SDPA via is_causal
            is_causal=True,  # causal mask: each position attends to previous only
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class GQABlock(nn.Module):
    """A transformer block with GQA and RMSNorm pre-normalization.

    This block uses:
    1. RMSNorm (pre-norm) + Grouped-Query Self-Attention + residual
    2. RMSNorm (pre-norm) + MLP (SiLU activation) + residual

    The pre-normalization pattern applies RMSNorm _before_ each sub-layer,
    which improves training stability in modern architectures.

    Args:
        n_embd: Embedding dimension
        n_head: Number of query attention heads
        n_kv_groups: Number of key-value groups for GQA
        resid_pdrop: Dropout for residual connections
        attn_pdrop: Dropout for attention weights
        use_rope: Whether to apply RoPE (default: True)
        rope_base: Base for RoPE frequency (default: 10000)
        rope_percentage: Fraction of features for RoPE (default: 1.0)
        rms_eps: Epsilon for RMSNorm (default: 1e-5)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_groups: int,
        resid_pdrop: float,
        attn_pdrop: float,
        use_rope: bool = True,
        rope_base: int = 10000,
        rope_percentage: float = 1.0,
        rms_eps: float = 1e-5,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_groups == 0

        # RMSNorm pre-normalization (pre-norm pattern)
        self.attn_norm = torch.nn.RMSNorm(n_embd, eps=rms_eps)
        self.ff_norm = torch.nn.RMSNorm(n_embd, eps=rms_eps)

        # GQA self-attention
        self.attn = GroupedQuerySelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            n_kv_groups=n_kv_groups,
            attn_pdrop=attn_pdrop,
            use_rope=use_rope,
            rope_base=rope_base,
            rope_percentage=rope_percentage,
        )

        # MLP with SiLU activation
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=torch.nn.SiLU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GQA transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        # Attention sub-layer with pre-norm
        x = x + self.attn(self.attn_norm(x))
        # Feedforward sub-layer with pre-norm
        x = x + self.mlpf(self.ff_norm(x))
        return x


class GPT2RoPEGQAPolicy(BasePolicy):
    """GPT-2 based policy with RMSNorm, RoPE, and Grouped-Query Attention.

    This policy upgrades the standard GPT2Policy with three modern techniques:

    1. **RMSNorm (Pre-Norm)**: Replaces LayerNorm with Root Mean Square Layer
       Normalization. No mean centering, only RMS normalization with a learnable
       scale parameter. Faster and uses fewer parameters than LayerNorm.

    2. **Rotary Positional Embeddings (RoPE)**: Replaces absolute positional
       embeddings with rotary encodings that naturally incorporate relative
       position information through 2D rotations.

    3. **Grouped-Query Attention (GQA)**: Replaces multi-head attention with
       grouped-query attention, where multiple query heads share key/value
       projections. Reduces KV cache memory and improves inference efficiency.

    Architecture (Pre-Norm pattern):
        x -> RMSNorm -> GQA -> + -> RMSNorm -> MLP -> + -> output

    Args:
        action_space: The action space (gymnasium.spaces.Discrete)
        observation_space: The observation space (gymnasium.spaces.Box)
        n_layer: Number of transformer layers. Default: 4
        n_head: Number of query attention heads. Default: 4
        n_kv_groups: Number of key-value groups for GQA. Default: 4
                     (set < n_head for GQA, = n_head for MHA, = 1 for MQA)
        n_embd: Embedding dimension. Default: 128
        block_size: Maximum sequence length. Default: 64
        embd_pdrop: Embedding dropout rate. Default: 0.1
        resid_pdrop: Residual dropout rate. Default: 0.1
        attn_pdrop: Attention dropout rate. Default: 0.1
        use_value_head: Whether to include a value head for RL. Default: False
        rope_base: Base frequency for RoPE. Default: 10000
        rope_percentage: Fraction of features for RoPE. Default: 1.0
        rms_eps: Epsilon for RMSNorm. Default: 1e-5

    Example:
        >>> from gymnasium import spaces
        >>> action_space = spaces.Discrete(10)
        >>> observation_space = spaces.Box(low=-1, high=1, shape=(768,))
        >>> policy = GPT2RoPEGQAPolicy(
        ...     action_space, observation_space,
        ...     n_layer=6, n_head=8, n_kv_groups=2, n_embd=256
        ... )
        >>> obs = torch.randn(8, 768)
        >>> logits = policy(obs)  # (8, 10)
    """

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_space: spaces.Box,
        weight: Optional[str] = None,
        n_layer: int = 4,
        n_head: int = 4,
        n_kv_groups: int = 4,
        n_embd: int = 128,
        block_size: int = 64,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        use_value_head: bool = False,
        rope_base: int = 10000,
        rope_percentage: float = 1.0,
        rms_eps: float = 1e-5,
    ):
        super().__init__(action_space, observation_space)

        self.action_dim = int(action_space.n)
        self.use_value_head = use_value_head
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_groups = n_kv_groups
        self.n_embd = n_embd
        self.block_size = block_size

        # Project observations to embedding dimension
        obs_dim = observation_space.shape[0]
        self.obs_projection = nn.Linear(obs_dim, n_embd)

        # Dropout
        self.drop = nn.Dropout(embd_pdrop)

        # Transformer blocks with GQA + RMSNorm + RoPE
        self.blocks = nn.ModuleList(
            [
                GQABlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    n_kv_groups=n_kv_groups,
                    resid_pdrop=resid_pdrop,
                    attn_pdrop=attn_pdrop,
                    use_rope=True,
                    rope_base=rope_base,
                    rope_percentage=rope_percentage,
                    rms_eps=rms_eps,
                )
                for _ in range(n_layer)
            ]
        )

        # Final RMSNorm (pre-norm pattern)
        self.ln_f = torch.nn.RMSNorm(n_embd, eps=rms_eps)

        # Action head - maps embeddings to action logits
        self.action_head = nn.Linear(n_embd, self.action_dim)

        # Optional value head - maps embeddings to scalar state values
        self.value_head = nn.Linear(n_embd, 1) if use_value_head else None

        # Initialize weights
        if weight is None:
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith("c_proj.weight"):
                    torch.nn.init.normal_(
                        p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer)
                    )
        else:
            self.load_state_dict(torch.load(weight))

    def _init_weights(self, module: nn.Module):
        """Initialize weights.

        - Linear layers: normal initialization with std=0.02
        - RMSNorm layers: weight = ones
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.RMSNorm):
            torch.nn.init.ones_(module.weight)

    def _get_hidden_states(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute shared transformer hidden states for observations.

        Args:
            obs: Observations of shape (batch_size, seq_len, obs_dim) or
                 (batch_size, obs_dim) for single-step inference.

        Returns:
            Hidden state tensor of shape (batch_size, seq_len, n_embd)
        """
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)

        batch_size, seq_len, obs_dim = obs.shape
        assert seq_len <= self.block_size, (
            f"Cannot forward sequence of length {seq_len}, "
            f"block_size is only {self.block_size}"
        )

        x = self.obs_projection(obs)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return x

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GPT-2 RoPE GQA policy.

        Args:
            obs: Observations of shape (batch_size, seq_len, obs_dim) or
                 (batch_size, obs_dim).

        Returns:
            Action logits of shape (batch_size, seq_len, action_dim)
        """
        hidden_states = self._get_hidden_states(obs)
        logits = self.action_head(hidden_states)
        return logits

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions taken under the policy.

        Args:
            obs: Observations of shape (batch_size, seq_len, obs_dim) or
                 (batch_size, obs_dim)
            actions: Actions taken of shape (batch_size, seq_len) or
                     (batch_size,)
            masks: Masks of shape (batch_size, seq_len) or (batch_size,)

        Returns:
            Tuple of (values, action_log_probs, entropy, logits):
                - values: Value predictions of shape (batch_size, seq_len)
                - action_log_probs: Log probs of taken actions of shape (batch_size, seq_len)
                - entropy: Per-position entropy of shape (batch_size, seq_len)
                - logits: Action logits of shape (batch_size, seq_len, action_dim)
        """
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        if masks.ndim == 1:
            masks = masks.unsqueeze(-1)

        hidden_states = self._get_hidden_states(obs)
        logits = self.action_head(hidden_states)

        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        if self.use_value_head and self.value_head is not None:
            values = self.value_head(hidden_states).squeeze(-1)
            values = values * masks
        else:
            values = torch.zeros_like(masks, dtype=torch.float32, device=obs.device)

        return values, action_log_probs, entropy, logits
