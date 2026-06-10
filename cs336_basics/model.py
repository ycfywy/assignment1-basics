"""
CS336 Assignment 1 - Section 3: Transformer Language Model Implementation.

Implements all components needed for a GPT-style language model:
- Linear (no bias)
- Embedding
- RMSNorm
- SiLU activation
- SwiGLU FFN
- Scaled Dot-Product Attention
- Rotary Positional Embeddings (RoPE)
- Multi-Head Self-Attention (with and without RoPE)
- Transformer Block (pre-norm)
- Transformer Language Model
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class Linear(nn.Module):
    """Linear layer without bias: y = xW^T"""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        # Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    """Embedding lookup table."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def silu(x: Tensor) -> Tensor:
    """SiLU (Swish) activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    SwiGLU(x) = W2 * (SiLU(W1 * x) ⊙ W3 * x)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
) -> Tensor:
    """Scaled Dot-Product Attention.

    Args:
        Q: (..., queries, d_k)
        K: (..., keys, d_k)
        V: (..., keys, d_v)
        mask: (..., queries, keys) boolean mask. True = keep, False = mask out.

    Returns:
        (..., queries, d_v)
    """
    d_k = Q.shape[-1]
    # (..., queries, keys)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    # Handle the case where all values in a row are -inf (all masked)
    attn_weights = attn_weights.nan_to_num(0.0)

    return attn_weights @ V


class RoPE(nn.Module):
    """Rotary Positional Embedding."""

    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency bands: theta_i = theta^(-2i/d_k) for i=0,...,d_k/2-1
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        # Shape: (max_seq_len, d_k/2)
        positions = torch.arange(max_seq_len).float()
        # Outer product: (max_seq_len, d_k/2)
        angles = torch.outer(positions, freqs)

        # Precompute cos and sin
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len) integer positions

        Returns:
            (..., seq_len, d_k) with rotary embeddings applied
        """
        # Get cos/sin for the given positions
        # token_positions: (..., seq_len) -> cos/sin: (..., seq_len, d_k/2)
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k/2)

        # Split x into pairs: x = [x0, x1, x2, x3, ...] -> [x0,x1], [x2,x3], ...
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]   # (..., seq_len, d_k/2)

        # Apply rotation:
        # [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # Interleave back
        out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention (without RoPE)."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., seq_len, d_model)
        Returns:
            (..., seq_len, d_model)
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        # Project Q, K, V
        Q = self.q_proj(x)  # (..., seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (..., num_heads, seq_len, d_k)
        Q = Q.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Apply attention
        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Reshape back: (..., num_heads, seq_len, d_k) -> (..., seq_len, d_model)
        attn_out = attn_out.transpose(-3, -2).contiguous().view(*batch_shape, seq_len, self.d_model)

        return self.output_proj(attn_out)


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """Multi-Head Self-Attention with Rotary Positional Embeddings."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        self.rope = RoPE(self.d_k, max_seq_len, theta)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (..., seq_len, d_model)
            token_positions: (..., seq_len) or None (defaults to 0..seq_len-1)
        Returns:
            (..., seq_len, d_model)
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Project Q, K, V
        Q = self.q_proj(x)  # (..., seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (..., num_heads, seq_len, d_k)
        Q = Q.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)

        # Apply RoPE to Q and K
        # token_positions: (..., seq_len) -> need to expand for heads
        # RoPE expects (..., seq_len, d_k), positions (..., seq_len)
        # Q/K are (..., num_heads, seq_len, d_k), positions should be (..., 1, seq_len) for broadcast
        pos_for_rope = token_positions.unsqueeze(-2)  # (..., 1, seq_len)
        Q = self.rope(Q, pos_for_rope)
        K = self.rope(K, pos_for_rope)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Apply attention
        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Reshape back
        attn_out = attn_out.transpose(-3, -2).contiguous().view(*batch_shape, seq_len, self.d_model)

        return self.output_proj(attn_out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer Block with RoPE.

    Structure:
        x -> LN1 -> MHA (with RoPE) -> + x -> LN2 -> SwiGLU FFN -> + residual
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x), token_positions)
        # Pre-norm FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Transformer Language Model.

    Structure:
        token_ids -> Embedding -> [TransformerBlock x N] -> RMSNorm -> LM Head -> logits
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: (batch_size, seq_len)
        Returns:
            (batch_size, seq_len, vocab_size) unnormalized logits
        """
        seq_len = token_ids.shape[-1]
        # Token positions: 0, 1, ..., seq_len - 1
        token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)

        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
