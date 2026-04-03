"""
TurboQuant compression core - Implementation based on arXiv:2504.19874

Implements:
- Random rotation preprocessing (simplifies vector geometry)
- PolarQuant for K-cache (magnitude + direction quantization)
- Residual QJL for V-cache (MSE quantization + 1-bit QJL on residual)
- Compressed attention: compute Q·K^T and apply V without full decompression
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# ============================================================================
# Compressed Attention Kernel (Phase 2)
# ============================================================================

def _dequantize_direction_fast(
    direction_uint8: torch.Tensor,
) -> torch.Tensor:
    """Fast dequantize direction from uint8 to normalized float32."""
    # [B, H, S, D] uint8 -> float32 in [-1, 1]
    direction = (direction_uint8.float() / 255.0) * 2 - 1
    # Normalize to unit length
    direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
    return direction


def _dequantize_magnitude_fast(
    magnitude_uint8: torch.Tensor,
    mag_min: float,
    mag_max: float,
) -> torch.Tensor:
    """Fast dequantize magnitude from uint8 to float32."""
    mag_range = mag_max - mag_min + 1e-8
    return (magnitude_uint8.float() / 255.0) * mag_range + mag_min


def _dequantize_v_primary(
    v_primary: torch.Tensor,
    v_min: torch.Tensor,
    v_max: torch.Tensor,
) -> torch.Tensor:
    """Dequantize primary V from 5-bit quantization."""
    n_levels = 32
    v_range = v_max - v_min + 1e-8
    return (v_primary.float() / (n_levels - 1)) * v_range + v_min


@dataclass
class CompressedKCache:
    """Container for PolarQuant-compressed K-cache.

    Storage format:
    - magnitude: 8-bit quantized L2 norm per token [batch, heads, seq_len]
    - direction: 8-bit quantized unit vector [batch, heads, seq_len, head_dim]
    - mag_min, mag_max: Global scale for magnitude dequantization
    """
    magnitude: torch.Tensor  # [B, H, S] uint8
    direction: torch.Tensor  # [B, H, S, D] uint8
    mag_min: float
    mag_max: float
    original_shape: Tuple[int, int, int, int]
    rotation_seed: int


@dataclass
class CompressedVCache:
    """Container for residual QJL-compressed V-cache.

    Two-stage compression:
    1. Primary: 5-bit min-max quantization
    2. Residual: 1-bit QJL sign
    """
    v_primary: torch.Tensor  # [B, H, S, D] uint8 (5-bit)
    v_min: torch.Tensor  # [B, H, S, 1] float32
    v_max: torch.Tensor  # [B, H, S, 1] float32
    v_residual_sign: torch.Tensor  # [B, H, S, sketch_dim] int8
    projection_matrix: torch.Tensor  # [sketch_dim, D] float32
    residual_scale: torch.Tensor  # [B, H, S, 1] float32
    original_shape: Tuple[int, int, int, int]
    rotation_seed: int


class TurboQuantCompressor:
    """TurboQuant KV cache compressor based on arXiv:2504.19874."""

    def __init__(
        self,
        bits: float = 3.5,
        k_bits: int = 8,
        v_primary_bits: int = 5,
        sketch_dim: Optional[int] = None,
    ):
        self.bits = bits
        self.k_bits = k_bits
        self.v_primary_bits = v_primary_bits
        self.sketch_dim = sketch_dim

    def get_compressed_size(self, compressed_data) -> int:
        """Calculate compressed data size in bytes."""
        if hasattr(compressed_data, 'magnitude'):  # CompressedKCache
            # K-cache: magnitude (8-bit) + direction (8-bit per dim)
            mag_bytes = compressed_data.magnitude.numel() * 1  # 1 byte per token
            dir_bytes = compressed_data.direction.numel() * 1  # 1 byte per coord
            # Plus scalar min/max (negligible)
            return mag_bytes + dir_bytes
        elif hasattr(compressed_data, 'v_primary'):  # CompressedVCache
            # V-cache: primary (5-bit ~ 1 byte) + residual sign (1-bit ~ 1 byte per projected dim)
            primary_bytes = compressed_data.v_primary.numel() * 1
            residual_bytes = compressed_data.v_residual_sign.numel() * 1
            proj_bytes = compressed_data.projection_matrix.numel() * 2  # float16
            scale_bytes = compressed_data.residual_scale.numel() * 2  # float16
            return primary_bytes + residual_bytes + proj_bytes + scale_bytes
        return 0

    def _random_rotate(
        self,
        x: torch.Tensor,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random orthogonal rotation."""
        *batch_dims, head_dim = x.shape

        generator = torch.Generator(device=x.device).manual_seed(seed)
        random_matrix = torch.randn(
            head_dim, head_dim, dtype=torch.float32, device=x.device, generator=generator
        )
        Q, _ = torch.linalg.qr(random_matrix)

        x_flat = x.reshape(-1, head_dim)
        x_rotated = torch.matmul(x_flat, Q)
        x_rotated = x_rotated.reshape(*batch_dims, head_dim)

        return x_rotated, Q

    def _inverse_rotate(
        self,
        x_rotated: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        """Apply inverse rotation (Q^T since Q is orthogonal)."""
        *batch_dims, head_dim = x_rotated.shape
        x_flat = x_rotated.reshape(-1, head_dim)
        x_orig = torch.matmul(x_flat, Q.t())
        return x_orig.reshape(*batch_dims, head_dim)

    def compress_k_cache(
        self,
        k: torch.Tensor,
        layer_idx: int,
    ) -> CompressedKCache:
        """Apply PolarQuant to key cache."""
        original_shape = tuple(k.shape)
        *batch_dims, head_dim = k.shape
        seed = layer_idx * 1000 + 42

        # Step 1: Random rotation
        k_rotated, Q = self._random_rotate(k.float(), seed)

        # Step 2: Polar decomposition
        magnitude = torch.norm(k_rotated, dim=-1)  # [B, H, S]
        eps = 1e-8
        direction = k_rotated / (magnitude.unsqueeze(-1) + eps)  # Unit vector

        # Step 3: Quantize magnitude (8 bits) - store global min/max
        mag_min = magnitude.min().item()
        mag_max = magnitude.max().item()
        mag_range = mag_max - mag_min + eps
        mag_normalized = (magnitude - mag_min) / mag_range
        mag_quant = torch.floor(mag_normalized * 255).clamp(0, 255).to(torch.uint8)

        # Step 4: Quantize direction (8 bits per coordinate)
        dir_normalized = (direction + 1) / 2  # [-1,1] -> [0,1]
        dir_quant = torch.floor(dir_normalized * 255).clamp(0, 255).to(torch.uint8)

        return CompressedKCache(
            magnitude=mag_quant,
            direction=dir_quant,
            mag_min=mag_min,
            mag_max=mag_max,
            original_shape=original_shape,
            rotation_seed=seed,
        )

    def compress_v_cache(
        self,
        v: torch.Tensor,
        layer_idx: int,
    ) -> CompressedVCache:
        """Apply residual QJL to value cache."""
        original_shape = tuple(v.shape)
        *batch_dims, head_dim = v.shape
        sketch_dim = self.sketch_dim or max(16, head_dim // 4)
        seed = layer_idx * 1000 + 100

        # Step 1: Random rotation
        v_rotated, Q = self._random_rotate(v.float(), seed)

        # Step 2: Primary quantization (5 bits per token)
        v_min = v_rotated.min(dim=-1, keepdim=True).values
        v_max = v_rotated.max(dim=-1, keepdim=True).values
        v_range = v_max - v_min + 1e-8

        n_levels = 32  # 5 bits
        normalized = (v_rotated - v_min) / v_range
        v_primary = torch.floor(normalized * (n_levels - 1)).clamp(0, n_levels - 1).to(torch.uint8)

        # Step 3: Compute residual
        v_dequant = (v_primary.float() / (n_levels - 1)) * v_range + v_min
        residual = v_rotated - v_dequant

        # Step 4: QJL on residual (1-bit)
        generator = torch.Generator(device=v.device).manual_seed(seed + 1000)
        jl_matrix = torch.randn(
            sketch_dim, head_dim, dtype=torch.float32, device=v.device, generator=generator
        )
        jl_matrix = jl_matrix / torch.sqrt(torch.tensor(sketch_dim, dtype=torch.float32, device=v.device))

        residual_proj = torch.matmul(residual, jl_matrix.t())
        residual_sign = torch.sign(residual_proj).to(torch.int8)
        residual_scale = torch.abs(residual).mean(dim=-1, keepdim=True)

        return CompressedVCache(
            v_primary=v_primary,
            v_min=v_min,
            v_max=v_max,
            v_residual_sign=residual_sign,
            projection_matrix=jl_matrix,
            residual_scale=residual_scale,
            original_shape=original_shape,
            rotation_seed=seed,
        )

    def decompress_k_cache(
        self,
        ck: CompressedKCache,
    ) -> torch.Tensor:
        """Decompress K-cache from PolarQuant."""
        *batch_dims, head_dim = ck.original_shape

        # Dequantize magnitude
        mag_range = ck.mag_max - ck.mag_min + 1e-8
        magnitude = (ck.magnitude.float() / 255.0) * mag_range + ck.mag_min

        # Dequantize direction and normalize
        direction = (ck.direction.float() / 255.0) * 2 - 1
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)

        # Reconstruct
        k_rotated = direction * magnitude.unsqueeze(-1)

        # Get rotation matrix and inverse rotate
        Q = self._get_rotation_matrix(head_dim, ck.rotation_seed, ck.direction.device)
        k = self._inverse_rotate(k_rotated, Q)

        return k.half()

    def decompress_v_cache(
        self,
        cv: CompressedVCache,
    ) -> torch.Tensor:
        """Decompress V-cache from residual QJL."""
        *batch_dims, head_dim = cv.original_shape

        # Dequantize primary
        n_levels = 32
        v_range = cv.v_max - cv.v_min + 1e-8
        v_dequant = (cv.v_primary.float() / (n_levels - 1)) * v_range + cv.v_min

        # Reconstruct residual from QJL
        residual_recon = torch.matmul(
            cv.v_residual_sign.float(),
            cv.projection_matrix
        ) * (cv.residual_scale * torch.sqrt(torch.tensor(torch.pi / 2, device=cv.residual_scale.device)))

        # Combine
        v_rotated = v_dequant + residual_recon

        # Inverse rotation
        Q = self._get_rotation_matrix(head_dim, cv.rotation_seed, cv.v_primary.device)
        v = self._inverse_rotate(v_rotated, Q)

        return v.half()

    def _get_rotation_matrix(
        self,
        head_dim: int,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Reconstruct rotation matrix from seed."""
        generator = torch.Generator(device=device).manual_seed(seed)
        random_matrix = torch.randn(
            head_dim, head_dim, dtype=torch.float32, device=device, generator=generator
        )
        Q, _ = torch.linalg.qr(random_matrix)
        return Q

    def compute_attention_scores_compressed(
        self,
        query: torch.Tensor,
        ck: CompressedKCache,
    ) -> torch.Tensor:
        """Compute attention scores Q · K^T using COMPRESSED K (no full decompression).

        Key insight from paper: For PolarQuant K-cache:
        - K_rotated ≈ magnitude * direction (in rotated space)
        - Q · K = Q_rotated · K_rotated (rotation preserves inner products)
        - Q_rotated · (magnitude * direction) = magnitude * (Q_rotated · direction)

        Steps:
        1. Rotate query with same rotation matrix
        2. Dequantize K direction (lightweight)
        3. Compute Q_rotated · direction^T
        4. Scale by K magnitude

        This avoids materializing full K tensor.
        """
        *batch_dims, head_dim = ck.original_shape

        # Step 1: Get rotation matrix and rotate query (match dtypes)
        Q = self._get_rotation_matrix(head_dim, ck.rotation_seed, query.device)
        query_float = query.float()
        query_flat = query_float.reshape(-1, head_dim)
        query_rotated = torch.matmul(query_flat, Q)  # Rotate query
        query_rotated = query_rotated.reshape(*query.shape[:-1], head_dim)

        # Step 2: Dequantize K direction (still need this, but it's smaller than full K)
        k_direction = _dequantize_direction_fast(ck.direction)  # [B, H, S, D]

        # Step 3: Compute Q · direction^T
        # query_rotated: [B, H, seq_q, D]
        # k_direction: [B, H, seq_k, D]
        # Result: [B, H, seq_q, seq_k]
        scores = torch.matmul(query_rotated, k_direction.transpose(-2, -1))

        # Step 4: Scale by K magnitude
        k_magnitude = _dequantize_magnitude_fast(ck.magnitude, ck.mag_min, ck.mag_max)
        # k_magnitude: [B, H, S] -> need to broadcast to [B, H, seq_q, seq_k]
        # Each query position attends to all key positions, scale by key magnitude
        k_magnitude_scaled = k_magnitude.unsqueeze(-2)  # [B, H, 1, seq_k]
        scores = scores * k_magnitude_scaled

        # Scale by head_dim (standard attention scaling)
        scores = scores / (head_dim ** 0.25)

        return scores

    def compute_attention_scores(
        self,
        query: torch.Tensor,
        ck: CompressedKCache,
    ) -> torch.Tensor:
        """Compute attention scores Q · K^T.

        Uses compressed attention if available, otherwise falls back to decompression.
        """
        return self.compute_attention_scores_compressed(query, ck)

    def apply_compressed_v_fast(
        self,
        attention_weights: torch.Tensor,
        cv: CompressedVCache,
    ) -> torch.Tensor:
        """Apply compressed V to attention weights WITHOUT full V decompression.

        Key insight from paper: For residual QJL V-cache:
        - V = V_primary + V_residual
        - V_residual ≈ sqrt(π/2) * S^T * sign(S · residual)

        So: attention_weights @ V = attention_weights @ V_primary + attention_weights @ V_residual

        For V_primary: dequantize per-token (cheaper than full)
        For V_residual: use JL property to compute projection directly
        """
        *batch_dims, head_dim = cv.original_shape
        sketch_dim = cv.v_residual_sign.shape[-1]

        # Step 1: Dequantize V_primary (still need this)
        v_primary_dequant = _dequantize_v_primary(cv.v_primary, cv.v_min, cv.v_max)

        # Step 2: Apply attention weights to V_primary
        # attention_weights: [B, H, seq_q, seq_k]
        # v_primary_dequant: [B, H, seq_k, D]
        # Result: [B, H, seq_q, D]
        output_primary = torch.matmul(attention_weights, v_primary_dequant)

        # Step 3: Apply attention weights to V_residual using QJL
        # V_residual ≈ scale * sqrt(π/2) * S^T * sign
        # attention_weights @ V_residual ≈ scale * sqrt(π/2) * attention_weights @ S^T @ sign
        # = scale * sqrt(π/2) * (attention_weights @ S^T) @ sign
        # But sign is [B, H, seq_k, sketch_dim], we need to apply attention first

        # Residual sign: [B, H, seq_k, sketch_dim]
        # attention_weights: [B, H, seq_q, seq_k]
        # attention_weights @ sign: [B, H, seq_q, sketch_dim]
        attn_residual = torch.matmul(attention_weights, cv.v_residual_sign.float())

        # Now apply S^T: [B, H, seq_q, sketch_dim] @ [sketch_dim, D] = [B, H, seq_q, D]
        residual_output = torch.matmul(attn_residual, cv.projection_matrix)

        # Scale by residual scale and sqrt(π/2)
        # cv.residual_scale: [B, H, S, 1] where S = seq_k
        scale_factor = cv.residual_scale.squeeze(-1) * torch.sqrt(torch.tensor(torch.pi / 2, device=cv.residual_scale.device))
        # scale_factor: [B, H, S]

        # Apply attention weights to get per-query scale
        # attention_weights: [B, H, seq_q, seq_k]
        # scale_factor: [B, H, seq_k]
        # Result: [B, H, seq_q]
        scale_weighted = torch.matmul(attention_weights, scale_factor.unsqueeze(-1)).squeeze(-1)

        output_residual = residual_output * scale_weighted.unsqueeze(-1)

        # Step 4: Combine primary and residual
        output = output_primary + output_residual

        # Step 5: Inverse rotation
        Q = self._get_rotation_matrix(head_dim, cv.rotation_seed, output.device)
        output_flat = output.reshape(-1, head_dim)
        output_rotated = torch.matmul(output_flat, Q.t())
        output = output_rotated.reshape(*output.shape[:-1], head_dim)

        return output.half()

    def apply_compressed_v(
        self,
        attention_weights: torch.Tensor,
        cv: CompressedVCache,
    ) -> torch.Tensor:
        """Apply compressed V to attention weights."""
        return self.apply_compressed_v_fast(attention_weights, cv)

    def compress_for_attention(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[CompressedKCache, CompressedVCache]:
        """Compress KV cache for attention."""
        ck = self.compress_k_cache(k, layer_idx)
        cv = self.compress_v_cache(v, layer_idx)
        return ck, cv
