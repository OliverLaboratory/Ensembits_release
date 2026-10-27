# =============================================================================
# geom_attn_encoder.py  —  NOT USED by the model reported in the paper.
#
# What: Learned geometric-attention pre-encoder over the 16 neighbour frames (trainer option --geom-attn-encoder).
# Paper: not described in the camera-ready paper.
# Origin: copied from `Ensembits/ensembits/tokenizer/geom_attn_encoder.py` (unchanged). Kept only
#   because train/train.py still exposes the option and imports it.
# =============================================================================
"""Geometric-attention pre-encoder over K=16 nearest neighbors.

Wraps AminoAseed/ESM3's `VanillaTransformerStack` (with
`use_geom_attn=True`, `use_plain_attn=False`) so we can plug it before
our existing SetTransformer encoder.

Architecture (per residue, per frame):
    K=16 relative-Affine3D tokens (12-D each, token 0 = identity = query-relative-to-self)
        ↓
    Initial feature = learned rank-positional embedding (one per kNN rank 0..15)
        ↓
    N layers of VanillaUnifiedTransformerBlock(use_geom_attn=True, use_plain_attn=False)
        each layer computes per-head rotation-similarity + distance-similarity attention
        biases (NOT plain dot-product) from the Affine3D — see
        StructTokenBench/src/vqvae/geom_attention.py.
        ↓
    Take token 0 (query) embedding → (d_model)

The geometric attention's `query.rot.T @ key.rot` and `‖query.apply(v) − key.apply(v)‖`
terms are invariant to a shared left-multiplication of all tokens' Affine3D
by an arbitrary rigid transformation. Our cache stores relative-to-query
Affine3D (rel_i = q_query^{-1} ∘ q_i), so passing those as "affine" to the
geometric block gives the same attention weights as if we had passed
absolute Affine3D — `q_query^{-1}` cancels in both terms. Verified by hand:

    rotation_term:  rel_i.rot.T @ rel_j.rot
        = (q⁻¹ rot @ abs_i.rot).T @ (q⁻¹ rot @ abs_j.rot)
        = abs_i.rot.T @ q⁻¹.rot.T @ q⁻¹.rot @ abs_j.rot
        = abs_i.rot.T @ abs_j.rot                      ✓ (q⁻¹ rot orthogonal cancels)

    distance_term:  ‖rel_i.apply(v_i) − rel_j.apply(v_j)‖
        = ‖q⁻¹.apply(abs_i.apply(v_i)) − q⁻¹.apply(abs_j.apply(v_j))‖
        = ‖q⁻¹.rot @ (abs_i.apply(v_i) − abs_j.apply(v_j))‖
        = ‖abs_i.apply(v_i) − abs_j.apply(v_j)‖        ✓ (q⁻¹ is rigid → preserves distances)

So no need to recompute absolute Affine3D / kNN indices — the existing
`esm3desc_K16_P10.npy` cache reshaped as (N, P, 16, 12) is the right input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# StructTokenBench's geometric transformer + Affine3D from the esm package.
# Both should already be on sys.path in the analysis environment; in other envs
# we add them explicitly.
from ensembits.paths import STRUCTTOKENBENCH_SRC as _STB_PATH   # data/external/StructTokenBench/src
_STB = str(_STB_PATH)
if _STB not in sys.path:
    sys.path.insert(0, _STB)

# StructTokenBench imports `deepspeed` unconditionally for some training
# utilities even when we only use the inference-time encoder block. Stub it
# in case it isn't installed in the current env.
if "deepspeed" not in sys.modules:
    import types
    _ds = types.ModuleType("deepspeed")
    _ds.utils = types.SimpleNamespace(safe_get_full_grad=lambda p: None)
    sys.modules["deepspeed"] = _ds

from esm.utils.structure.affine3d import Affine3D  # noqa: E402
from vqvae.transformer_stack import VanillaTransformerStack  # noqa: E402


class GeometricKNNEncoder(nn.Module):
    """K=16 geometric self-attention block over relative-Affine3D tokens.

    Input
    -----
    neigh_12 : Tensor (B, P, K=16, 12) float32
        Per-(residue, frame, neighbor) relative-Affine3D in Affine3D.tensor
        format (3×3 rot flattened to 9 floats + 3-D translation = 12). Token 0
        is identity (the query relative to itself). Tokens 1..K-1 are the
        K-1 spatially-nearest neighbors of the query, relative to the query
        frame, in distance-sorted order.

    Output
    ------
    Tensor (B, P, d_model) — query-token embedding after `n_layers` geometric
    self-attention blocks.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        v_heads: int = 8,
        n_layers: int = 2,
        n_neighbors: int = 16,
        ffn_type: str = "swiglu",
        expansion_ratio: float = 8 / 3,
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.d_model = d_model
        # Learned per-rank positional embedding (rank 0 = query, rank i = i-th
        # nearest neighbor). Replaces AminoAseed's residue-index relative
        # positional embedding since we don't store residue indices in our cache.
        self.rank_emb = nn.Embedding(n_neighbors, d_model)
        # The geometric-attention stack itself.
        self.stack = VanillaTransformerStack(
            d_model=d_model,
            n_heads=n_heads,
            v_heads=v_heads,
            n_layers=n_layers,
            n_layers_geom=n_layers,        # all layers geometric, no plain attn
            scale_residue=False,
            mask_and_zero_frameless=False,
            bias=False,
            qk_layernorm=True,
            ffn_type=ffn_type,
            expansion_ratio=expansion_ratio,
        )

    def forward(self, neigh_12: torch.Tensor) -> torch.Tensor:
        B, P, K, D12 = neigh_12.shape
        assert K == self.n_neighbors, f"expected K={self.n_neighbors} got {K}"
        assert D12 == 12, f"expected 12-D Affine3D tensor got {D12}"

        device = neigh_12.device
        flat = neigh_12.reshape(B * P, K, D12)

        # Initial features: same rank embedding broadcast to every (B*P).
        ranks = torch.arange(K, device=device)
        x = self.rank_emb(ranks).unsqueeze(0).expand(B * P, K, self.d_model).contiguous()

        # Construct Affine3D objects from the 12-D tensors.
        # Affine3D.from_tensor expects (..., 12) and reads rot[:9] + trans[9:].
        affine = Affine3D.from_tensor(flat)            # (B*P, K) of Affine3D
        # All tokens valid (we don't pad).
        attn_mask = torch.ones((B * P, K), dtype=torch.bool, device=device)
        affine_mask = torch.ones((B * P, K), dtype=torch.bool, device=device)
        sequence_id = torch.zeros((B * P, K), dtype=torch.int64, device=device)
        chain_id = torch.zeros((B * P, K), dtype=torch.int64, device=device)

        x_post, _ = self.stack(
            x,
            attention_mask=attn_mask,
            sequence_id=sequence_id,
            affine=affine,
            affine_mask=affine_mask,
            chain_id=chain_id,
        )                                              # (B*P, K, d_model)

        # Take the query token's embedding.
        out = x_post[:, 0, :]                           # (B*P, d_model)
        return out.view(B, P, self.d_model)


__all__ = ["GeometricKNNEncoder"]
