# SPDX-License-Identifier: BSD-3-Clause
"""
GNN policy model for graph observations.

Processes graph (node_features, edge_index, mask, self_node_idx) with message passing,
then uses the self node's embedding (after message passing) for action logits and value.

Agent embedding flow:
  1. Apply mask to raw node features (zero disabled slots) before the node MLP.
  2. Node embedding: (B, N, F) -> linear+ReLU+linear -> (B, N, H); zero disabled slots again (bias leak).
  3. Message passing: each layer masks embeddings before gather; each message is multiplied by
     mask[src] so disabled sources send no signal (ReLU+Linear bias would otherwise leak).
     Mean aggregation; residual h = h + agg; then h *= mask again.
  4. Self embedding: take the node at self_node_idx -> (B, H).
  5. Policy / value heads from self_emb.

  Masking before aggregation is required when many of N nodes are padding/disabled (e.g. 1v1 with
  10 inactive slots of 12): otherwise inactive neighbors corrupt every node's embedding during MP.
"""

import numpy as np
import torch
import torch.nn as nn

from pyquaticus.envs.graph_obs_wrapper import MAX_AGENTS, NODE_FEAT_DIM

try:
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.utils.annotations import override
    from ray.rllib.utils.framework import try_import_torch
    _HAS_RAY = True
except ImportError:
    TorchModelV2 = object
    override = lambda f: f
    try_import_torch = lambda: (None, None)
    _HAS_RAY = False

torch, _ = try_import_torch()

# Flattened size when Dict is flattened (key order: node_features, edge_index, mask, self_node_idx)
_NUM_EDGES = MAX_AGENTS * (MAX_AGENTS - 1)
_FLAT_NODE = MAX_AGENTS * NODE_FEAT_DIM
_FLAT_EDGE = 2 * _NUM_EDGES
_FLAT_MASK = MAX_AGENTS
_FLAT_SELF = 1


def _fixed_all_to_all_src_dst():
    """Same topology as graph_obs_wrapper._build_edge_index (fully connected, no self-loops)."""
    src, dst = [], []
    for i in range(MAX_AGENTS):
        for j in range(MAX_AGENTS):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)


class GNNModel(TorchModelV2, nn.Module):
    """GNN: node embed -> message passing -> self-node embedding -> policy/value heads. See module docstring for flow."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)
        self.num_outputs = num_outputs
        self._value_out = None

        hidden = int(model_config.get("custom_model_config", {}).get("gnn_hidden", 64))
        num_layers = int(model_config.get("custom_model_config", {}).get("gnn_layers", 2))

        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(NODE_FEAT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Linear(hidden * 2, hidden) for _ in range(num_layers)
        ])
        # Policy head (from self embedding)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_outputs),
        )
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Fixed all-to-all topology (matches GraphObsWrapper): avoids per-forward edge parsing and GPU H2D for edges.
        fs, fd = _fixed_all_to_all_src_dst()
        self.register_buffer("_fixed_src_idx", fs)
        self.register_buffer("_fixed_dst_idx", fd)
        # Each node has exactly (MAX_AGENTS - 1) incoming edges in the full graph; mean aggregation divisor is constant.
        self.register_buffer("_inv_in_degree", torch.tensor(1.0 / float(MAX_AGENTS - 1)))

        # Per-process (each Ray worker runs this once): Tensor Cores + cudnn autotune for steady batch shapes.
        if torch is not None and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

    def _unflatten_obs(self, obs):
        """Unflatten preprocessor output: order is node_features, edge_index, mask, self_node_idx (Dict order)."""
        B = obs.shape[0]
        o = obs.view(B, -1)
        i = 0
        nf_flat = o[:, i : i + _FLAT_NODE]
        i += _FLAT_NODE
        edge_flat = o[:, i : i + _FLAT_EDGE]
        i += _FLAT_EDGE
        mask = o[:, i : i + _FLAT_MASK]
        i += _FLAT_MASK
        self_idx = o[:, i : i + _FLAT_SELF].long()
        node_features = nf_flat.reshape(B, MAX_AGENTS, NODE_FEAT_DIM)
        edge_index = edge_flat.reshape(B, 2, _NUM_EDGES)
        return node_features, edge_index, mask, self_idx.clamp(0, MAX_AGENTS - 1)

    @override(TorchModelV2)
    def forward(self, input_dict, state_batches, seq_lens):
        obs = input_dict["obs"]
        dev = next(self.parameters()).device
        nb = dev.type == "cuda"
        if isinstance(obs, dict):
            node_features = obs["node_features"]
            mask = obs["mask"]
            self_node_idx = obs["self_node_idx"]
            if isinstance(node_features, np.ndarray):
                node_features = torch.as_tensor(node_features, dtype=torch.float32, device=dev, non_blocking=nb)
            else:
                node_features = node_features.to(device=dev, non_blocking=nb)
            if isinstance(mask, np.ndarray):
                mask = torch.as_tensor(mask, dtype=torch.float32, device=dev, non_blocking=nb)
            else:
                mask = mask.to(device=dev, non_blocking=nb)
            if isinstance(self_node_idx, np.ndarray):
                self_node_idx = torch.as_tensor(self_node_idx, dtype=torch.long, device=dev, non_blocking=nb)
            else:
                self_node_idx = self_node_idx.to(device=dev, non_blocking=nb)
        else:
            node_features, _edge_index, mask, self_node_idx = self._unflatten_obs(obs)
            node_features = node_features.to(device=dev, non_blocking=nb)
            mask = mask.to(device=dev, non_blocking=nb)
            self_node_idx = self_node_idx.to(device=dev, non_blocking=nb)

        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        B = node_features.shape[0]
        device = node_features.device

        # mask: 1 = active, 0 = disabled/padding — apply before embed so inactive nodes are not encoded
        mask_exp = mask.unsqueeze(-1)  # (B, N, 1)
        node_features = node_features * mask_exp

        # Embed nodes: (B, N, F) -> (B, N, H); zero again in case Linear has bias
        x = self.node_embed(node_features)
        H = x.size(-1)
        x = x * mask_exp

        # Fixed all-to-all edges (same as env); avoids scatter-based degree counts every layer.
        src_idx = self._fixed_src_idx.unsqueeze(0).expand(B, -1)
        dst_idx = self._fixed_dst_idx.unsqueeze(0).expand(B, -1)
        inv_deg = self._inv_in_degree.to(dtype=x.dtype)
        mask_src = torch.gather(mask, 1, src_idx).unsqueeze(-1)  # (B, E, 1); reused each MP layer
        dst_expand = dst_idx.unsqueeze(-1).expand(-1, -1, H)

        # Message passing (vectorized aggregation per batch item)
        for layer in self.message_layers:
            x = x * mask_exp
            src_feat = torch.gather(x, 1, src_idx.unsqueeze(-1).expand(-1, -1, H))
            dst_feat = torch.gather(x, 1, dst_idx.unsqueeze(-1).expand(-1, -1, H))
            msg = torch.relu(layer(torch.cat([src_feat, dst_feat], dim=-1)))
            msg = msg * mask_src
            agg = torch.zeros(B, MAX_AGENTS, H, device=device, dtype=x.dtype)
            agg.scatter_add_(1, dst_expand, msg)
            active_count = torch.zeros(B, MAX_AGENTS, 1, device=device, dtype=x.dtype)
            active_count.scatter_add_(1, dst_idx.unsqueeze(-1), mask_src)
            agg = agg / active_count.clamp(min=1)
            x = x + agg
            x = x * mask_exp

        # Self embedding: for each batch item, take node at self_node_idx (int64 for indexing)
        self_idx = self_node_idx.view(B).long()
        if self_idx.dim() == 2:
            self_idx = self_idx.squeeze(-1)
        self_emb = x[torch.arange(B, device=device), self_idx.clamp(0, MAX_AGENTS - 1)]  # (B, H)

        logits = self.policy_head(self_emb)
        self._value_out = self.value_head(self_emb).squeeze(-1)
        return logits, state_batches

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out
