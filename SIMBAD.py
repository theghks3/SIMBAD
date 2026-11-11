import torch
import torch.nn as nn
import torch.nn.functional as F

from uncertainty import UncertaintyAwareAdjacency, UncertaintyProcessor

# ============================================================
#  Utility / adjacency modules
# ============================================================

class BatchSymNorm(nn.Module):
    """Symmetric degree normalization to batched adjacency matrices."""

    def __init__(self):
        super().__init__()

    def forward(self, adj):
        """Normalize adjacency matrices with shape (B, N, N)."""
        # Degree: (B, N)
        deg = adj.sum(dim=2)  # sum over rows → degree for each node

        # D^{-1/2}: (B, N)
        deg_inv_sqrt = torch.pow(deg + 1e-10, -0.5)

        # Reshape for broadcasting: (B, N, 1) and (B, 1, N)
        D_left = deg_inv_sqrt.unsqueeze(2)     # (B, N, 1)
        D_right = deg_inv_sqrt.unsqueeze(1)    # (B, 1, N)

        # Normalize: A_hat = D^{-1/2} A D^{-1/2}
        adj_norm = D_left * adj * D_right      # broadcasting multiplication

        return adj_norm  # shape: (B, N, N)

class Adjacency_Weight(nn.Module):
    """Build a weighted adjacency matrix from recent L1 distances."""

    def __init__(self, similarity_window):
        super().__init__()
        self.similarity_window = similarity_window

    def forward(self, x, y, adj, threshold, scale_input, scale_sim):
        x = x[..., -self.similarity_window:]
        y = y[..., -self.similarity_window:]

        x_inv = scale_input.inverse_transform(x).flatten(start_dim=2)
        y_inv = scale_input.inverse_transform(y).flatten(start_dim=2)

        l1_adjacency = scale_sim.transform(torch.cdist(x_inv, y_inv, p=1))

        l1_thresh_adj = torch.sigmoid(threshold - l1_adjacency)

        weighted_matrix = l1_thresh_adj * adj.to(x.device)

        return weighted_matrix

class AdjacencyRowSoftmax(nn.Module):
    """Row-wise softmax while preserving zero-valued edge masks."""

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        mask = adj != 0
        adj_masked = adj.masked_fill(~mask, float('-inf'))
        out = F.softmax(adj_masked, dim=-1)
        out = torch.nan_to_num(out, nan=0.0)
        return out


class AperiodicityAwareModulationLayer(nn.Module):

    def __init__(
        self,
        adjacency,
        scale_input,
        scale_sim,
        similarity_window,
        num_candidates,
        num_for_prediction,
        input_dim,
    ):
        super().__init__()
        self.adjacency = adjacency
        self.scale_input = scale_input
        self.scale_sim = scale_sim
        self.num_candidates = num_candidates
        self.num_for_prediction = num_for_prediction
        self.input_dim = input_dim
        self.weight_adjacency = Adjacency_Weight(similarity_window)
        self.row_softmax = AdjacencyRowSoftmax()
        self.adjacency_norm = BatchSymNorm()

    def forward(
        self,
        x_week,
        x_day,
        x_hour,
        input_similarity,
        threshold,
        uncertainty_processor,
        uncertainty_adjacency,
    ):
        """Modulate historical inputs and construct semantic adjacencies."""
        processing_output = uncertainty_processor(input_similarity)
        probabilities = processing_output['probs']
        predicted_similarities = processing_output['pred_sims']
        uncertainty = processing_output['uncertainty']
        confidence = processing_output['uncertainty_adj']

        historical = torch.cat([x_week, x_day], dim=-1)
        historical = historical.reshape(
            historical.shape[0],
            historical.shape[1],
            self.input_dim,
            self.num_candidates,
            self.num_for_prediction,
        )
        aggregated_history = (
            historical * probabilities.unsqueeze(2).unsqueeze(-1)
        ).sum(dim=3).float()

        expected_similarity = (
            probabilities * predicted_similarities
        ).sum(dim=-1, keepdim=True).float()
        expected_confidence = (
            probabilities * confidence
        ).sum(dim=-1, keepdim=True).float()
        historical_mask = (
            torch.sigmoid(threshold - expected_similarity)
            * expected_confidence
        ).unsqueeze(-1)
        modulated_history = aggregated_history * historical_mask

        historical_adjacency = uncertainty_adjacency(
            predicted_similarities,
            uncertainty,
        ).float()
        recent_adjacency = self.row_softmax(
            self.weight_adjacency(
                x_hour,
                x_hour,
                self.adjacency,
                threshold,
                self.scale_input,
                self.scale_sim,
            )
        ).float()
        recent_adjacency = self.adjacency_norm(
            recent_adjacency
        ).float().unsqueeze(1)

        return {
            'historical_input': modulated_history,
            'historical_adjacency': historical_adjacency.unsqueeze(1),
            'recent_adjacency': recent_adjacency,
            'historical_mask': historical_mask.float(),
            'loss_compute': processing_output['loss_compute'],
        }


# ============================================================
#  Linear / conv helpers
# ============================================================

class FC_Linear(nn.Module):
    """Pointwise linear projection with a 1x1 convolution."""

    def __init__(self, c_in, c_out):
        super(FC_Linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


# ============================================================
#  Temporal module  (GTU)
# ============================================================

class GTU(nn.Module):
    """Stacked gated temporal convolutions."""

    def __init__(self, in_channels, out_channels):
        super(GTU, self).__init__()
        self.temp_conv   = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.gate_conv   = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.temp_conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.gate_conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

    def forward(self, x):
        """Transform features with shape (B, N, F, T)."""
        x_conv = x.permute(0, 2, 1, 3)          # B, F, N, T

        temp = self.temp_conv(x_conv)
        gate = self.gate_conv(x_conv)
        h    = torch.tanh(temp) * torch.sigmoid(gate)  # B, F, N, T

        temp2 = self.temp_conv_2(h)
        gate2 = self.gate_conv_2(h)
        out   = torch.tanh(temp2) * torch.sigmoid(gate2)  # B, F, N, T

        return out.permute(0, 2, 1, 3)  # B, N, F, T


class Temporal_self_att(nn.Module):
    """Multi-head self-attention along the time axis."""

    def __init__(self, f_in, f_out, dropout, num_heads=4):
        super(Temporal_self_att, self).__init__()
        if f_out % num_heads != 0:
            raise ValueError("f_out must be divisible by num_heads")
        self.head_dim = f_out // num_heads
        self.num_heads = num_heads

        self.fc_q   = nn.Linear(f_in, f_out)
        self.fc_k   = nn.Linear(f_in, f_out)
        self.fc_v   = nn.Linear(f_in, f_out)
        self.fc_out = nn.Linear(f_out, f_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Transform features with shape (B, N, F, T)."""
        B, N, _, T = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()  # B, N, T, F

        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        # Match SIMBAD_final: treat each attention head as an independent
        # batch item and compute scaled dot-product attention explicitly.
        q = torch.cat(torch.split(q, self.head_dim, dim=-1), dim=0)
        k = torch.cat(torch.split(k, self.head_dim, dim=-1), dim=0)
        v = torch.cat(torch.split(v, self.head_dim, dim=-1), dim=0)

        attention = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5
        attention = torch.softmax(attention, dim=-1)
        out = attention @ v
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)

        out = self.fc_out(out)
        out = self.dropout(out)
        return out.permute(0, 1, 3, 2).contiguous()  # B, N, F, T


# ============================================================
#  Spatial module  (Graph conv + spatial attention)
# ============================================================

class AggSpatialConv(nn.Module):
    """Two-hop graph convolution with weighted adjacency."""

    def __init__(self, in_channels, out_channels):
        super(AggSpatialConv, self).__init__()
        self.conv = FC_Linear(in_channels * 3, out_channels)

    def forward(self, x, weighted_adj):
        """Transform (B, N, F, T) features using weighted adjacency."""
        adj_2 = torch.matmul(weighted_adj, weighted_adj)

        x_t = x.permute(0, 3, 1, 2)  # B, T, N, F

        agg1 = torch.matmul(weighted_adj, x_t)  # B, T, N, F
        agg2 = torch.matmul(adj_2,        x_t)  # B, T, N, F

        concat = torch.cat([x_t, agg1, agg2], dim=-1)  # B, T, N, 3F
        concat = concat.permute(0, 3, 2, 1)              # B, 3F, N, T
        out    = self.conv(concat)                        # B, F, N, T

        return out.permute(0, 2, 1, 3)  # B, N, F, T


class Spatial_self_att(nn.Module):
    """Multi-head self-attention along the node axis."""

    def __init__(self, f_in, f_out, dropout, num_heads=4):
        super(Spatial_self_att, self).__init__()
        if f_out % num_heads != 0:
            raise ValueError("f_out must be divisible by num_heads")
        self.head_dim = f_out // num_heads
        self.num_heads = num_heads

        self.fc_q   = nn.Linear(f_in, f_out)
        self.fc_k   = nn.Linear(f_in, f_out)
        self.fc_v   = nn.Linear(f_in, f_out)
        self.fc_out = nn.Linear(f_out, f_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Transform features with shape (B, N, F, T)."""
        x_in = x.permute(0, 3, 1, 2).contiguous()  # B, T, N, F

        q = self.fc_q(x_in)
        k = self.fc_k(x_in)
        v = self.fc_v(x_in)

        B, T, N, C = q.shape
        H, D = self.num_heads, self.head_dim

        q = q.reshape(B * T, N, H, D).transpose(1, 2)
        k = k.reshape(B * T, N, H, D).transpose(1, 2)
        v = v.reshape(B * T, N, H, D).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)  # (B*T, H, N, D)
        out = out.transpose(1, 2).reshape(B, T, N, C)
        out = self.fc_out(out)
        out = self.dropout(out)

        return out.permute(0, 2, 3, 1)  # B, N, F, T  (C==F here)


# ============================================================
#  Sequential Temporal → Spatial Layer
# ============================================================

class SpatioTemporalModelingLayer(nn.Module):

    def __init__(self, in_channels, out_channels, dropout, num_heads):
        super().__init__()
        self.ln_temp_att = nn.LayerNorm(in_channels)
        self.temp_att = Temporal_self_att(
            in_channels,
            out_channels,
            dropout,
            num_heads,
        )
        self.ln_gtu = nn.LayerNorm(in_channels)
        self.gate_conv = GTU(in_channels, out_channels)
        self.ln_spat_att = nn.LayerNorm(out_channels)
        self.spat_att = Spatial_self_att(
            out_channels,
            out_channels,
            dropout,
            num_heads,
        )
        self.ln_graph = nn.LayerNorm(out_channels)
        self.graph_conv = AggSpatialConv(out_channels, out_channels)
        self.ln_ff = nn.LayerNorm(out_channels)
        self.ff = nn.Sequential(
            FC_Linear(out_channels, out_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            FC_Linear(out_channels * 2, out_channels),
        )

    @staticmethod
    def _pre_ln_residual(ln, sublayer, x, **kwargs):
        """Feature-wise Pre-LN, a sublayer, and a residual."""
        x_norm = ln(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return x + sublayer(x_norm, **kwargs)

    def _ff_residual(self, x):
        """Feed-forward block with Pre-LN and a residual."""
        x_norm = self.ln_ff(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        h = x_norm.transpose(1, 2)
        h = self.ff(h)
        h = h.transpose(1, 2)
        return x + h

    def forward(self, x, weighted_adj):
        """Transform (B, N, F, T) features using weighted adjacency."""
        x = self._pre_ln_residual(self.ln_temp_att, self.temp_att, x)
        x = self._pre_ln_residual(self.ln_gtu, self.gate_conv, x)
        x = self._pre_ln_residual(self.ln_spat_att, self.spat_att, x)
        x = self._pre_ln_residual(
            self.ln_graph,
            self.graph_conv,
            x,
            weighted_adj=weighted_adj,
        )
        return self._ff_residual(x)


class SpatioTemporalModelingStack(nn.Module):
    """A sequence of spatio-temporal modeling layers."""

    def __init__(self, in_channels, out_channels, dropout, modules, num_heads):
        super().__init__()

        self.stack = nn.ModuleList(
            [SpatioTemporalModelingLayer(
                in_channels, out_channels, dropout, num_heads
            )
             for _ in range(modules)]
        )

    def forward(self, x, weighted_adj):
        """Transform features with shape (B, N, F, T)."""
        for block in self.stack:
            x = block(x, weighted_adj)
        return x


# ============================================================
#  His_to_Recent
# ============================================================

class His_to_Recent(nn.Module):
    """Fuse selected historical patterns with recent observations."""

    def __init__(self, input_dim, hidden_dim, num_layers, num_heads,
                 num_hours, dropout=0.2):
        super(His_to_Recent, self).__init__()

        self.block = SpatioTemporalModelingStack(
            hidden_dim, hidden_dim, dropout, num_layers, num_heads
        )
        self.block_2 = SpatioTemporalModelingStack(
            hidden_dim, hidden_dim, dropout, num_layers, num_heads
        )

        self.fc_1 = FC_Linear(input_dim, hidden_dim)
        self.fc_2 = FC_Linear(input_dim * num_hours, hidden_dim)

        self.final_fc = FC_Linear(hidden_dim, 1)

        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, weighted_adj, mask):
        """Fuse historical and recent inputs into a (B, N, T) output."""
        # ── historical branch ──────────────────────────────────
        x_1 = self.fc_1(x[0].transpose(1, 2)).transpose(1, 2).float()
        block_output_1 = self.block(x_1, weighted_adj[0])

        # ── fusion with recent ────────────────────────────────
        x_2      = self.fc_2(x[1].transpose(1, 2)).transpose(1, 2).float()
        # mask tells how confident historical is; when mask≈0 rely on recent
        x_fused  = block_output_1 + x_2 + (1 - mask[0]) * x_2
        x_fused  = self.ln(x_fused.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # ── recent branch (temporal patterns → spatial propagation) ─
        block_output_2 = self.block_2(x_fused, weighted_adj[1])

        final_output = self.final_fc(block_output_2.transpose(1, 2)).squeeze(1)  # B,N,T

        return final_output


# ============================================================
#  SIMBAD
# ============================================================

class SIMBAD(nn.Module):
    """Predict future traffic with similarity-aware historical selection."""

    def __init__(self, device, num_for_prediction, adj, threshold, stat,
                 num_of_vertices=307, tau=0.05, dropout=0.1,
                 similarity_window=6, num_weeks=2, num_days=1,
                 num_hours=1, input_dim=1, hidden_dim=64,
                 num_layers=2, num_heads=4):
        super(SIMBAD, self).__init__()
        if not 0 < similarity_window <= num_for_prediction:
            raise ValueError(
                "similarity_window must be between 1 and num_for_prediction"
            )

        if num_weeks < 0 or num_days < 0 or num_weeks + num_days < 1:
            raise ValueError("weeks and days must be non-negative with a positive sum")
        if num_hours < 1:
            raise ValueError("hours must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_candidates = num_weeks + num_days
        self.num_for_prediction = num_for_prediction
        self.input_dim = input_dim
        self.num_weeks = num_weeks
        self.num_days = num_days
        self.num_hours = num_hours

        sim = torch.as_tensor(threshold, device=device)
        self.threshold = nn.Parameter(sim)

        self.submodule = His_to_Recent(
            input_dim, hidden_dim, num_layers, num_heads, num_hours, dropout
        )

        self.W = nn.Parameter(torch.ones(num_of_vertices, num_for_prediction))

        self.unc_adj = UncertaintyAwareAdjacency(self.num_candidates)
        self.improve_mask = UncertaintyProcessor(
            self.num_candidates, hidden_dim, tau, dropout
        )

        self.aperiodicity_aware_modulation = AperiodicityAwareModulationLayer(
            adjacency=adj,
            scale_input=stat['input'],
            scale_sim=stat['sim'],
            similarity_window=similarity_window,
            num_candidates=self.num_candidates,
            num_for_prediction=num_for_prediction,
            input_dim=input_dim,
        )

    def forward(self, x_list):
        if len(x_list) != 4:
            raise ValueError("SIMBAD expects week, day, hour, and similarity inputs")

        x_week = x_list[0].float()
        x_day  = x_list[1].float()
        x_hour = x_list[2].float()

        input_l1 = x_list[3].float()

        expected_lengths = (
            self.num_weeks * self.num_for_prediction,
            self.num_days * self.num_for_prediction,
            self.num_hours * self.num_for_prediction,
        )
        for name, tensor, expected_length in zip(
            ("week", "day", "hour"),
            (x_week, x_day, x_hour),
            expected_lengths,
        ):
            if tensor.ndim != 4 or tensor.shape[2] != self.input_dim:
                raise ValueError(
                    f"{name} input must have shape (batch, nodes, input_dim, time)"
                )
            if tensor.shape[-1] != expected_length:
                raise ValueError(
                    f"{name} input has {tensor.shape[-1]} time steps; "
                    f"expected {expected_length}"
                )
        if input_l1.ndim != 3 or input_l1.shape[-1] != self.num_candidates:
            raise ValueError(
                "similarity input must have shape (batch, nodes, weeks + days)"
            )

        modulation_output = self.aperiodicity_aware_modulation(
            x_week=x_week,
            x_day=x_day,
            x_hour=x_hour,
            input_similarity=input_l1,
            threshold=self.threshold,
            uncertainty_processor=self.improve_mask,
            uncertainty_adjacency=self.unc_adj,
        )

        recent = x_hour.reshape(
            x_hour.shape[0], x_hour.shape[1], self.input_dim,
            self.num_hours, self.num_for_prediction
        )
        recent = recent.permute(0, 1, 3, 2, 4).flatten(start_dim=2, end_dim=3)

        final_norm_adj = [
            modulation_output['historical_adjacency'],
            modulation_output['recent_adjacency'],
        ]
        x_final = [
            modulation_output['historical_input'].float(),
            recent.float(),
        ]
        mask_his = [modulation_output['historical_mask']]

        output = self.submodule(x_final, final_norm_adj, mask_his)  # B, N, T

        final_output = self.W * output

        return final_output, modulation_output['loss_compute']
