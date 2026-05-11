import torch
import torch.nn as nn
import torch.nn.functional as F
from uncertainty import *
from utils import *

# ============================================================
#  Utility / adjacency modules
# ============================================================

class BatchSymNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, adj):
        """
        adj: (B, N, N) weighted adjacency matrices with self-loops
        return: (B, N, N) normalized adjacency matrices
        """

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

class SimilarityAdjacency(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, adj, threshold, scale_input, scale_sim):

        x = x[...,-6:].squeeze(2)
        y = y[...,-6:].squeeze(2)

        x_inv = scale_input.inverse_transform(x)
        y_inv = scale_input.inverse_transform(y)

        l1_adjacency = scale_sim.transform(torch.cdist(x_inv, y_inv, p=1))

        l1_thresh_adj = torch.sigmoid(threshold - l1_adjacency)

        weighted_matrix = l1_thresh_adj * adj.to(x.device)          

        return weighted_matrix

class AdaptiveAdjacency(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(6, 10))
        self.W_2 = nn.Parameter(torch.empty(6, 10))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, pred_sims, uncertainty):
        node_confidence = torch.sigmoid(-uncertainty)
        sim_i = torch.sigmoid(-pred_sims)

        temp = torch.cat([node_confidence, sim_i], dim=-1)  # (B, N, 6)

        W1 = torch.matmul(temp, self.W_1)           # (B, N, 64)
        W2 = torch.matmul(temp, self.W_2).transpose(1, 2)  # (B, 64, N)

        temp_adj = torch.matmul(W1, W2) / 8

        adaptive_adj = F.softmax(F.relu(temp_adj), dim=-1)

        return adaptive_adj

class AdjacencyRowSoftmax(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        mask = adj != 0
        adj_masked = adj.masked_fill(~mask, float('-inf'))
        out = F.softmax(adj_masked, dim=-1)
        out = torch.nan_to_num(out, nan=0.0)
        return out


# ============================================================
#  Linear / conv helpers
# ============================================================

class FC_Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(FC_Linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


# ============================================================
#  Temporal module  (GTU)
# ============================================================

class GTU(nn.Module):
    """Gated Temporal Unit — two stacked gated convolutions."""
    def __init__(self, in_channels, out_channels):
        super(GTU, self).__init__()
        self.temp_conv   = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.gate_conv   = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.temp_conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.gate_conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.in_channels  = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        """x : (B, N, F, T)  →  (B, N, F, T)"""
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
        self.head_dim = f_out // num_heads

        self.fc_q   = nn.Linear(f_in, f_out)
        self.fc_k   = nn.Linear(f_in, f_out)
        self.fc_v   = nn.Linear(f_in, f_out)
        self.fc_out = nn.Linear(f_out, f_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x : (B, N, F, T)  →  (B, N, F, T)"""
        B, N, F, T = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()  # B, N, T, F

        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        q = torch.cat(torch.split(q, self.head_dim, dim=-1), dim=0)
        k = torch.cat(torch.split(k, self.head_dim, dim=-1), dim=0)
        v = torch.cat(torch.split(v, self.head_dim, dim=-1), dim=0)

        attn = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5
        attn = torch.softmax(attn, dim=-1)
        out  = attn @ v
        out  = torch.cat(torch.split(out, B, dim=0), dim=-1)  # B, N, T, F

        out = self.fc_out(out)
        out = self.dropout(out)
        return out.permute(0, 1, 3, 2).contiguous()  # B, N, F, T


# ============================================================
#  Spatial module  (Graph conv + spatial attention)
# ============================================================

class AggSpatialConv(nn.Module):
    """2-hop graph convolution with a weighted adjacency matrix."""
    def __init__(self, device, in_channels, out_channels):
        super(AggSpatialConv, self).__init__()
        self.conv = FC_Linear(in_channels * 3, out_channels)

    def forward(self, x, weighted_adj):
        """
        x           : (B, N, F, T)
        weighted_adj: (B, 1, N, N)  or  (B, N, N)
        → (B, N, F, T)
        """
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
        self.head_dim = f_out // num_heads
        self.num_heads = num_heads

        self.fc_q   = nn.Linear(f_in, f_out)
        self.fc_k   = nn.Linear(f_in, f_out)
        self.fc_v   = nn.Linear(f_in, f_out)
        self.fc_out = nn.Linear(f_out, f_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x : (B, N, F, T)  →  (B, N, F, T)"""
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
#  Temporal → Spatial Layer
# ============================================================

class STM_Layer(nn.Module):
    """
    Sequential Temporal → Spatial layer.

    Flow (per sub-layer uses Pre-LN for stability):
    ┌────────────────────────────────────────────────────┐
    │  x_in  (B, N, F, T)                                │
    │    ↓  LayerNorm                                    │
    │  Temporal self-attention                           │
    │    ↓  + x_in  (residual)                           │
    │  x_t                                               │
    │    ↓  LayerNorm                                    │
    │  GTU  (gated temporal conv)                        │
    │    ↓  + x_t  (residual)                            │
    │  x_temporal   ← full temporal representation       │
    │    ↓  LayerNorm                                    │
    │  Spatial self-attention                            │
    │    ↓  + x_temporal  (residual)                     │
    │  x_s                                               │
    │    ↓  LayerNorm                                    │
    │  Graph convolution  (weighted adj)                 │
    │    ↓  + x_s  (residual)                            │
    │  x_out   (B, N, F, T)                              │
    └────────────────────────────────────────────────────┘

    """

    def __init__(self, device, hidden_dim, dropout):
        super(STM_Layer, self).__init__()

        # ── Temporal sub-layers ──────────────────────────────────
        self.ln_temp_att  = nn.LayerNorm(hidden_dim)
        self.temp_att     = Temporal_self_att(hidden_dim, hidden_dim, dropout)

        self.ln_gtu       = nn.LayerNorm(hidden_dim)
        self.gate_conv    = GTU(hidden_dim, hidden_dim)

        # ── Spatial sub-layers ──────────────────────────────────
        self.ln_spat_att  = nn.LayerNorm(hidden_dim)
        self.spat_att     = Spatial_self_att(hidden_dim, hidden_dim, dropout)

        self.ln_graph     = nn.LayerNorm(hidden_dim)
        self.graph_conv   = AggSpatialConv(device, hidden_dim, hidden_dim)

        # ── Feed-forward mixing (optional, keeps expressiveness) ─
        self.ln_ff  = nn.LayerNorm(hidden_dim)
        self.ff     = nn.Sequential(
            FC_Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            FC_Linear(hidden_dim * 2, hidden_dim),
        )

    # ----------------------------------------------------------
    # helpers: apply Pre-LN → sublayer → residual
    # ----------------------------------------------------------
    @staticmethod
    def _pre_ln_residual(ln, sublayer, x, **kwargs):
        """Pre-LayerNorm residual wrapper.  LN is applied on the F dim."""
        # x: (B, N, F, T)
        # LayerNorm over F (last non-time dim) → reshape trick
        x_norm = ln(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # B,N,F,T
        return x + sublayer(x_norm, **kwargs)

    @staticmethod
    def _pre_ln_residual_plain(ln, sublayer, x):
        x_norm = ln(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return x + sublayer(x_norm)

    def _ff_residual(self, x):
        """Feed-forward over F dimension with Pre-LN."""
        x_norm = self.ln_ff(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        # apply FF independently at every (B,N,T) position
        h = x_norm.transpose(1,2)   # B, F, N, T
        h = self.ff(h)
        h = h.transpose(1,2)        # B, N, F, T
        return x + h

    # ----------------------------------------------------------
    def forward(self, x, weighted_adj):
        """
        x            : (B, N, F, T)
        weighted_adj : (B, 1, N, N)
        → (B, N, F, T)
        """
        # ── 1. Temporal self-attention ──────────────────────────
        x = self._pre_ln_residual_plain(self.ln_temp_att, self.temp_att, x)

        # ── 2. Gated temporal convolution ───────────────────────
        x = self._pre_ln_residual_plain(self.ln_gtu, self.gate_conv, x)

        # x is now the "temporal representation" fed into spatial
        # ── 3. Spatial self-attention ───────────────────────────
        x = self._pre_ln_residual_plain(self.ln_spat_att, self.spat_att, x)

        # ── 4. Graph convolution with adaptive adj ───────────────
        x = self._pre_ln_residual(self.ln_graph, self.graph_conv, x, weighted_adj=weighted_adj)

        # ── 5. Feed-forward (token-mixing along F) ───────────────
        x = self._ff_residual(x)

        return x  # B, N, F, T


class STM_Stack(nn.Module):
    def __init__(self, device, hidden_dim, dropout, num_modules):
        super(STM_Stack, self).__init__()

        self.stack = nn.ModuleList(
            [STM_Layer(device, hidden_dim, dropout)
             for _ in range(num_modules)]
        )

    def forward(self, x, weighted_adj):
        """x : (B, N, F, T)  →  (B, N, F, T)"""
        for block in self.stack:
            x = block(x, weighted_adj)
        return x


# ============================================================
#  His_to_Recent
# ============================================================

class His_to_Recent(nn.Module):
    def __init__(self, device, hidden_dim, num_for_prediction, dropout=0.2, num_modules=2):
        super(His_to_Recent, self).__init__()

        self.block   = STM_Stack(device, hidden_dim, dropout, num_modules)
        self.block_2 = STM_Stack(device, hidden_dim, dropout, num_modules)

        self.fc_1 = FC_Linear(1,64)
        self.fc_2 = FC_Linear(1,64)

        self.final_fc = FC_Linear(64,1)

        self.ln = nn.LayerNorm(64)

    def forward(self, x, weighted_adj, mask):
        """
        x[0] : historical input (B, N, 1, T)
        x[1] : recent input    (B, N, 1, T)
        weighted_adj[0] : his adj (B,1,N,N)
        weighted_adj[1] : hour adj (B,1,N,N)
        mask : (B, N, 1, 1)
        """
        # ── historical branch ──────────────────────────────────
        x_1 = self.fc_1(x[0].transpose(1, 2)).transpose(1, 2).float()  # B,N,64,T
        block_output_1 = self.block(x_1, weighted_adj[0])               # B,N,64,T

        # ── fusion with recent ────────────────────────────────
        x_2      = self.fc_2(x[1].transpose(1, 2)).transpose(1, 2).float()
        # mask tells how confident historical is; when mask≈0 rely on recent
        x_fused  = block_output_1 + x_2 + (1 - mask) * x_2
        x_fused  = self.ln(x_fused.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # ── recent branch (temporal patterns → spatial propagation) ─
        block_output_2 = self.block_2(x_fused, weighted_adj[1])  # B,N,64,T

        final_output = self.final_fc(block_output_2.transpose(1, 2)).squeeze(1)  # B,N,T

        return final_output

# ============================================================
#  SIMBAD
# ============================================================

class SIMBAD(nn.Module):
    def __init__(self, device, num_for_prediction, adj, threshold, stat,
                 num_of_vertices, hidden_dim, output_dim, tau, dropout):
        super(SIMBAD, self).__init__()

        sim = torch.from_numpy(threshold).to(device)
        self.threshold = nn.Parameter(sim)

        self.submodule = His_to_Recent(device, hidden_dim, num_for_prediction, dropout)

        self.W = nn.Parameter(torch.ones(num_of_vertices, num_for_prediction))

        self.node     = num_of_vertices
        self.output_dim = output_dim
        self.timestep = num_for_prediction
        self.num_nodes = num_of_vertices

        self.adap_adj      = AdaptiveAdjacency()
        self.improve_mask = UncertaintyRegulation(hidden_dim, tau, num_of_vertices, sim, dropout)

        self.adj_      = adj
        self.softmax   = AdjacencyRowSoftmax()
        self.adj_norm = BatchSymNorm()

        self.weight_adj = SimilarityAdjacency()
        self.tau        = tau

        self.scale_input = stat['input']
        self.scale_sim   = stat['sim']

        self.device = device

    def forward(self, x_list):
        x_week = x_list[0].float()
        x_day  = x_list[1].float()
        x_hour = x_list[2].float()

        input_l1 = x_list[3].float()

        processing_output = self.improve_mask(input_l1)

        probs         = processing_output['probs']
        pred_sims     = processing_output['pred_sims']
        uncertainty   = processing_output['uncertainty']
        unc_adj       = processing_output['uncertainty_adj']
        loss_compute  = processing_output['loss_compute']

        p_w2, p_w1, p_d1 = probs

        x_his_input = (
            p_w2.unsqueeze(-1) * x_week[..., :12].contiguous() +
            p_w1.unsqueeze(-1) * x_week[..., 12:].contiguous() +
            p_d1.unsqueeze(-1) * x_day.contiguous()
        ).float()

        expected_pred = (
            p_w2 * pred_sims[0] +
            p_w1 * pred_sims[1] +
            p_d1 * pred_sims[2]
        ).float()

        expected_unc = (
            p_w2 * unc_adj[0] +
            p_w1 * unc_adj[1] +
            p_d1 * unc_adj[2]
        ).float()

        mask = torch.sigmoid(self.threshold - expected_pred)
        mask = (mask * expected_unc).unsqueeze(-1).float()

        x_final_his_input = x_his_input * mask

        his_adj  = self.adap_adj(
            torch.cat(pred_sims, dim=-1),
            torch.cat(uncertainty, dim=-1)
        ).float()

        hour_adj = self.softmax(
            self.weight_adj(x_hour, x_hour, self.adj_, self.threshold,
                            self.scale_input, self.scale_sim)
        ).float()

        normalized_hour_adj = self.adj_norm(hour_adj).float().unsqueeze(1)

        final_norm_adj = [his_adj.unsqueeze(1), normalized_hour_adj]
        x_final        = [x_final_his_input.float(), x_hour.float()]

        output = self.submodule(x_final, final_norm_adj, mask)  # B, N, T

        final_output = self.W * output

        return final_output, loss_compute
