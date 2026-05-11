import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
#  Uncertainty
# ============================================================

class SimilarityPredictor(nn.Module):
    def __init__(self, dropout, hidden_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, sim_features):
        h = self.encoder(sim_features)
        pred_target_sim = self.predictor(h)
        uncertainty     = self.uncertainty_head(h)

        return pred_target_sim, uncertainty

# ============================================================
#  Similarity Distribution
# ============================================================

class UncertaintyRegulation(nn.Module):
    def __init__(self, hidden_dim, tau, num_vertices, sim, dropout):
        super().__init__()
        self.tau   = tau

        self.sim_predictor = SimilarityPredictor(dropout, hidden_dim)

    def forward(self, recent_sims):
        pred_target_sims, uncertainty = self.sim_predictor(recent_sims)

        pred_week2_sim = pred_target_sims[..., 0:1]
        pred_week_sim  = pred_target_sims[..., 1:2]
        pred_day_sim   = pred_target_sims[..., 2:3]

        uncertainty_adj = torch.sigmoid(-uncertainty)

        logits = -pred_target_sims * uncertainty_adj

        logits_scaled = logits / self.tau
        probs = F.softmax(logits_scaled, dim=-1)

        p_w2 = probs[..., 0:1]
        p_w1 = probs[..., 1:2]
        p_d1 = probs[..., 2:3]

        unc_w2, unc_w, unc_d = uncertainty[..., 0:1], uncertainty[..., 1:2], uncertainty[..., 2:3]

        return {
            'probs':         (p_w2, p_w1, p_d1),
            'pred_sims':     (pred_week2_sim, pred_week_sim, pred_day_sim),
            'uncertainty':   (unc_w2, unc_w, unc_d),
            'uncertainty_adj': (uncertainty_adj[...,0:1], uncertainty_adj[...,1:2], uncertainty_adj[...,2:3]),
            'loss_compute':  (pred_target_sims, uncertainty)
        }
