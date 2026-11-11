import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_uncertainty_loss(true_target, pred_target, uncertainty):
    return (
        0.5 * torch.exp(-uncertainty) * (pred_target - true_target) ** 2
        + 0.5 * uncertainty
    ).mean()


class UncertaintyAwareAdjacency(nn.Module):
    """Construct adaptive adjacency from similarity and uncertainty."""

    def __init__(self, num_candidates):
        super().__init__()
        projection_dim = 10
        input_dim = num_candidates * 2
        self.W_1 = nn.Parameter(torch.empty(input_dim, projection_dim))
        self.W_2 = nn.Parameter(torch.empty(input_dim, projection_dim))
        self.scale = 8.0
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)

    def forward(self, pred_sims, uncertainty):
        node_confidence = torch.sigmoid(-uncertainty)
        similarity_confidence = torch.sigmoid(-pred_sims)
        features = torch.cat(
            [node_confidence, similarity_confidence], dim=-1
        )

        left = torch.matmul(features, self.W_1)
        right = torch.matmul(features, self.W_2).transpose(1, 2)
        adjacency_logits = torch.matmul(left, right) / self.scale

        return F.softmax(F.relu(adjacency_logits), dim=-1)


class SimilarityPredictor(nn.Module):
    """Predict target similarities and their uncertainty."""

    def __init__(self, num_candidates, hidden_dim, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_candidates, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_candidates),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_candidates),
        )

    def forward(self, similarity_features):
        encoded = self.encoder(similarity_features)
        predicted_similarity = self.predictor(encoded)
        uncertainty = self.uncertainty_head(encoded)
        return predicted_similarity, uncertainty


class UncertaintyProcessor(nn.Module):
    """Convert similarity predictions and uncertainty into candidate weights."""

    def __init__(self, num_candidates, hidden_dim, tau, dropout):
        super().__init__()
        self.tau = tau
        self.sim_predictor = SimilarityPredictor(
            num_candidates, hidden_dim, dropout
        )

    def forward(self, recent_similarities):
        predicted_similarities, uncertainty = self.sim_predictor(
            recent_similarities
        )
        confidence = torch.sigmoid(-uncertainty)
        logits = -predicted_similarities * confidence
        probabilities = F.softmax(logits / self.tau, dim=-1)

        return {
            'probs': probabilities,
            'pred_sims': predicted_similarities,
            'uncertainty': uncertainty,
            'uncertainty_adj': confidence,
            'loss_compute': (predicted_similarities, uncertainty),
        }
