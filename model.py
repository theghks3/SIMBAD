import numpy as np
import os
import csv
import pickle
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import BatchSampler
from torch_geometric.nn import MessagePassing
from utils import *

class Adjacency_Weight(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, adj):
        x = x.squeeze(2)
        y = y.squeeze(2)

        x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-10 # The L2 norm of x
        y_norm = torch.norm(y, dim=-1, keepdim=True) + 1e-10 # The L2 norm of y
        
        # cosine similarity calculation (x dot y / (x_norm * y_norm))
        similarity = torch.bmm(x, y.transpose(1, 2)) / (x_norm * y_norm.transpose(1, 2))

        eye = torch.eye(similarity.size(1), device=similarity.device).unsqueeze(0)
        similarity_matrix = similarity * (1 - eye)

        weighted_matrix = similarity_matrix * adj

        return weighted_matrix

class AdjacencyRowSoftmax(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        mask = adj != 0
        exps = torch.exp(adj) * mask
        sums = exps.sum(dim=-1, keepdim=True)
        out = exps / torch.clamp(sums, min=self.eps)
        return out

class NodewiseLearnableAdjWeight(nn.Module):
    def __init__(self, device, adj):
        '''
        adj : adjacency matrix with self-loop // shape : (node, node)
        '''
        super().__init__()
        self.adj = adj.to(device)
        self.theta = nn.Parameter(torch.full((adj.size(0), 1), 0.5))
        self.device = device

    def forward(self):
        '''
        :return:
        weights : balances the strength of self-loop and its neighbors // theta for self-loop, (1-theta)/n for neighbor
        '''
        N = self.adj.size(0)
        eye = torch.eye(N, device=self.adj.device)
        is_self_loop = eye.bool()

        # number of neighbors
        neighbor_count = (self.adj - eye).sum(dim=1, keepdim=True) + 1e-10  # shape: [N, 1]

        # weight matrix initialization
        weights = torch.zeros_like(self.adj, dtype=torch.float).to(self.device)

        # theta for each node
        weights[is_self_loop] = self.theta.squeeze()

        # neighbors : (1 - theta) / n
        neighbor_mask = self.adj.bool() & (~is_self_loop)
        neighbor_weights = (self.theta / neighbor_count).repeat(1, N)  # shape: [N, N]
        weights[neighbor_mask] = neighbor_weights[neighbor_mask].float()
        return weights  # shape: [N, N]

# Self attention on spatial axis
class Spatial_self_att(nn.Module):
    def __init__(self, f_in, f_out, num_heads=4):
        '''
        f_in : input dimension size for query, key, value
        f_out : output dimension size for qeury, key, value
        num_heads : number of heads for self-attention
        '''
        super(Spatial_self_att, self).__init__()
        self.f_in = f_in
        self.f_out = f_out

        self.fc_q = nn.Linear(f_in, f_out)
        self.fc_k = nn.Linear(f_in, f_out)
        self.fc_v = nn.Linear(f_in, f_out)

        self.head_dim = f_out // num_heads

        self.fc_out = nn.Linear(f_out, f_out)

        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        '''
        :param:
        x : input // shape : (batch_size, node, f_in, timestep)

        :return:
        out : output // shape : (batch_size, node, f_out, timestep)
        '''
        x = x.permute(0,3,1,2).contiguous() # B, T, N, F

        batch_size = x.shape[0]

        query = self.fc_q(x)
        key = self.fc_k(x)
        value = self.fc_v(x)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1,-2)

        attn_score = (query @ key) / self.head_dim**0.5

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        
        out = self.fc_out(out)
        out = self.dropout(out)

        out = out.permute(0,2,3,1) # B, N, F, T

        return out

class Temporal_self_att(nn.Module):
    def __init__(self, f_in, f_out, num_heads=4):
        super(Temporal_self_att, self).__init__()
        self.f_in = f_in
        self.f_out = f_out

        self.fc_q = nn.Linear(f_in, f_out)
        self.fc_k = nn.Linear(f_in, f_out)
        self.fc_v = nn.Linear(f_in, f_out)

        self.head_dim = f_out // num_heads

        self.fc_out = nn.Linear(f_out, f_out)

        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.float()
        B, N, F, T = x.shape
        x = x.permute(0,1,3,2).contiguous() # B, N, T, F

        batch_size = x.shape[0]

        query = self.fc_q(x)
        key = self.fc_k(x)
        value = self.fc_v(x)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1,-2)

        attn_score = (query @ key) / self.head_dim**0.5

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)

        out = self.fc_out(out)
        
        out = self.dropout(out)

        out = out.permute(0,1,3,2).contiguous() # B, N, F, T

        return out

# Spatial convolution with weighted adjacency matrix
class AggSpatialConv(nn.Module):
    def __init__(self, device, in_channels, out_channels, adj_mx):
        '''
        in_channels : input feature
        out_channels : output feature
        adj_mx : for theta operation // shape : (node, node) // with self-loop
        '''
        super(AggSpatialConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight_1 = nn.Parameter(torch.randn(in_channels, out_channels))
        self.weight_2 = nn.Parameter(torch.randn(in_channels, out_channels))
        self.adj_param_1 = NodewiseLearnableAdjWeight(device, adj_mx)
        self.adj_param_2 = NodewiseLearnableAdjWeight(device, adj_mx)

    def forward(self, x, weighted_adj):
        '''
        :param:
        x: input // shape : (batch_size, node, feature, timestep)
        weighted_adj: Weighted adjacency matrix of shape (batch_size, 1, node, node)

        :return:
        final_output : output // shape : (batch_size, node, feature, timestep)
        '''
        # theta, 1-theta
        norm_A_1 = self.adj_param_1().float()
        norm_A_2 = self.adj_param_2().float()

        adj_1_hop = weighted_adj * norm_A_1
        adj_2_hop = weighted_adj * norm_A_2
        adj_2_hop = torch.matmul(adj_2_hop, adj_2_hop)

        x = x.permute(0,3,1,2) # B, T, N, F

        agg_features_1 = torch.matmul(adj_1_hop, x)
        agg_features_2 = torch.matmul(adj_2_hop, x)

        transformed_1 = torch.matmul(agg_features_1, self.weight_1)
        transformed_2 = torch.matmul(agg_features_2, self.weight_2)

        final_output = torch.sigmoid(transformed_1 + transformed_2) # B, T, N, F

        final_output = final_output.permute(0,2,3,1) # B, N, F, T

        return final_output

# Gated convolution along time axis
class GTU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GTU, self).__init__()
        self.temp_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=(1,1), padding=(0,1))

        self.temp_conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.gate_conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=(1,1), padding=(0,1))

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        '''
        :param:
        x : input // shape : (batch_size, node, feature, timestep)

        :return:
        gated_conv_output : output // shape : (batch_size, node, feature, timestep)
        '''
        x_conv = x.permute(0,2,1,3)

        temp = self.temp_conv(x_conv)
        gate = self.gate_conv(x_conv)

        gated_conv_output = torch.tanh(temp) * torch.sigmoid(gate) # B, F, N, T

        temp_2 = self.temp_conv_2(gated_conv_output)
        gate_2 = self.gate_conv_2(gated_conv_output)

        final_output = torch.tanh(temp_2) * torch.sigmoid(gate_2)
        final_output = final_output.permute(0,2,1,3)

        return final_output

class STM_Layer(nn.Module):
    def __init__(self, device, in_channels, out_channels, adj_mx, edge_index):
        super(STM_Layer, self).__init__()
        
        self.temp_att = Temporal_self_att(in_channels, out_channels)
        self.spat_att = Spatial_self_att(in_channels, out_channels)

        self.mean_agg_1 = AggSpatialConv(device, in_channels, out_channels, adj_mx)

        self.gate_conv = GTU(in_channels, out_channels)

        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x, weighted_adj):
        '''
        :param:
        x : input // shape : (batch, node, feature, timestep)
        weighted_adj : weighted adjacency matrix // shape : (batch_size, 1, node, node)

        :return:
        final_output : output // shape : (batch_size, node, feature, timestep)
        '''
        x_input = x

        spatial_att = self.spat_att(x)
        temporal_att = self.temp_att(x)

        # spatial convolution
        mean_aggregation = self.mean_agg_1(spatial_att, weighted_adj)

        # temporal convolution
        gated = self.gate_conv(temporal_att) # B, N, F, T
        
        output = mean_aggregation + gated # B, N, F, T

        x_residual = self.ln(F.relu((output + x_input).permute(0,3,1,2))) # B, T, N, F

        final_output = x_residual.permute(0,2,3,1) # B, N, F, T

        return final_output

class STM_Stack(nn.Module):
    def __init__(self, device, in_channels, out_channels, adj_mx, edge_index, modules):
        super(STM_Stack, self).__init__()
        
        self.stack = nn.ModuleList(
            [   
                STM_Layer(device, in_channels, out_channels, adj_mx, edge_index)
                for _ in range(modules)
            ]
        )
        
    def forward(self, x, weighted_adj):
        '''
        :param:
        x : input // shape : (batch, node, feature, timestep)
        weighted_adj : weighted adjacency matrix // shape : (batch_size, 1, node, node)
        
        :return:
        x : output // shape : (batch_size, node, feature, timestep)
        '''
        x_input = x
        
        for block in self.stack:
            x = block(x, weighted_adj)
        
        return x

class His_to_Recent(nn.Module):
    def __init__(self, device, num_for_prediction, backbone, sem_adj, modules=2):
        super(His_to_Recent, self).__init__()

        self.block = STM_Stack(device, 64, 64, sem_adj, backbone['edge_index'], modules)
        self.block_2 = STM_Stack(device, 64, 64, backbone['adj_param'], backbone['edge_index'], modules)
  
        self.fc_1 = nn.Linear(2,64)
        self.fc_2 = nn.Linear(1,64)

        self.final_fc = nn.Linear(64, 1)

        self.ln = nn.LayerNorm(64)
    
    def forward(self, x, weighted_adj, mask, embed):
        '''
        :param:
        x : input // shape : (batch, node, feature, timestep)
        weighted_adj : weighted adjacency matrix // shape : (batch_size, 1, node, node)
        mask : masking matrices of historical inputs // shape : (batch_size, node, 1, 1)
        embed : embedding vector // shape : (batch_size, node, 1)

        :return:
        final_output : output // shape : (batch_size, node, timestep)
        '''
        x_1 = self.fc_1(x[0].transpose(2,3)).transpose(2,3) + embed.unsqueeze(-1).repeat(1,1,1,12)
        block_output_1 = self.block(x_1, weighted_adj[0]) # B, N, F, T

        x_2 = self.fc_2(x[1].transpose(2,3)).transpose(2,3)
        x_block_2 = block_output_1 + x_2 + (1 - mask[0] * mask[1]) * x_2
        x_block_2_in = self.ln(x_block_2.permute(0,3,1,2)).permute(0,2,3,1)
        block_output_2 = self.block_2(x_block_2_in, weighted_adj[1])

        final_output = self.final_fc(block_output_2.transpose(2,3)).squeeze(-1) # B, N, T

        return final_output

class SIMBAD(nn.Module):
    def __init__(self, device, num_for_prediction, backbone, adj, threshold_week, threshold_day, sem_adj, num_of_vertices=307, in_dim=64, output_dim=1, scale=20, tau=0.05):
        super(SIMBAD, self).__init__()

        week_sim = torch.from_numpy(threshold_week).to(device)
        day_sim = torch.from_numpy(threshold_day).to(device)
        
        self.threshold_week = nn.Parameter(week_sim)
        self.threshold_day = nn.Parameter(day_sim)

        self.submodule = His_to_Recent(device, num_for_prediction, backbone[0], sem_adj)

        self.W = nn.Parameter(torch.ones(num_of_vertices, num_for_prediction))
        self.embed = nn.Embedding(4,64)

        self.node = num_of_vertices
        self.in_dim = in_dim
        self.output_dim = output_dim
        self.timestep = num_for_prediction
        self.num_nodes = num_of_vertices

        self.softmax = AdjacencyRowSoftmax()

        self.adj_ = adj
        self.sem_adj = sem_adj

        self.weight_adj = Adjacency_Weight()
        self.scale = scale
        self.tau = tau

        self.device = device

    def forward(self, x_list):
        '''
        x_list[0]: weekly input // shape : (batch_size, node, feature, timestep)
        x_list[1]: daily input // shape : (batch_size, node, feature, timestep)
        x_list[2]: recent input // shape : (batch_size, node, feature, timestep)
        x_list[3]: last 2 week dtw // shape : (batch_size, node, 1)
        x_list[4]: last week dtw // shape : (batch_size, node, 1)
        x_list[5]: last day dtw // shape : (batch_size, node, 1)
        '''
        
        x_week = x_list[0].float()
        x_day = x_list[1].float()
        x_hour = x_list[2].float()

        week2_l1 = x_list[3]
        week_l1 = x_list[4]
        day_l1 = x_list[5]

        # 1) L1-based Proabability Distribution
        logits = torch.cat([-week_l1, -week2_l1], dim=-1)  # (B, N, 2) 
        prob = torch.softmax(logits / self.tau, dim=-1)    # (B, N, 2)

        p_w1 = prob[..., 0].unsqueeze(-1).float()  # (B, N, 1)
        p_w2 = prob[..., 1].unsqueeze(-1).float()  # (B, N, 1)

        # 2) Expectation of Weekly Input
        x_week_input = (
            p_w2.unsqueeze(-1) * x_week[..., :12].contiguous() +
            p_w1.unsqueeze(-1) * x_week[..., 12:].contiguous()
        ).float() 

        # 3) Expectation of L1-similarity of Weekly Input
        mask_week_val = p_w2 * week2_dtw_value + p_w1 * week_dtw_value   # (B, N, 1)

        # 4-1-1) Masking Matrix of Weekly Input
        mask_week = torch.sigmoid(self.scale * (self.threshold_week - mask_week_val)).float()
        mask_week_idx = (mask_week >= 0.5).long().squeeze(-1)
        mask_week = mask_week.unsqueeze(-1)  # (B, N, 1, 1)

        # 4-1-2) Mask Weekly Input
        x_final_week_input = x_week_input * mask_week

        # 4-2-1) Masking Matrix of Daily Input
        mask_day = torch.sigmoid(self.scale * (self.threshold_day - day_dtw_value)).float()
        mask_day_idx = (mask_day >= 0.5).long().squeeze(-1)
        mask_day = mask_day.unsqueeze(-1) # B, N, 1, 1

        # 4-2-2) Mask Daily Input
        x_final_day_input = x_day * mask_day

        # 5) Weighted Adjacency Matrix of Historical and Recent Inputs
        his = torch.cat([x_week_input, x_day], dim=-1)
        his_hour = torch.cat([x_hour, x_hour], dim=-1)
        
        his_sem_adj = self.sem_adj - torch.eye(self.num_nodes, device=self.sem_adj.device)
        his_adj = self.softmax(self.weight_adj(his, his_hour, his_sem_adj)).float()
        hour_adj = self.softmax(self.weight_adj(x_hour, x_hour, self.adj_)).float()

        # Embedding
        mask_final_idx = (mask_week_idx << 1) | mask_day_idx
        embedding_vec = self.embed(mask_final_idx)

        # Concat Historical Inputs
        x_final_weekday = torch.cat([x_final_week_input, x_final_day_input], dim=2)

        # Weighted Normalization
        normalized_his_adj = batch_symmetric_normalize(his_adj).float().unsqueeze(1) # (B, 1, N, N)
        normalized_hour_adj = batch_symmetric_normalize(hour_adj).float().unsqueeze(1) # (B, 1, N, N)

        final_norm_adj = [final_his_adj, final_hour_adj]

        x_final = [x_final_weekday, x_hour]
        mask_his = [mask_week, mask_day]

        output = self.submodule(x_final, final_norm_adj, mask_his, embedding_vec) # B, N, F, T

        final_output = self.W * output

        return final_output
