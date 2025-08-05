import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from utils import *

class NodewiseLearnableAdjWeight(nn.Module):
    def __init__(self, device, adj):
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

class WeightedAdjacency(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(2))

    def forward(self, adj_list):
        '''
        :param:
        adj_list : list of adjacency matrices // shape : (batch_size, node, node)

        :return:
        A : balanced adjacency matrix // shape : (batch_size, node, node)
        '''
        # sum of each parameter to 1
        weights = F.softmax(self.weights, dim=0)
        
        theta_1, theta_2 = weights[0], weights[1]

        A = theta_1 * adj_list[0] + theta_2 * adj_list[1]

        return A

# Self attention on spatial axis
class Spatial_self_att(nn.Module):
    '''
    Self attention on spatial axis
    https://github.com/XDZhelheim/STAEformer/blob/main/model/STAEformer.py
    '''
    def __init__(self, f_in, f_out, dropout, num_heads=4):
        super(Spatial_self_att, self).__init__()
        self.f_in = f_in
        self.f_out = f_out

        self.fc_q = nn.Linear(f_in, f_out)
        self.fc_k = nn.Linear(f_in, f_out)
        self.fc_v = nn.Linear(f_in, f_out)

        self.head_dim = f_out // num_heads

        self.fc_out = nn.Linear(f_out, f_out)

        self.dropout = nn.Dropout(dropout)
    
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
        
        out = self.dropout(out)
        out = self.fc_out(out)

        out = out.permute(0,2,3,1) # B, N, F, T

        return out

class Temporal_self_att(nn.Module):
    '''
    Self attention on temporal axis
    https://github.com/XDZhelheim/STAEformer/blob/main/model/STAEformer.py
    '''
    def __init__(self, f_in, f_out, dropout, num_heads=4):
        super(Temporal_self_att, self).__init__()
        self.f_in = f_in
        self.f_out = f_out

        self.fc_q = nn.Linear(f_in, f_out)
        self.fc_k = nn.Linear(f_in, f_out)
        self.fc_v = nn.Linear(f_in, f_out)

        self.head_dim = f_out // num_heads

        self.fc_out = nn.Linear(f_out, f_out)

        self.dropout = nn.Dropout(dropout)
    
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

        out = self.dropout(out)
        
        out = self.fc_out(out)

        out = out.permute(0,1,3,2).contiguous() # B, N, F, T

        return out

# Spatial convolution with weighted adjacency matrix
class AggSpatialConv(nn.Module):
    def __init__(self, device, in_channels, out_channels, adj_mx):
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
        adj_2_hop = torch.matmul(weighted_adj, weighted_adj)
        adj_2_hop = adj_2_hop * norm_A_2

        x = x.permute(0,3,1,2) # B, T, N, F

        agg_features_1 = torch.matmul(adj_1_hop, x)
        agg_features_2 = torch.matmul(adj_2_hop, x)

        transformed_1 = torch.matmul(agg_features_1, self.weight_1)
        transformed_2 = torch.matmul(agg_features_2, self.weight_2)

        final_output = torch.sigmoid(transformed_1 + transformed_2) # B, T, N, F

        final_output = final_output.permute(0,2,3,1) # B, N, F, T

        return final_output

# Max-pooling operation
class MaxPoolAggregator(MessagePassing, nn.Module):
    def __init__(self, device, in_channels, out_channels, edge_index):
        super().__init__(aggr='max')
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.edge_index = edge_index.to(device)

    def forward(self, x):
        '''
        :param:
        x : input // shape : (batch_size, node, feature, timestep)

        :return:
        final_output : output // shape : (batch_size, node, feature, timestep)
        '''
        # x: [N, in_channels], edge_index: [2, E]
        x = x.float()
        batch, node, feature, timestep = x.shape
        
        x = x.permute(0,3,1,2) # B, T, N, F
        x = self.linear(x)

        x = x.reshape(-1, node, feature) # b*t, n, f
        final_output = self.propagate(self.edge_index, x=x)
        final_output = final_output.reshape(batch, timestep, node, feature).permute(0,2,3,1)

        return final_output

# Gated convolution along time axis
class Gated_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gated_Conv, self).__init__()
        self.temp_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=(1,1), padding=(0,1))

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

        gated_conv_output = temp * torch.sigmoid(gate) # B, F, N, T
        gated_conv_output = gated_conv_output.permute(0,2,1,3) # B, N, F, T

        return gated_conv_output

class SIMBAD_block(nn.Module):
    def __init__(self, device, in_channels, out_channels, adj_mx, edge_index, dropout):
        super(SIMBAD_block, self).__init__()
        
        self.temp_att = Temporal_self_att(in_channels, out_channels, dropout)
        self.spat_att = Spatial_self_att(in_channels, out_channels, dropout)

        self.mean_agg_1 = AggSpatialConv(device, in_channels, out_channels, adj_mx)
        self.mean_agg_2 = AggSpatialConv(device, in_channels, out_channels, adj_mx)

        self.max_pool_1 = MaxPoolAggregator(device, in_channels, out_channels, edge_index)

        self.gate_conv = Gated_Conv(in_channels, out_channels)

        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x, weighted_adj, mask_week, mask_day, mask):
        '''
        :param:
        x : input // shape : (batch, node, feature, timestep)
        weighted_adj : weighted adjacency matrix // shape : (batch_size, 1, node, node)
        mask_week : mask value by weekly input // shape : (batch_size, node, 1, 1)
        mask_day : mask value by daily input // shape : (batch_size, node, 1, 1)
        mask : whether or not to mask spatial convolution result // bool

        :return:
        final_output : output // shape : (batch_size, node, feature, timestep)
        '''
        x_input = x

        spatial_att = self.spat_att(x)
        temporal_att = self.temp_att(x)

        # spatial convolution
        mean_aggregation = self.mean_agg_1(spatial_att, weighted_adj)
        max_pooled = self.max_pool_1(mean_aggregation)
        mean_2 = self.mean_agg_2(max_pooled, weighted_adj)

        if mask:
            last_output = mean_2 * mask_week * mask_day
        else:
            last_output = mean_2

        # temporal convolution
        gated = self.gate_conv(temporal_att) # B, N, F, T

        output = last_output + gated # B, N, F, T

        x_residual = self.ln(F.relu((output + x_input).permute(0,3,1,2))) # B, T, N, F

        final_output = x_residual.permute(0,2,3,1) # B, N, F, T

        return final_output

class SIMBAD_module(nn.Module):
    def __init__(self, device, in_channels, out_channels, adj_mx, edge_index, dropout, modules):
        super(SIMBAD_module, self).__init__()
        
        self.stack = nn.ModuleList(
            [   
                SIMBAD_block(device, in_channels, out_channels, adj_mx, edge_index, dropout)
                for _ in range(modules)
            ]
        )
        
    def forward(self, x, weighted_adj, mask_week, mask_day, mask):
        '''
        :param:
        x : input // shape : (batch, node, feature, timestep)
        weighted_adj : weighted adjacency matrix // shape : (batch_size, 1, node, node)
        mask_week : mask value by weekly input // shape : (batch_size, node, 1, 1)
        mask_day : mask value by daily input // shape : (batch_size, node, 1, 1)
        mask : whether or not to mask spatial convolution result // bool
        
        :return:
        x : output // shape : (batch_size, node, feature, timestep)
        '''
        
        for block in self.stack:
            x = block(x, weighted_adj, mask_week, mask_day, mask)
        
        return x

class SIMBAD_layers(nn.Module):
    def __init__(self, device, num_for_prediction, backbone, output_dim, dropout, modules=2):
        super(SIMBAD_layers, self).__init__()

        self.layer_1 = SIMBAD_module(device, backbone['layer_1_in_channels'], backbone['layer_1_out_channels'], backbone['adj_param'], backbone['edge_index'], dropout, modules)
        self.layer_2 = SIMBAD_module(device, backbone['layer_2_in_channels'], backbone['layer_2_out_channels'], backbone['adj_param'], backbone['edge_index'], dropout, modules)
  
        self.fc_1 = nn.Linear(4, backbone['layer_1_in_channels'])
        self.fc_2 = nn.Linear(1, backbone['layer_2_in_channels'])
        self.fc_3 = nn.Linear(backbone['layer_1_out_channels'] + backbone['layer_2_in_channels'], backbone['layer_2_in_channels'])

        self.final_fc = nn.Linear(backbone['layer_2_out_channels'], output_dim)
    
    def forward(self, x, weighted_adj, mask_week, mask_day):
        '''
        :param:
        x : input // shape : (batch, node, feature, timestep)
        weighted_adj : weighted adjacency matrix // shape : (batch_size, 1, node, node)
        mask_week : mask value by weekly input // shape : (batch_size, node, 1, 1)
        mask_day : mask value by daily input // shape : (batch_size, node, 1, 1)

        :return:
        final_output : output // shape : (batch_size, node, timestep)
        '''

        x_1 = self.fc_1(x[0].transpose(2,3)).transpose(2,3)
        layer_1_output = self.layer_1(x_1, weighted_adj[0], mask_week, mask_day, True) # B, N, F, T

        x_2 = self.fc_2(x[1].transpose(2,3)).transpose(2,3)
        layer_2_input = torch.cat([layer_1_output, x_2], dim=2)
        layer_2_input = self.fc_3(layer_2_input.transpose(2,3)).transpose(2,3)
        layer_2_output = self.layer_2(layer_2_input, weighted_adj[1], mask_week, mask_day, False)

        final_output = self.final_fc(layer_2_output.transpose(2,3)).squeeze(-1) # B, N, T

        return final_output

class SIMBAD(nn.Module):
    def __init__(self, device, num_for_prediction, backbone, adj, num_of_vertices, in_dim, output_dim, dropout, scale):
        super(SIMBAD, self).__init__()
        
        self.threshold_week = nn.Parameter(torch.full((num_of_vertices, 1), 0.2))
        self.threshold_day = nn.Parameter(torch.full((num_of_vertices, 1), 0.2))

        self.submodule = SIMBAD_layers(device, num_for_prediction, backbone, output_dim, dropout)

        self.adj = WeightedAdjacency()
        self.W = nn.Parameter(torch.ones(num_of_vertices, num_for_prediction))
        self.embed = nn.Embedding(4,2)

        self.adj_input = adj
        self.scale = scale

    def forward(self, x_list):
        '''
        x_list[0]: weekly input // shape : (batch_size, node, feature, timestep)
        x_list[1]: daily input // shape : (batch_size, node, feature, timestep)
        x_list[2]: recent input // shape : (batch_size, node, feature, timestep)
        x_list[3]: last 2 week similarity // shape : (batch_size, node, 1)
        x_list[4]: last week similarity // shape : (batch_size, node, 1)
        x_list[5]: last day similarity // shape : (batch_size, node, 1)
        x_list[6]: weekly adjacency matrix // shape : (batch_size, node, node)
        x_list[7]: daily adjacency matrix // shape : (batch_size, node, node)
        x_list[8]: recent adjacency matrix // shape : (batch_size, node, node)
        '''
        
        x_week = x_list[0].float()
        x_day = x_list[1].float()
        x_hour = x_list[2].float()

        week2_sim_value = x_list[3]
        week_sim_value = x_list[4]
        day_sim_value = x_list[5]

        week_adj = torch.sigmoid(x_list[6]) * self.adj_input
        day_adj =  torch.sigmoid(x_list[7]) * self.adj_input
        hour_adj = torch.sigmoid(x_list[8]) * self.adj_input

        # weekly input
        week_choose = (week2_sim_value < week_sim_value).float()
        x_week_input = week_choose.unsqueeze(-1) * x_week[...,:12].contiguous() + (1 - week_choose.unsqueeze(-1)) * x_week[...,12:].contiguous()
                                                 
        mask_week = week_choose * week2_sim_value + (1 - week_choose) * week_sim_value
        mask_week = torch.sigmoid(self.scale * (self.threshold_week - mask_week)).float()
        mask_week_idx = (mask_week >= 0.5).long().squeeze(-1)
        mask_week = mask_week.unsqueeze(-1) # B, N, 1, 1

        x_week_input = x_week_input.repeat(1,1,2,1)
        x_week_input = x_week_input * mask_week

        # daily input
        mask_day = torch.sigmoid(self.scale * (self.threshold_day - day_sim_value)).float()
        mask_day_idx = (mask_day >= 0.5).long().squeeze(-1)
        mask_day = mask_day.unsqueeze(-1) # B, N, 1, 1

        x_day = x_day.repeat(1,1,2,1)
        x_day_input = x_day * mask_day

        mask_final_idx = (mask_week_idx << 1) | mask_day_idx

        # embed weekly and daily
        embedding_vec = self.embed(mask_final_idx)
        x_week_final_input = x_week_input * embedding_vec.unsqueeze(-1)
        x_day_final_input = x_day_input * embedding_vec.unsqueeze(-1)

        # concat the weekly and daily inputs - layer 1
        x_final_weekday = torch.cat([x_week_final_input, x_day_final_input], dim=2)

        # recent input - layer 2
        x_final_recent = x_hour

        # weighted adjacency matrix normalization
        normalized_week_adj = batch_symmetric_normalize(week_adj).float()
        normalized_day_adj = batch_symmetric_normalize(day_adj).float()
        normalized_hour_adj = batch_symmetric_normalize(hour_adj).float()

        # balance the weekly and daily adjacency matrices
        weekday_adj = self.adj([normalized_week_adj, normalized_day_adj]).float()
        weekday_adj = weekday_adj.unsqueeze(1) # B, 1, N, N
        final_hour_adj = normalized_hour_adj.unsqueeze(1) # B, 1, N, N

        # final adjacency matrix for each layer
        final_norm_adj = [weekday_adj, final_hour_adj]

        # input for each layer
        x_final = [x_final_weekday, x_final_recent]

        output = self.submodule(x_final, final_norm_adj, mask_week, mask_day) # B, N, F, T

        # final parametrization
        final_output = self.W * output

        return final_output
