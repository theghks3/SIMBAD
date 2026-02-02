import numpy as np
import os
import csv
import torch
import torch.nn.functional as F

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

class StandardScaler:
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    :param:
    distance_df_filename : edge information file
    num_of_vertices : number of vertices

    :return:
    A : unweighted symmetric adjacency matrix (no self-loop)
    '''
    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    return A

def get_connected_nodes(adj_matrix):
    '''
    :param:
    adj_matrix : adjacency matrix (no self-loop)

    :return:
    edges : list of (node_1, node_2) / one-way / no self-loop
    '''
    adj_matrix = np.array(adj_matrix)

    edges = []

    for i in range(adj_matrix.shape[0]):
        for j in range(i+1, adj_matrix.shape[1]):
            if adj_matrix[i,j] == 1:
                edges.append((i,j))

    return edges

def adj_to_edge_index(adj_filename, num_of_vertices):
    '''
    :param:
    adj_filename : edge information file
    num_of_vertices : number of vertices
    
    :return:
    edge_index : list of edges with self-loop / size - (2, num_edges)
    '''
    adj = torch.from_numpy(get_adjacency_matrix(adj_filename, num_of_vertices))
    N = adj.shape[0]
    adj = adj + torch.eye(N)

    row, col = torch.nonzero(adj, as_tuple=True)

    edge_index = torch.stack([row, col], dim=0) # (2, num_edges)

    return edge_index

def get_backbones(adj_filename, num_of_vertices, input_dim, hidden_dim):
    '''
    :param:
    adj_filename : edge information file
    num_of_vertices : number of vertices

    :return:
    all_backbones : backbone for model
    '''
    adj = get_adjacency_matrix(adj_filename, num_of_vertices)
    adj = adj + np.eye(num_of_vertices)
    edge_index = adj_to_edge_index(adj_filename, num_of_vertices)

    backbone_1 = {
        "block_1_in_channels": hidden_dim,
        "block_1_out_channels": hidden_dim,
        "block_1_spatial_out": hidden_dim,
        "block_1_temporal_out": hidden_dim,
        "block_1_residual_out": hidden_dim,
        "block_2_in_channels": hidden_dim,
        "block_2_out_channels": hidden_dim,
        "block_2_spatial_out": hidden_dim,
        "block_2_temporal_out": hidden_dim,
        "block_2_residual_out": hidden_dim,
        "time_conv_strides": input_dim,
        "edge_index": edge_index,
        "adj_param": torch.from_numpy(adj)
    }

    all_backbones = [backbone_1]

    return all_backbones

def batch_symmetric_normalize(adj):
    """
    adj: (B, N, N) weighted adjacency matrices with self-loops
    return: (B, N, N) normalized adjacency matrices
    """
    B, N, _ = adj.shape

    # Degree: (B, N)
    deg = adj.sum(dim=2)  # sum over rows → degree for each node
    #print('deg', deg[:,165])

    # D^{-1/2}: (B, N)
    deg_inv_sqrt = torch.pow(deg + 1e-10, -0.5)
    #print('nan',torch.nonzero(torch.isnan(deg_inv_sqrt), as_tuple=False))
    #print('inv', deg_inv_sqrt[:,165])

    # Reshape for broadcasting: (B, N, 1) and (B, 1, N)
    D_left = deg_inv_sqrt.unsqueeze(2)     # (B, N, 1)
    D_right = deg_inv_sqrt.unsqueeze(1)    # (B, 1, N)

    # Normalize: A_hat = D^{-1/2} A D^{-1/2}
    adj_norm = D_left * adj * D_right      # broadcasting multiplication

    return adj_norm  # shape: (B, N, N)