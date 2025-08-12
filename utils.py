import numpy as np
import csv
import torch

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

def masked_softmax_tensor(matrix):
    '''
    :param:
    matrix : torch.Tensor of shape (batch or 1, N, N), adjacency matrix (no self-loop)

    :return:
    softmax_matrix : row-wise softmax applied only to non-zero entries
    '''
    # mask: True where matrix != 0
    mask = (matrix != 0)

    # Replace 0s with -inf for masking in softmax
    masked_matrix = matrix.masked_fill(~mask, float('-inf'))

    # Stabilize with max subtraction
    max_per_row = torch.max(masked_matrix, dim=-1, keepdim=True).values
    exps = torch.exp(masked_matrix - max_per_row)

    # Zero out the masked parts (optional since -inf→0, but keeps it safe)
    exps = exps * mask

    sum_exps = torch.sum(exps, dim=-1, keepdim=True) + 1e-10
    softmax_matrix = exps / sum_exps

    # Keep masked parts as 0
    softmax_matrix = softmax_matrix * mask

    return softmax_matrix

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

    backbone = {
        "layer_1_in_channels": hidden_dim,
        "layer_1_out_channels": hidden_dim,
        "layer_2_in_channels": hidden_dim,
        "layer_2_out_channels": hidden_dim,
        "time_conv_strides": input_dim,
        "edge_index": edge_index,
        "adj_param": torch.from_numpy(adj)
    }

    return backbone

def calculate_similarity(x, y):
    '''
    :param:
    x : torch.Tensor of shape (batch_size, num_nodes, seq_len)
    y : torch.Tensor of shape (batch_size, num_nodes, seq_len)

    :return:
    similarity : torch.Tensor of shape (batch_size, num_nodes, num_nodes)
    '''
    x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-10 # The L2 norm of x
    y_norm = torch.norm(y, dim=-1, keepdim=True) + 1e-10 # The L2 norm of y
    
    # cosine similarity calculation (x dot y / (x_norm * y_norm))
    similarity = torch.bmm(x, y.transpose(1, 2)) / (x_norm * y_norm.transpose(1, 2))
    
    return similarity
 
def get_sample_matrix(x, y):
    '''
    :param:
    x : input // shape : (batch_size, node, feature, timestep)
    y : input // shape : (batch_size, node, feature, timestep)

    :return:
    similarity_matrix : weighted adjacency matrix // shape : (batch_size, node, node)
    '''
    # calculate the cosine similarity of each input
    x = x.squeeze(2)
    y = y.squeeze(2)
    similarity_matrix = calculate_similarity(x, y)
    
    # the cosine similarity of each node except for self-loop
    eye = torch.eye(similarity_matrix.size(1), device=similarity_matrix.device).unsqueeze(0)
    similarity_matrix = similarity_matrix * (1 - eye) 
    
    return similarity_matrix

def batch_symmetric_normalize(adj):
    '''
    :param:
    adj : weighted adjacency matrices // shape : (batch_size, node, node)

    :return:
    adj_norm : normalized adjacency matrices for each batch incdex // shape : (batch_size, node, node)
    '''
    B, N, _ = adj.shape

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
