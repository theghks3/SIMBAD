import csv

import numpy as np
import torch


def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


class StandardScaler:
    """Standardize NumPy arrays and tensors with shared statistics."""

    def __init__(self, mean=None, std=None):
        if std is not None and np.any(np.asarray(std) == 0):
            raise ValueError("standard deviation must be non-zero")
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()
        if np.any(np.asarray(self.std) == 0):
            raise ValueError("standard deviation must be non-zero")

        return (data - self.mean) / self.std

    def transform(self, data):
        mean, std = self._parameters_for(data)
        return (data - mean) / std

    def inverse_transform(self, data):
        mean, std = self._parameters_for(data)
        return (data * std) + mean

    def _parameters_for(self, data):
        if torch.is_tensor(data):
            mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
            std = torch.as_tensor(self.std, dtype=data.dtype, device=data.device)
            return mean, std
        return self.mean, self.std


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    :param:
    distance_df_filename : edge information file
    num_of_vertices : number of vertices

    :return:
    A : unweighted symmetric adjacency matrix (no self-loop)
    '''
    with open(distance_df_filename, 'r', newline='') as f:
        reader = csv.reader(f)
        edges = []
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 2:
                raise ValueError(f"invalid adjacency row {row_number}: {row}")
            try:
                source, target = int(row[0]), int(row[1])
            except ValueError:
                if row_number == 1:
                    continue
                raise ValueError(f"invalid adjacency row {row_number}: {row}")
            if not (0 <= source < num_of_vertices and 0 <= target < num_of_vertices):
                raise ValueError(
                    f"node index out of range at adjacency row {row_number}"
                )
            edges.append((source, target))

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    return A

