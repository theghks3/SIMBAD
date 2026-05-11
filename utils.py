import numpy as np
import csv
import pickle

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

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

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

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def get_adjacency_metrbay(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)

    adj = (adj_mx + adj_mx.T) / 2
    adj = (adj != 0).astype(int)

    return adj

def get_adjacency_sd(npy_filename, num_nodes):
    adj = np.load(npy_filename)
    eye = np.eye(num_nodes)

    adj = adj - eye
    adj = (adj + adj.T) / 2
    adj = (adj != 0).astype(int)

    return adj
