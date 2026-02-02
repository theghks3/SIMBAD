import numpy as np
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from time import time

def get_semantic_adjacency(x, y, neigh_rate, eps=1e-10, self_loop=True):
    """
    x: (time, node, feature)
    return: (node, node)
    """
    time, num_node, feat = x.shape
    neigh_num = max(1, int(num_node * neigh_rate))

    # (time, node, feature) -> (node, time * feature)
    x_1 = x.transpose(1, 0, 2).reshape(num_node, time * feat)
    y_1 = y.transpose(1, 0, 2).reshape(num_node, time * feat)

    x_norm = np.linalg.norm(x_1, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y_1, axis=1, keepdims=True)

    sim = (x_1 @ y_1.T) / (x_norm * y_norm.T + eps)

    sim2 = sim.copy()
    np.fill_diagonal(sim2, -np.inf)

    topk = np.argpartition(sim2, -neigh_num, axis=1)[:, -neigh_num:]

    adj = np.zeros((num_node, num_node), dtype=np.float32)
    rows = np.arange(num_node)[:, None]
    adj[rows, topk] = 1.0

    if self_loop:
        np.fill_diagonal(adj, 1.0)
    #print(np.where(adj[0]==1.))

    return adj

def normalization(train, val, test):
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    min_val = train.min(axis=0, keepdims=True)
    max_val = train.max(axis=0, keepdims=True)

    def normalize(x):
        return (x - min_val) / (max_val - min_val + 1e-10)

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'min': min_val, 'max': max_val}, train_norm, val_norm, test_norm

def normalization_given_stat(x, stats):
    min_val = stats['min']
    max_val = stats['max']

    def normalize(x, min_value, max_value):
        return (x - min_value) / (max_value - min_val + 1e-10)
    
    x_norm = normalize(x, min_val, max_val)

    return x_norm

def get_stats(x):
    min_val = x.min(dim=0, keepdims=True).values
    max_val = x.max(dim=0, keepdims=True).values

    return {'min': min_val, 'max': max_val}

def get_stats_numpy(x):
    min_val = x.min(axis=0, keepdims=True)
    max_val = x.max(axis=0, keepdims=True)

    return {'min': min_val, 'max': max_val}


def search_data(sequence_length, num_of_batches, label_start_idx, num_for_predict, units, points_per_hour=12):
    '''
    :param:
    sequence_length : length of total sequence
    num_of_batches : number of batches
    label_start_idx : start index for each input
    num_for_predict : number of data for prediction
    units : unit of week/day/hour
    points_per_hour : number of points for an hour

    :return:
    x_idx : list of tuple of each input index
    '''
    if points_per_hour<0:
        raise ValueError("points_per_hour should be greater than 0!")
    if label_start_idx + num_for_predict > sequence_length:
        return None
    
    x_idx = []
    for i in range(1, num_of_batches+1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 6:
            x_idx.append((start_idx, end_idx))
        else:
            return None
    
    if len(x_idx) != num_of_batches:
        return None
    
    return x_idx[::-1]

def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours, label_start_idx, num_for_predict, points_per_hour=12, scaler=None):
    week_indices = search_data(data_sequence.shape[0], num_of_weeks, label_start_idx, num_for_predict, 7*24, points_per_hour)
    if not week_indices:
        return None

    day_indices = search_data(data_sequence.shape[0], num_of_days, label_start_idx, num_for_predict, 24, points_per_hour)
    if not day_indices:
        return None

    hour_indices = search_data(data_sequence.shape[0], num_of_hours, label_start_idx, num_for_predict, 1, points_per_hour)
    if not hour_indices:
        return None

    if scaler:
        i, j = hour_indices[-1]
        last2_week = np.concatenate([data_sequence[i-12*24*7*2+6:j-12*24*7*2]], axis=0)
        last_week = np.concatenate([data_sequence[i-12*24*7+6:j-12*24*7]], axis=0)
        last_day = np.concatenate([data_sequence[i-12*24+6:j-12*24]], axis=0)
        this_week = np.concatenate([data_sequence[i+6:j]], axis=0)

        #last2_week = np.concatenate([data_sequence[i-12*24*7*2:j-12*24*7*2]], axis=0)
        #last_week = np.concatenate([data_sequence[i-12*24*7:j-12*24*7]], axis=0)
        #last_day = np.concatenate([data_sequence[i-12*24:j-12*24]], axis=0)
        #this_week = np.concatenate([data_sequence[i:j]], axis=0)

        '''last2_week = scaler.transform(last2_week)
        last_week = scaler.transform(last_week)
        last_day = scaler.transform(last_day)
        this_week = scaler.transform(this_week)'''

        week2_sim = np.linalg.norm(last2_week - this_week, ord=1, axis=0).transpose(1,0)
        week_sim = np.linalg.norm(last_week - this_week, ord=1, axis=0).transpose(1,0)
        day_sim = np.linalg.norm(last_day - this_week, ord=1, axis=0).transpose(1,0)

        return week2_sim, week_sim, day_sim
    
    else:
        week_sample = np.concatenate([data_sequence[i:j] for i, j in week_indices], axis=0)
        day_sample = np.concatenate([data_sequence[i:j] for i, j in day_indices], axis=0)
        hour_sample = np.concatenate([data_sequence[i:j] for i, j in hour_indices], axis=0)
        target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

        return week_sample, day_sample, hour_sample, target

def read_and_generate_dataset(adj_filename, graph_signal_matrix_filename, num_of_weeks, num_of_days, num_of_hours, num_of_vertices, num_for_predict, input_dim, train_ratio, val_ratio, points_per_hour=12, node_ratio=0.05):
    data_seq = np.load(graph_signal_matrix_filename)['data'][...,:input_dim]
    print('Prepare data')

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days, num_of_hours, idx, num_for_predict, points_per_hour)
        if not sample:
            continue 
        week_sample, day_sample, hour_sample, target = sample

        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0,2,3,1)),
            np.expand_dims(day_sample, axis=0).transpose((0,2,3,1)),
            np.expand_dims(hour_sample, axis=0).transpose((0,2,3,1)),
            np.expand_dims(target, axis=0).transpose((0,2,3,1))[:,:,0,:]
        ))
    split_line1 = int(len(all_samples) * train_ratio)
    split_line2 = int(len(all_samples) * (train_ratio + val_ratio))

    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1:split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target = validation_set
    test_week, test_day, test_hour, test_target = testing_set

    train_dataset = np.concatenate([train_week, train_day, train_hour], axis=-1)
    scale = StandardScaler(mean=train_dataset.mean(), std=train_dataset.std())

    train_week = scale.transform(train_week)
    train_day = scale.transform(train_day)
    train_hour = scale.transform(train_hour)

    val_week = scale.transform(val_week)
    val_day = scale.transform(val_day)
    val_hour = scale.transform(val_hour)

    test_week = scale.transform(test_week)
    test_day = scale.transform(test_day)
    test_hour = scale.transform(test_hour)

    train_len = train_week.shape[0] + 12

    # Semantic adjacency
    sem_week2 = data_seq[6:6+train_len]
    sem_week = data_seq[2022:2022+train_len]
    sem_day = data_seq[3750:3750+train_len]
    sem_hour = data_seq[4026:4026+train_len]

    sem_his = np.concatenate([sem_week2, sem_week, sem_day], axis=0)
    sem_rec = np.concatenate([sem_hour, sem_hour, sem_hour], axis=0)

    his_sem = torch.from_numpy(get_semantic_adjacency(sem_rec, sem_his, node_ratio))

    #(week_stats, train_week_norm, val_week_norm, test_week_norm) = normalization(train_week, val_week, test_week)
    #(day_stats, train_day_norm, val_day_norm, test_day_norm) = normalization(train_day, val_day, test_day)
    #(hour_stats, train_hour_norm, val_hour_norm, test_hour_norm) = normalization(train_hour, val_hour, test_hour)

    remaining_samples = []
    indexing = 0

    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days, num_of_hours, idx, num_for_predict, points_per_hour, True)

        if not sample:
            indexing += 1
            continue
        new_idx = idx - indexing
        week2_sim, week_sim, day_sim = sample

        remaining_samples.append((
            np.expand_dims(week2_sim, axis=0).transpose((0,2,1)),
            np.expand_dims(week_sim, axis=0).transpose((0,2,1)),
            np.expand_dims(day_sim, axis=0).transpose((0,2,1))
        ))

    split_line1 = int(len(remaining_samples) * train_ratio)
    split_line2 = int(len(remaining_samples) * (train_ratio + val_ratio))

    remaining_training_set = [np.concatenate(i, axis=0) for i in zip(*remaining_samples[:split_line1])]
    remaining_val_set = [np.concatenate(i, axis=0) for i in zip(*remaining_samples[split_line1: split_line2])]
    remaining_test_set = [np.concatenate(i, axis=0) for i in zip(*remaining_samples[split_line2:])]

    train_week2_sim, train_week_sim, train_day_sim = remaining_training_set
    val_week2_sim, val_week_sim, val_day_sim = remaining_val_set
    test_week2_sim, test_week_sim, test_day_sim = remaining_test_set

    # Best performance
    week_sim_data = np.concatenate([train_week2_sim, train_week_sim], axis=0)
    
    scaler_week = StandardScaler(mean=week_sim_data.mean(), std=week_sim_data.std())
    scaler_day = StandardScaler(mean=train_day_sim.mean(), std=train_day_sim.std())

    train_week2_sim = scaler_week.transform(train_week2_sim)
    train_week_sim = scaler_week.transform(train_week_sim)
    train_day_sim = scaler_day.transform(train_day_sim)

    val_week2_sim = scaler_week.transform(val_week2_sim)
    val_week_sim = scaler_week.transform(val_week_sim)
    val_day_sim = scaler_day.transform(val_day_sim)

    test_week2_sim = scaler_week.transform(test_week2_sim)
    test_week_sim = scaler_week.transform(test_week_sim)
    test_day_sim = scaler_day.transform(test_day_sim)

    total_week_sim = np.concatenate([train_week2_sim, train_week_sim], axis=0)

    mean_week = np.mean(total_week_sim, axis=0)
    std_week = np.std(total_week_sim, axis=0)

    mean_day = np.mean(train_day_sim, axis=0)
    std_day = np.std(train_day_sim, axis=0)

    threshold_week = mean_week + 2 * std_week
    threshold_day = mean_day + 2 * std_day
    
    print(f'Train week: {train_week.shape}, Train day: {train_day.shape}, Train hour: {train_hour.shape}')
    print(f'Val week: {val_week.shape}, Val day: {val_day.shape}, Val hour: {val_hour.shape}')
    print(f'Test week: {test_week.shape}, Test day: {test_day.shape}, Test hour: {test_hour.shape}')

    all_data = {
        'train': {
            'week': train_week,
            'day': train_day,
            'hour': train_hour,
            'week2_sim': train_week2_sim,
            'week_sim': train_week_sim,
            'day_sim': train_day_sim,
            'target': train_target
        },
        'val': {
            'week': val_week,
            'day': val_day,
            'hour': val_hour,
            'week2_sim': val_week2_sim,
            'week_sim': val_week_sim,
            'day_sim': val_day_sim,
            'target': val_target
        },
        'test': {
            'week': test_week,
            'day': test_day,
            'hour': test_hour,
            'week2_sim': test_week2_sim,
            'week_sim': test_week_sim,
            'day_sim': test_day_sim,
            'target': test_target
        },
        'threshold': {
            'week': threshold_week,
            'day': threshold_day
        },
        'sem_adj': his_sem,
        'stats': scale
    }

    return all_data

def save_dataset(all_data, save_name):
    week_train_val = np.concatenate((all_data['train']['week'], all_data['val']['week']), axis=0)
    all_week = np.concatenate((week_train_val, all_data['test']['week']), axis=0)   

    day_train_val = np.concatenate((all_data['train']['day'], all_data['val']['day']), axis=0)
    all_day = np.concatenate((day_train_val, all_data['test']['day']), axis=0)

    hour_train_val = np.concatenate((all_data['train']['hour'], all_data['val']['hour']), axis=0)
    all_recent = np.concatenate((hour_train_val, all_data['test']['hour']), axis=0)

    week_train_val_adj = np.concatenate((all_data['train']['week_adj'], all_data['val']['week_adj']), axis=0)
    all_week_adj = np.concatenate((week_train_val_adj, all_data['test']['week_adj']), axis=0)

    day_train_val_adj = np.concatenate((all_data['train']['day_adj'], all_data['val']['day_adj']), axis=0)
    all_day_adj = np.concatenate((day_train_val_adj, all_data['test']['day_adj']), axis=0)

    hour_train_val_adj = np.concatenate((all_data['train']['hour_adj'], all_data['val']['hour_adj']), axis=0)
    all_recent_adj = np.concatenate((hour_train_val_adj, all_data['test']['hour_adj']), axis=0)

    week2_l1_train_val = np.concatenate((all_data['train']['week2_l1'], all_data['val']['week2_l1']), axis=0)
    all_week2_l1 = np.concatenate((week2_l1_train_val, all_data['test']['week2_l1']), axis=0)

    week_l1_train_val = np.concatenate((all_data['train']['week_l1'], all_data['val']['week_l1']), axis=0)
    all_week_l1 = np.concatenate((week_l1_train_val, all_data['test']['week_l1']), axis=0)

    day_l1_train_val = np.concatenate((all_data['train']['day_l1'], all_data['val']['day_l1']), axis=0)
    all_day_l1 = np.concatenate((day_l1_train_val, all_data['test']['day_l1']), axis=0)

    target_train_val = np.concatenate((all_data['train']['target'], all_data['val']['target']), axis=0)
    all_target = np.concatenate((target_train_val, all_data['test']['target']), axis=0)

    week = torch.from_numpy(all_week)
    day = torch.from_numpy(all_day)
    hour = torch.from_numpy(all_recent)
    week_adj = torch.from_numpy(all_week_adj)
    day_adj = torch.from_numpy(all_day_adj)
    hour_adj = torch.from_numpy(all_recent_adj)
    week2_l1 = torch.from_numpy(all_week2_l1)
    week_l1 = torch.from_numpy(all_week_l1)
    day_l1 = torch.from_numpy(all_day_l1)
    target = torch.from_numpy(all_target)

    tensor_list = [week, day, hour, week_adj, day_adj, hour_adj, week2_l1, week_l1, day_l1, target]
    torch.save(tensor_list, save_name)

def get_final_dataset(final_dataset, batch_size):
    train_data = final_dataset['train']
    val_data = final_dataset['val']
    test_data = final_dataset['test']

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_data['week']),
            torch.tensor(train_data['day']),
            torch.tensor(train_data['hour']),
            torch.tensor(train_data['week2_sim']),
            torch.tensor(train_data['week_sim']),
            torch.tensor(train_data['day_sim']),
            torch.tensor(train_data['target'])
        ),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(val_data['week']),
            torch.tensor(val_data['day']),
            torch.tensor(val_data['hour']),
            torch.tensor(val_data['week2_sim']),
            torch.tensor(val_data['week_sim']),
            torch.tensor(val_data['day_sim']),
            torch.tensor(val_data['target'])
        ),
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(test_data['week']),
            torch.tensor(test_data['day']),
            torch.tensor(test_data['hour']),
            torch.tensor(test_data['week2_sim']),
            torch.tensor(test_data['week_sim']),
            torch.tensor(test_data['day_sim']),
            torch.tensor(test_data['target'])
        ),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, final_dataset['stats']