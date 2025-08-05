import numpy as np
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from time import time

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


def search_data(sequence_length, num_of_batches, label_start_idx, num_for_predict, units, window, points_per_hour=12):
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
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")
    if label_start_idx + num_for_predict > sequence_length:
        return None
    
    x_idx = []
    for i in range(1, num_of_batches+1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= window:
            x_idx.append((start_idx, end_idx))
        else:
            return None
    
    if len(x_idx) != num_of_batches:
        return None
    
    return x_idx[::-1]

def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours, label_start_idx, num_for_predict, window, points_per_hour=12):
    week_indices = search_data(data_sequence.shape[0], num_of_weeks, label_start_idx, num_for_predict, 7*24, window, points_per_hour)
    if not week_indices:
        return None

    day_indices = search_data(data_sequence.shape[0], num_of_days, label_start_idx, num_for_predict, 24, window, points_per_hour)
    if not day_indices:
        return None

    hour_indices = search_data(data_sequence.shape[0], num_of_hours, label_start_idx, num_for_predict, 1, window, points_per_hour)
    if not hour_indices:
        return None

    week_sample = np.concatenate([data_sequence[i:j] for i, j in week_indices], axis=0)
    day_sample = np.concatenate([data_sequence[i:j] for i, j in day_indices], axis=0)
    hour_sample = np.concatenate([data_sequence[i:j] for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    p, q = hour_indices[-1]
    last2_week = np.concatenate([data_sequence[p-12*24*7*2+6:q-12*24*7*2]], axis=0)
    last_week = np.concatenate([data_sequence[p-12*24*7+6:q-12*24*7]], axis=0)
    last_day = np.concatenate([data_sequence[p-12*24+6:q-12*24]], axis=0)
    recent_data = np.concatenate([data_sequence[p+6:q]], axis=0)

    week2_sim = np.linalg.norm(last2_week - recent_data, ord=1, axis=0).transpose(1,0)
    week_sim = np.linalg.norm(last_week - recent_data, ord=1, axis=0).transpose(1,0)
    day_sim = np.linalg.norm(last_day - recent_data, ord=1, axis=0).transpose(1,0)

    return week_sample, day_sample, hour_sample, target, week2_sim, week_sim, day_sim

def read_and_generate_dataset(graph_signal_matrix_filename, num_of_weeks, num_of_days, num_of_hours, num_for_predict, window, input_dim, train_ratio, val_ratio, points_per_hour=12):
    data_seq = np.load(graph_signal_matrix_filename)['data'][...,:input_dim]
    print('Prepare data')

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days, num_of_hours, idx, num_for_predict, window, points_per_hour)
        if not sample:
            continue 
        week_sample, day_sample, hour_sample, target, week2_sim, week_sim, day_sim = sample

        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0,2,3,1)),
            np.expand_dims(day_sample, axis=0).transpose((0,2,3,1)),
            np.expand_dims(hour_sample, axis=0).transpose((0,2,3,1)),
            np.expand_dims(target, axis=0).transpose((0,2,3,1))[:,:,0,:],
            np.expand_dims(week2_sim, axis=0).transpose((0,2,1)),
            np.expand_dims(week_sim, axis=0).transpose((0,2,1)),
            np.expand_dims(day_sim, axis=0).transpose((0,2,1))
        ))
    split_line1 = int(len(all_samples) * train_ratio)
    split_line2 = int(len(all_samples) * (train_ratio + val_ratio))

    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1:split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

    train_week, train_day, train_hour, train_target, train_week2_sim, train_week_sim, train_day_sim = training_set
    val_week, val_day, val_hour, val_target, val_week2_sim, val_week_sim, val_day_sim = validation_set
    test_week, test_day, test_hour, test_target, test_week2_sim, test_week_sim, test_day_sim = testing_set

    week_stats = get_stats_numpy(train_week)
    day_stats = get_stats_numpy(train_day)
    hour_stats = get_stats_numpy(train_hour)
    
    (_, train_week2_sim_norm, val_week2_sim_norm, test_week2_sim_norm) = normalization(train_week2_sim, val_week2_sim, test_week2_sim)
    (_, train_week_sim_norm, val_week_sim_norm, test_week_sim_norm) = normalization(train_week_sim, val_week_sim, test_week_sim)
    (_, train_day_sim_norm, val_day_sim_norm, test_day_sim_norm) = normalization(train_day_sim, val_day_sim, test_day_sim)

    print(f'Train week: {train_week.shape},Train day: {train_day.shape}, Train hour: {train_hour.shape}')
    print(f'Val week: {val_week.shape},Val day: {val_day.shape}, Val hour: {val_hour.shape}')
    print(f'Test week: {test_week.shape},Test day: {test_day.shape}, Test hour: {test_hour.shape}')

    all_data = {
        'train': {
            'week': train_week,
            'day': train_day,
            'hour': train_hour,
            'week2_sim': train_week2_sim_norm,
            'week_sim': train_week_sim_norm,
            'day_sim': train_day_sim_norm,
            'target': train_target
        },
        'val': {
            'week': val_week,
            'day': val_day,
            'hour': val_hour,
            'week2_sim': val_week2_sim_norm,
            'week_sim': val_week_sim_norm,
            'day_sim': val_day_sim_norm,
            'target': val_target
        },
        'test': {
            'week': test_week,
            'day': test_day,
            'hour': test_hour,
            'week2_sim': test_week2_sim_norm,
            'week_sim': test_week_sim_norm,
            'day_sim': test_day_sim_norm,
            'target': test_target
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'hour': hour_stats
        }
    }

    return all_data

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
