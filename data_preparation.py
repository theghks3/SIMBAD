import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import *

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

def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours, label_start_idx, num_for_predict, points_per_hour=12):
    week_indices = search_data(data_sequence.shape[0], num_of_weeks, label_start_idx, num_for_predict, 7*24, points_per_hour)
    if not week_indices:
        return None

    day_indices = search_data(data_sequence.shape[0], num_of_days, label_start_idx, num_for_predict, 24, points_per_hour)
    if not day_indices:
        return None

    hour_indices = search_data(data_sequence.shape[0], num_of_hours, label_start_idx, num_for_predict, 1, points_per_hour)
    if not hour_indices:
        return None

    week_sample = np.concatenate([data_sequence[i:j] for i, j in week_indices], axis=0)
    day_sample = np.concatenate([data_sequence[i:j] for i, j in day_indices], axis=0)
    hour_sample = np.concatenate([data_sequence[i:j] for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    m, n = hour_indices[-1]
    last2_week = np.concatenate([data_sequence[m-12*24*7*2+6:n-12*24*7*2]], axis=0)
    last_week = np.concatenate([data_sequence[m-12*24*7+6:n-12*24*7]], axis=0)
    last_day = np.concatenate([data_sequence[m-12*24+6:n-12*24]], axis=0)
    this_week = np.concatenate([data_sequence[m+6:n]], axis=0)

    week2_sim = np.linalg.norm(last2_week - this_week, ord=1, axis=0).transpose(1,0)
    week_sim = np.linalg.norm(last_week - this_week, ord=1, axis=0).transpose(1,0)
    day_sim = np.linalg.norm(last_day - this_week, ord=1, axis=0).transpose(1,0)

    o, p = label_start_idx, label_start_idx + 6

    last2_week_target = np.concatenate([data_sequence[o-12*24*7*2:p-12*24*7*2]], axis=0)
    last_week_target = np.concatenate([data_sequence[o-12*24*7:p-12*24*7]], axis=0)
    last_day_target = np.concatenate([data_sequence[o-12*24:p-12*24]], axis=0)
    this_week_target = np.concatenate([data_sequence[o:p]], axis=0)

    week2_sim_target = np.linalg.norm(last2_week_target - this_week_target, ord=1, axis=0).transpose(1,0)
    week_sim_target = np.linalg.norm(last_week_target - this_week_target, ord=1, axis=0).transpose(1,0)
    day_sim_target = np.linalg.norm(last_day_target - this_week_target, ord=1, axis=0).transpose(1,0)

    return week_sample, day_sample, hour_sample, target, week2_sim, week_sim, day_sim, week2_sim_target, week_sim_target, day_sim_target

def read_and_generate_dataset(log, graph_signal_matrix_filename, num_of_weeks, num_of_days, num_of_hours, num_for_predict, input_dim, train_ratio, val_ratio, points_per_hour=12):
    data_seq = np.load(graph_signal_matrix_filename)['data'][...,:input_dim]
    print('Prepare data')

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days, num_of_hours, idx, num_for_predict, points_per_hour)
        if not sample:
            continue 
        week_sample, day_sample, hour_sample, target, week2_sim, week_sim, day_sim, week2_sim_target, week_sim_target, day_sim_target = sample

        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0,2,3,1)),
            np.expand_dims(day_sample, axis=0).transpose((0,2,3,1)),
            np.expand_dims(hour_sample, axis=0).transpose((0,2,3,1)),
            np.expand_dims(target, axis=0).transpose((0,2,3,1))[:,:,0,:],
            np.expand_dims(week2_sim, axis=0).transpose((0,2,1)),
            np.expand_dims(week_sim, axis=0).transpose((0,2,1)),
            np.expand_dims(day_sim, axis=0).transpose((0,2,1)),
            np.expand_dims(week2_sim_target, axis=0).transpose((0,2,1)),
            np.expand_dims(week_sim_target, axis=0).transpose((0,2,1)),
            np.expand_dims(day_sim_target, axis=0).transpose((0,2,1))
        ))
    split_line1 = int(len(all_samples) * train_ratio)
    split_line2 = int(len(all_samples) * (train_ratio + val_ratio))

    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1:split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

    train_week, train_day, train_hour, train_target, train_week2_sim, train_week_sim, train_day_sim, train_week2_target_sim, train_week_target_sim, train_day_target_sim = training_set
    val_week, val_day, val_hour, val_target, val_week2_sim, val_week_sim, val_day_sim, _, _, _ = validation_set
    test_week, test_day, test_hour, test_target, test_week2_sim, test_week_sim, test_day_sim, _, _, _ = testing_set

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

    sim_data = np.concatenate([train_week2_sim, train_week_sim, train_day_sim], axis=0)
    
    scaler_sim = StandardScaler(mean=sim_data.mean(), std=sim_data.std())

    train_week2_sim = scaler_sim.transform(train_week2_sim)
    train_week_sim = scaler_sim.transform(train_week_sim)
    train_day_sim = scaler_sim.transform(train_day_sim)

    final_train_sim = np.concatenate([train_week2_sim, train_week_sim, train_day_sim], axis=-1)

    train_week2_target_sim = scaler_sim.transform(train_week2_target_sim)
    train_week_target_sim = scaler_sim.transform(train_week_target_sim)
    train_day_target_sim = scaler_sim.transform(train_day_target_sim)

    final_train_target_sim = np.concatenate([train_week2_target_sim, train_week_target_sim, train_day_target_sim], axis=-1)

    val_week2_sim = scaler_sim.transform(val_week2_sim)
    val_week_sim = scaler_sim.transform(val_week_sim)
    val_day_sim = scaler_sim.transform(val_day_sim)

    final_val_sim = np.concatenate([val_week2_sim, val_week_sim, val_day_sim], axis=-1)

    test_week2_sim = scaler_sim.transform(test_week2_sim)
    test_week_sim = scaler_sim.transform(test_week_sim)
    test_day_sim = scaler_sim.transform(test_day_sim)

    final_test_sim = np.concatenate([test_week2_sim, test_week_sim, test_day_sim], axis=-1)

    total_sim = np.concatenate([train_week2_sim, train_week_sim, train_day_sim], axis=0)

    q3 = np.percentile(total_sim, 75, axis=0)
    q1 = np.percentile(total_sim, 25, axis=0)
    threshold_sim = q3 + 3.0 * (q3 - q1)
    
    log_string(log, f'Train week: {train_week.shape}, Train day: {train_day.shape}, Train hour: {train_hour.shape}')
    log_string(log, f'Val week: {val_week.shape}, Val day: {val_day.shape}, Val hour: {val_hour.shape}')
    log_string(log, f'Test week: {test_week.shape}, Test day: {test_day.shape}, Test hour: {test_hour.shape}')

    all_data = {
        'train': {
            'week': train_week,
            'day': train_day,
            'hour': train_hour,
            'sim': final_train_sim,
            'sim_target': final_train_target_sim,
            'target': train_target
        },
        'val': {
            'week': val_week,
            'day': val_day,
            'hour': val_hour,
            'sim': final_val_sim,
            'target': val_target
        },
        'test': {
            'week': test_week,
            'day': test_day,
            'hour': test_hour,
            'sim': final_test_sim,
            'target': test_target
        },
        'threshold': {
            'sim': threshold_sim
        },
        'stats': {
            'input': scale,
            'sim': scaler_sim
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
            torch.tensor(train_data['sim']),
            torch.tensor(train_data['sim_target']),
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
            torch.tensor(val_data['sim']),
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
            torch.tensor(test_data['sim']),
            torch.tensor(test_data['target'])
        ),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, final_dataset['stats']
