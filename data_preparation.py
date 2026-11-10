import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import StandardScaler


HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
HISTORY_KEYS = ('week', 'day', 'hour')


def search_data(
    sequence_length,
    num_of_batches,
    label_start_idx,
    num_for_predict,
    units,
    points_per_hour=12,
    minimum_start=0,
):
    """Return oldest-to-newest history intervals for one temporal scale."""
    if points_per_hour <= 0:
        raise ValueError('points_per_hour must be positive')
    if num_of_batches < 0:
        raise ValueError('num_of_batches must be non-negative')
    if label_start_idx + num_for_predict > sequence_length:
        return None

    intervals = []
    for lag in range(num_of_batches, 0, -1):
        start = label_start_idx - points_per_hour * units * lag
        if start < minimum_start:
            return None
        intervals.append((start, start + num_for_predict))

    return intervals


def _concatenate_intervals(data, intervals):
    if not intervals:
        return np.empty((0, *data.shape[1:]), dtype=data.dtype)
    return np.concatenate([data[start:end] for start, end in intervals], axis=0)


def _node_l1_distance(candidate, reference):
    """Compute one L1 distance per node over time and feature dimensions."""
    return np.abs(candidate - reference).sum(axis=(0, 2))


def _similarities_for_candidates(
    data,
    candidate_intervals,
    label_start_idx,
    similarity_window,
):
    recent_reference = data[
        label_start_idx - similarity_window:label_start_idx
    ]
    target_reference = data[
        label_start_idx:label_start_idx + similarity_window
    ]

    recent_similarities = []
    target_similarities = []
    for candidate_start, _ in candidate_intervals:
        recent_candidate = data[
            candidate_start - similarity_window:candidate_start
        ]
        target_candidate = data[
            candidate_start:candidate_start + similarity_window
        ]
        recent_similarities.append(
            _node_l1_distance(recent_candidate, recent_reference)
        )
        target_similarities.append(
            _node_l1_distance(target_candidate, target_reference)
        )

    return (
        np.stack(recent_similarities, axis=-1),
        np.stack(target_similarities, axis=-1),
    )


def get_sample_indices(
    data_sequence,
    num_of_weeks,
    num_of_days,
    num_of_hours,
    label_start_idx,
    num_for_predict,
    points_per_hour=12,
    similarity_window=6,
):
    """Build history, target, and similarity arrays for one prediction point."""
    if not 0 < similarity_window <= num_for_predict:
        raise ValueError('similarity_window must be between 1 and num_for_predict')
    if num_of_weeks < 0 or num_of_days < 0 or num_of_weeks + num_of_days < 1:
        raise ValueError('weeks and days must be non-negative with a positive sum')
    if num_of_hours < 1:
        raise ValueError('hours must be positive')

    interval_specs = {
        'week': (num_of_weeks, DAYS_PER_WEEK * HOURS_PER_DAY),
        'day': (num_of_days, HOURS_PER_DAY),
        'hour': (num_of_hours, 1),
    }
    intervals = {
        name: search_data(
            sequence_length=data_sequence.shape[0],
            num_of_batches=count,
            label_start_idx=label_start_idx,
            num_for_predict=num_for_predict,
            units=units,
            points_per_hour=points_per_hour,
            minimum_start=similarity_window,
        )
        for name, (count, units) in interval_specs.items()
    }
    if any(value is None for value in intervals.values()):
        return None

    histories = {
        name: _concatenate_intervals(data_sequence, intervals[name])
        for name in HISTORY_KEYS
    }
    target = data_sequence[
        label_start_idx:label_start_idx + num_for_predict
    ]
    candidate_intervals = intervals['week'] + intervals['day']
    recent_similarities, target_similarities = _similarities_for_candidates(
        data_sequence,
        candidate_intervals,
        label_start_idx,
        similarity_window,
    )

    return (
        histories['week'],
        histories['day'],
        histories['hour'],
        target,
        recent_similarities,
        target_similarities,
    )


def _history_to_model_layout(history):
    """Convert (time, nodes, features) to (1, nodes, features, time)."""
    return history.transpose(1, 2, 0)[None, ...]


def _format_sample(sample, target_channel):
    week, day, hour, target, similarity, similarity_target = sample
    return {
        'week': _history_to_model_layout(week),
        'day': _history_to_model_layout(day),
        'hour': _history_to_model_layout(hour),
        'target': target[:, :, target_channel].T[None, ...],
        'sim': similarity[None, ...],
        'sim_target': similarity_target[None, ...],
    }


def _stack_samples(samples):
    return {
        key: np.concatenate([sample[key] for sample in samples], axis=0)
        for key in samples[0]
    }


def _split_samples(samples, train_ratio, val_ratio):
    train_end = int(len(samples) * train_ratio)
    val_end = int(len(samples) * (train_ratio + val_ratio))
    if train_end == 0 or val_end == train_end or val_end == len(samples):
        raise ValueError('train/validation/test splits must all contain samples')

    return (
        _stack_samples(samples[:train_end]),
        _stack_samples(samples[train_end:val_end]),
        _stack_samples(samples[val_end:]),
    )


def _create_input_scalers(train_split, target_channel):
    train_history = np.concatenate(
        [train_split[key] for key in HISTORY_KEYS],
        axis=-1,
    )
    feature_axes = (0, 1, 3)
    input_scaler = StandardScaler(
        mean=train_history.mean(axis=feature_axes, keepdims=True),
        std=train_history.std(axis=feature_axes, keepdims=True),
    )
    target_scaler = StandardScaler(
        mean=float(input_scaler.mean[0, 0, target_channel, 0]),
        std=float(input_scaler.std[0, 0, target_channel, 0]),
    )
    return input_scaler, target_scaler


def _normalize_splits(splits, target_channel):
    train_split, val_split, test_split = splits
    input_scaler, target_scaler = _create_input_scalers(
        train_split,
        target_channel,
    )
    similarity_scaler = StandardScaler(
        mean=train_split['sim'].mean(),
        std=train_split['sim'].std(),
    )

    for split in splits:
        for key in HISTORY_KEYS:
            split[key] = input_scaler.transform(split[key])
        split['sim'] = similarity_scaler.transform(split['sim'])

    train_split['sim_target'] = similarity_scaler.transform(
        train_split['sim_target']
    )
    scalers = {
        'input': input_scaler,
        'target': target_scaler,
        'sim': similarity_scaler,
    }
    return scalers


def _similarity_threshold(train_similarity, threshold_multiplier):
    by_node = train_similarity.transpose(0, 2, 1).reshape(
        -1,
        train_similarity.shape[1],
    )
    q1, q3 = np.percentile(by_node, [25, 75], axis=0)
    return (q3 + threshold_multiplier * (q3 - q1))[:, None]


def _load_signal_data(filename, data_key, input_dim, target_channel):
    with np.load(filename) as data_file:
        if data_key not in data_file:
            raise KeyError(f'{data_key!r} not found in {filename}')
        raw_data = data_file[data_key]

    if raw_data.ndim != 3:
        raise ValueError('dataset array must have shape (time, nodes, features)')
    if not 0 < input_dim <= raw_data.shape[-1]:
        raise ValueError('input_dim exceeds the number of dataset features')
    if not 0 <= target_channel < input_dim:
        raise ValueError('target_channel must refer to an input feature')
    return raw_data[..., :input_dim]


def read_and_generate_dataset(
    graph_signal_matrix_filename,
    num_of_weeks,
    num_of_days,
    num_of_hours,
    num_for_predict,
    input_dim,
    train_ratio,
    val_ratio,
    points_per_hour=12,
    similarity_window=6,
    threshold_multiplier=3.0,
    data_key='data',
    target_channel=0,
):
    """Load raw signals and return normalized chronological data splits."""
    data_sequence = _load_signal_data(
        graph_signal_matrix_filename,
        data_key,
        input_dim,
        target_channel,
    )
    print('Prepare data')

    samples = []
    for label_start in range(data_sequence.shape[0]):
        sample = get_sample_indices(
            data_sequence=data_sequence,
            num_of_weeks=num_of_weeks,
            num_of_days=num_of_days,
            num_of_hours=num_of_hours,
            label_start_idx=label_start,
            num_for_predict=num_for_predict,
            points_per_hour=points_per_hour,
            similarity_window=similarity_window,
        )
        if sample is not None:
            samples.append(_format_sample(sample, target_channel))

    if not samples:
        raise ValueError('dataset is too short for the configured history windows')

    train_split, val_split, test_split = _split_samples(
        samples,
        train_ratio,
        val_ratio,
    )
    splits = (train_split, val_split, test_split)
    scalers = _normalize_splits(splits, target_channel)
    threshold = _similarity_threshold(
        train_split['sim'],
        threshold_multiplier,
    )

    return {
        'train': train_split,
        'val': {
            key: val_split[key]
            for key in (*HISTORY_KEYS, 'sim', 'target')
        },
        'test': {
            key: test_split[key]
            for key in (*HISTORY_KEYS, 'sim', 'target')
        },
        'threshold': {'sim': threshold},
        'stats': scalers,
    }


def _make_loader(data, batch_size, fields, shuffle):
    dataset = TensorDataset(*(
        torch.as_tensor(data[field])
        for field in fields
    ))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_final_dataset(final_dataset, batch_size):
    """Create train, validation, and test loaders in model input order."""
    common_fields = (*HISTORY_KEYS, 'sim')
    train_loader = _make_loader(
        final_dataset['train'],
        batch_size,
        (*common_fields, 'sim_target', 'target'),
        shuffle=True,
    )
    val_loader = _make_loader(
        final_dataset['val'],
        batch_size,
        (*common_fields, 'target'),
        shuffle=False,
    )
    test_loader = _make_loader(
        final_dataset['test'],
        batch_size,
        (*common_fields, 'target'),
        shuffle=False,
    )
    return train_loader, val_loader, test_loader, final_dataset['stats']
