import numpy as np
import pandas as pd

from data_generation import DATA_VERSION_1, DATA_VERSION_2, FeatureSetDateRangeScope, FeatureSetDateScope

# Task 1

data_a = DATA_VERSION_1.get_date_range_data(FeatureSetDateRangeScope(DATA_VERSION_1.start_date, DATA_VERSION_1.end_date, 'A'))
data_c = DATA_VERSION_1.get_date_range_data(FeatureSetDateRangeScope(DATA_VERSION_1.start_date, DATA_VERSION_1.end_date, 'C'))

# Simple linear model, average across all features
predicted_c = np.repeat(data_a.values.mean(axis=1)[:, np.newaxis], len(data_c.columns), axis=1)

def compute_r_squared(predictions: np.ndarray, realizations: np.ndarray) -> np.ndarray:
    residual_sum_of_squares = np.sum((predictions - realizations) ** 2.0, axis=0)
    total_sum_of_squares = np.sum((predictions - predictions.mean(axis=0)) ** 2.0, axis=0)
    return 1 - residual_sum_of_squares / total_sum_of_squares

r_squared = compute_r_squared(predicted_c, data_c.values)

# Task 5

target_feature = 'A_f09'
sorted_values = np.array([])
sorted_timestamps = np.array([])

selection_threshold = 0.01

data = DATA_VERSION_2

for date in data.dates():
    print(f'Processing {date}...')
    data_a = data.get_date_data(FeatureSetDateScope(date, 'A'))
    values = data_a[target_feature]
    for value, timestamp in zip(values, data_a.index):
        insertion_index = np.searchsorted(sorted_values, value)
        sorted_values = np.insert(sorted_values, insertion_index, value)
        sorted_timestamps = np.insert(sorted_timestamps, insertion_index, timestamp.value)

num_to_select = int(len(sorted_timestamps) * selection_threshold)
timestamps_to_select = sorted_timestamps[-num_to_select:]

selected_chunks = []
for date in data.dates():
    data_a = data.get_date_data(FeatureSetDateScope(date, 'A'))
    selected_chunks.append(data_a.loc[data_a.index.isin(timestamps_to_select)])

selected = pd.concat(selected_chunks)

# Simple linear model, average across all features
predicted_c = np.repeat(selected.values.mean(axis=1)[:, np.newaxis], DATA_VERSION_2.feature_set_c.num_features, axis=1)
