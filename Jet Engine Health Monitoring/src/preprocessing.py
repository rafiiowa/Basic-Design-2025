import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_rul(df):
    max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    df = df.merge(max_cycles, on='unit_number', how='left')
    df['RUL'] = df['max_cycles'] - df['time_cycles']
    df = df.drop('max_cycles', axis=1)
    return df

def feature_engineering(df):
    df_features = df.copy()
    window_sizes = [5, 10, 15]
    for window in window_sizes:
        for sensor in range(1, 22):
            col = f'sensor_{sensor}'
            if col in df_features.columns:
                df_features[f'{col}_moving_avg_{window}'] = df_features.groupby('unit_number')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean())
    key_sensors = [2, 3, 4, 7, 11, 12, 15]
    for sensor in key_sensors:
        col = f'sensor_{sensor}'
        if col in df_features.columns:
            df_features[f'{col}_rate'] = df_features.groupby('unit_number')[col].transform(
                lambda x: x.diff().fillna(0))
    return df_features

def normalize_data(train_df, test_df=None):
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
    op_cols = [col for col in train_df.columns if col.startswith('op_setting_')]
    cols_to_normalize = sensor_cols + op_cols
    scaler = MinMaxScaler()
    train_df[cols_to_normalize] = scaler.fit_transform(train_df[cols_to_normalize])
    if test_df is not None:
        common_cols = list(set(cols_to_normalize).intersection(set(test_df.columns)))
        test_df[common_cols] = scaler.transform(test_df[common_cols])
        return train_df, test_df, scaler
    return train_df, scaler

def prepare_data_for_modeling(df, sequence_length=30):
    features = df.drop(['unit_number', 'time_cycles', 'RUL'], axis=1).columns
    X, y = [], []
    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit]
        for i in range(len(unit_data) - sequence_length + 1):
            X.append(unit_data[features].iloc[i:i+sequence_length].values)
            y.append(unit_data['RUL'].iloc[i+sequence_length-1])
    return np.array(X), np.array(y)