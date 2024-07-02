import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.callbacks import LearningRateScheduler


def cosine_decay_with_warm_restart(initial_learning_rate, global_step, cycle_length, learning_rate_min):
    """
    Cosine decay with warm restarts as described in SGDR: Stochastic Gradient Descent with Warm Restarts
    """
    current_epoch = np.floor(global_step / cycle_length)
    cycle_progress = global_step - (current_epoch * cycle_length)
    new_learning_rate = learning_rate_min + 0.5 * (initial_learning_rate - learning_rate_min) * (1 + np.cos(np.pi * cycle_progress / cycle_length))
    return new_learning_rate


def create_sgdr_scheduler(initial_learning_rate, cycle_length, learning_rate_min):
    def scheduler(epoch, lr):
        return cosine_decay_with_warm_restart(initial_learning_rate, epoch, cycle_length, learning_rate_min)
    return LearningRateScheduler(scheduler)


def load_data(save=False):
    files = os.listdir("files/")
    window_size = 3
    batch_data = []

    for file in tqdm(files):
        if file == ".gitignore":
            continue
        raw_df = pd.read_csv("files/" + file, delimiter=';')
        features = [
            'u1', 'u2', 'u3', 'u0', 'u12', 'u23', 'u31', 'u1m', 'u1p', 'u2m', 'u2p', 'u3m', 'u3p',
            'u0m', 'u0p', 'u12m', 'u12p', 'u23m', 'u23p', 'u31m', 'u31p', 'upsm', 'upsp', 'unsm',
            'unsp', 'uzsm', 'uzsp', 'u_u2', 'u_u0', 'u1_thd', 'u2_thd', 'u3_thd', 'u0_thd', 'u12_thd',
            'u23_thd', 'u31_thd', 'i1', 'i2', 'i3', 'i0', 'i1m', 'i1p', 'i2m', 'i2p', 'i3m', 'i3p', 'i0m',
            'i0p', 'ipsm', 'ipsp', 'insm', 'insp', 'izsm', 'izsp', 'i_u2', 'i_u0', 'i1_thd', 'i2_thd', 'i3_thd',
            'i0_thd', 'p1', 'q1', 's1', 'phi', 'l1_cosphi', 'l2_cosphi', 'l3_cosphi', 'cosphi'
        ]
        df = raw_df[features]

        # Create rolling windows
        rolling_windows = df.rolling(window=window_size)

        # Iterate through the rolling windows and create batches
        for batch in rolling_windows:
            if batch.shape != (window_size, len(features)):
                continue
            batch_data.append(batch.values)
    x_train = np.stack(batch_data, axis=0)
    np.save(f"training_data/x_train", x_train[:-10000])
    np.save(f"training_data/x_test", x_train[-10000:])


def prepare_data(data_raw):
    min_per_feature = np.min(data_raw, axis=(0, 1))  # Minimum value per feature
    max_per_feature = np.max(data_raw, axis=(0, 1))  # Maximum value per feature

    # Normalize each feature to [0, 1]
    data = (data_raw - min_per_feature) / (max_per_feature - min_per_feature)
    return np.nan_to_num(data, nan=0.0)