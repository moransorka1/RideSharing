import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate
from ast import literal_eval
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Environment variables:
DATA_PATH = './Data'
COST = 10
R_PRICE = 14
C_PRICE = 18
R_REJECTION_COST = 1
C_REJECTION_COST = 0


def print_df_as_table(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))


def dump_pickle(obj, file_path):
    print(f"dump to pickle {file_path}")
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_path, print_=True):
    if print_:
        print(f"load from pickle {file_path}")
    with open(file_path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def split_train_test(data, train_prc):
    data_len = len(data)
    train_index = int(data_len * train_prc)

    train = data[:train_index]
    test = data[train_index:]

    return train, test


def get_profit(max_capacity, r_demand, c_demand):
    r_prc = r_demand / (r_demand + c_demand)
    operational_cost = max_capacity * COST
    penalty = 0

    if r_demand + c_demand > max_capacity:
        income = (max_capacity * r_prc) * R_PRICE + (max_capacity * (1 - r_prc)) * C_PRICE
        penalty = (r_demand - (max_capacity * r_prc)) * R_REJECTION_COST + (
                    c_demand - (max_capacity * (1 - r_prc))) * C_REJECTION_COST
    else:
        income = R_PRICE * r_demand + C_PRICE * c_demand

    profit = income - operational_cost - penalty
    return profit


def plot_moving_average(series, window, plot_actual=False, plot_intervals=False, scale=1, title_prefix=''):
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(12, 8))
    plt.title(f'{title_prefix} Moving average for {series.name}\n window size = {window}')
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='bounds: +- std')
        plt.plot(lower_bound, 'r--')

    if plot_actual:
        plt.plot(series[window:], label='Actual', alpha=0.5, marker='.')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()