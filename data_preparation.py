import numpy as np
import pandas as pd
import glob
import os
import pygeohash as gh
from IPython.display import IFrame
import osmnx as ox
import folium
from scipy.spatial import KDTree
import utils
import matplotlib.pyplot as plt


pd.options.mode.chained_assignment = None


class DataPreparation:

    def __init__(self, data_path):
        self.data_path = data_path
        self.timestamp_format = '%Y-%m-%d %H:%M:%S'
        self.date_format = '%Y-%m-%d'
        self.data = pd.DataFrame()
        self.daily_data = pd.DataFrame()

    def load_data(self):
        try:
            data = pd.read_csv(os.path.join(self.data_path, 'data.csv'), header=0)
            return data
        except FileNotFoundError as e:
            print(e)

    def feature_eng(self, df):
        print('Feature engineering...')

        # time-features engineering:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df['date'] = df['datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df['date'] = pd.to_datetime(df['date'])

        df['year'] = df['datetime'].apply(lambda x: x.year)
        df['month'] = df['datetime'].apply(lambda x: x.month)
        df['day'] = df['datetime'].apply(lambda x: x.dayofweek)
        df['hour'] = df['datetime'].apply(lambda x: x.hour)

        categorical_columns = ['year', 'month', 'day', 'season', 'holiday', 'workingday', 'weather']
        for col in categorical_columns:
            df[col] = df[col].astype("category")

        # daily aggregation:
        daily_cols = ['date', 'year', 'month', 'day', 'season', 'holiday', 'workingday']

        for col in ['temp', 'atemp', 'humidity', 'windspeed']:
            agg_col_name = f'avg_{col}'
            df[agg_col_name] = df.groupby('date')[col].transform('mean')
            daily_cols.append(agg_col_name)

        for col in ['casual', 'registered', 'count']:
            agg_col_name = f'tot_{col}'
            df[agg_col_name] = df.groupby('date')[col].transform('sum')
            daily_cols.append(agg_col_name)

        daily_df = df.loc[:, daily_cols].drop_duplicates()

        daily_df = daily_df.sort_values('date')
        daily_df['tot_count_avg_prev1days'] = daily_df['tot_count'].shift(1).rolling(window=1).mean()
        daily_df['tot_count_avg_prev3days'] = daily_df['tot_count'].shift(1).rolling(window=3).mean()
        daily_df['tot_count_avg_prev7days'] = daily_df['tot_count'].shift(1).rolling(window=7).mean()
        daily_df['tot_count_avg_prev30days'] = daily_df['tot_count'].shift(1).rolling(window=30).mean()
        daily_df = daily_df.dropna()
        daily_df = daily_df.reset_index(drop=True)

        return df, daily_df

    def data_analysis(self):
        pass
