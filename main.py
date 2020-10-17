import os
import numpy as np
import pandas as pd
import utils
from models import RFRegressor, Naive, XGBoost, KNNRegressor
import warnings
import matplotlib.pyplot as plt
from data_preparation import DataPreparation
from sklearn.metrics import mean_absolute_error
warnings.simplefilter('ignore')

# Environment variables:
DATA_PATH = './Data'
COST = 10
R_PRICE = 14
C_PRICE = 18
R_REJECTION_COST = 1
C_REJECTION_COST = 0

if __name__ == '__main__':

    DP = DataPreparation(DATA_PATH)
    df = DP.load_data()
    df, df_daily = DP.feature_eng(df)


    knn_model = KNNRegressor(df_daily, 'tot_count')
    knn_pred_df = knn_model.nested_cross_validation(df_daily, 5)
    utils.dump_pickle(knn_pred_df, os.path.join(DATA_PATH, 'knn_pred_df.pkl'))

    base_model = Naive(df_daily, 'tot_count', window=7)
    base_pred = base_model.predict(df_daily)

    rf_model = RFRegressor(df_daily, 'tot_count')
    rf_pred_df = rf_model.nested_cross_validation(df_daily, 5)
    utils.dump_pickle(rf_pred_df, os.path.join(DATA_PATH, 'rf_pred_df.pkl'))
    rf_model.plot_feature_importance()

    xgb_model = XGBoost(df_daily, 'tot_count')
    xgb_pred_df = xgb_model.nested_cross_validation(df_daily, 5)
    utils.dump_pickle(xgb_pred_df, os.path.join(DATA_PATH, 'xgb_pred_df.pkl'))
    xgb_model.plot_feature_importance()

    pred_df = rf_pred_df.loc[:, ['date', 'tot_registered', 'tot_casual', 'tot_count']]
    pred_df['best_profit'] = pred_df.apply(lambda x: utils.get_profit(x['tot_count'], x['tot_registered'], x['tot_casual']), axis=1)
    pred_df['rf_pred'] = rf_pred_df['pred']
    pred_df['rf_profit'] = pred_df.apply(lambda x: utils.get_profit(x['rf_pred'], x['tot_registered'], x['tot_casual']), axis=1)
    pred_df['xgb_pred'] = xgb_pred_df['pred']
    pred_df['xgb_profit'] = pred_df.apply(lambda x: utils.get_profit(x['xgb_pred'], x['tot_registered'], x['tot_casual']), axis=1)
    pred_df['knn_pred'] = knn_pred_df['pred']
    pred_df['knn_profit'] = pred_df.apply(lambda x: utils.get_profit(x['knn_pred'], x['tot_registered'], x['tot_casual']), axis=1)
    pred_df['naive_pred'] = base_pred
    pred_df['naive_profit'] = pred_df.apply(lambda x: utils.get_profit(x['naive_pred'], x['tot_registered'], x['tot_casual']), axis=1)

    summary_dict = {}
    summary_dict['actual'] = [0, pred_df['tot_count'].sum(), pred_df['best_profit'].sum()]
    for model in ['naive', 'knn', 'rf', 'xgb']:
        mae = mean_absolute_error(pred_df['tot_count'], pred_df[f'{model}_pred'])
        total_profit = pred_df[f'{model}_profit'].sum()
        total_pred = pred_df[f'{model}_pred'].sum()
        summary_dict[model] = [mae, total_pred, total_profit]

    summary_df = pd.DataFrame.from_dict(summary_dict).T
    summary_df.columns = ['MAE', 'Total Count Predicted', 'Total Profit']

    for col in summary_df.columns:
        plt.grid()
        summary_df[col].plot(kind='bar')
        plt.title(f'{col}')
        plt.show()

    print('END')




