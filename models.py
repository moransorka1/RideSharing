from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hyperopt import fmin, tpe, hp, Trials
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Environment variables:
DATA_PATH = './Data'
COST = 10
R_PRICE = 14
C_PRICE = 18
R_REJECTION_COST = 1
C_REJECTION_COST = 0


class BaseModel:
    def __init__(self, data, target, cv=5):
        self.name = ''
        self.data = data
        self.train, self.test = None, None
        self.target = target
        self.model = None
        self.cv = cv

    def set_train(self, data):
        self.train = data

    def set_test(self, data):
        self.test = data

    def get_x(self, data):
        columns_to_drop = ['datetime', 'date', 'tot_casual', 'tot_registered', 'tot_count']
        x = data.copy()
        for col in columns_to_drop:
            try:
                x = x.drop(columns=[col])
            except:
                continue
        return x

    def get_y(self, data):
            return data[self.target]

    def nested_cross_validation(self, data, folds):
        data_len = len(data)
        block_size = int(data_len / folds)
        pred_df = pd.DataFrame()
        for i in range(1, folds):
            try:
                # cut data sets for gridsearch:
                train_ind = i * block_size
                val_ind = train_ind + block_size
                train = data.iloc[:train_ind]
                val = data.iloc[train_ind: val_ind]
                # get best params:
                best_params, trails = self.gridsearch(train, val)
                # fit model with train+val data and predict next block:
                train_val = data[:val_ind]
                if i == folds-1:
                    test_ind = data_len
                else:
                    test_ind = val_ind + block_size
                test = data.iloc[val_ind: test_ind]

                self.fit_model(best_params, train_val, test)
                test['pred'] = self.predict(test)
                pred_df = pd.concat([pred_df, test])
            except BaseException as e:
                print(e)

        return pred_df

    def gridsearch(self, train, val):
        self.set_train(train)
        self.set_test(val)

        fspace = self.get_fspace()
        trials = Trials()
        best = fmin(fn=self.fit_model, space=fspace, algo=tpe.suggest, max_evals=500, trials=trials)
        print("best params:")
        best_params = self.get_params_from_space(best)
        print(best_params)
        # self.save_best_params(best_params)
        return best_params, trials

    def fit_model(self, params, train=None, test=None):
        # self.train, self.test should be initiated before calling this method if train and test is not given as input
        self.init_model(params)
        if train is not None:
            self.train = train
            self.test = test

        X_train = self.get_x(self.train)
        y_train = self.get_y(self.train)
        X_test = self.get_x(self.test)
        self.input_features = X_train.columns

        self.model.fit(X_train, y_train)

        res = self.test.copy()
        # res['best_profit'] = res.apply(lambda x: utils.get_profit(x['tot_count'], x['tot_registered'], x['tot_casual']), axis=1)

        res['pred'] = self.model.predict(X_test)

        if self.target == 'tot_count':
            res['pred_profit'] = res.apply(
                lambda x: utils.get_profit(x['pred'], x['tot_registered'], x['tot_casual']), axis=1)

        profit = res['pred_profit'].mean()

        return -profit

    def init_model(self, params):
        pass

    def predict(self, data):
        X = self.get_x(data)
        X = X.loc[:, self.input_features]
        pred = self.model.predict(X)
        return pred

    def get_fspace(self):
        params_space_dict = self.get_params_dict()
        fspace = {}
        for param, values in params_space_dict.items():
            fspace[param] = hp.choice(param, values)
        return fspace

    def get_params_from_space(self, best):
        best_params = {}
        params_space_dict = self.get_params_dict()
        for param, values in params_space_dict.items():
            best_params[param] = values[best[param]]
        return best_params

    def plot_feature_importance(self):
        try:
            features = self.input_features
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[:10]

            plt.title(f'{self.name} Feature Importances')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.show()
        except BaseException as e:
            print(f'Failed to plot feature importance: {e}')


class RFRegressor(BaseModel):
    def __init__(self, data, target):
        super().__init__(data, target)
        self.name = 'RandomForest Model'
        print(self.name)

    def init_model(self, params):
        self.model = RandomForestRegressor(**params)

    @staticmethod
    def get_params_dict():
        return {'n_estimators': [10, 20, 50, 100],
                'max_depth': [5, 10, 20, 30],
                'min_samples_split': [1., 2, 5, 10, 0.05],
                'min_samples_leaf': [1, 5, 10, 0.05],
                'max_features': ["auto", "sqrt", "log2", 10],
                'bootstrap': [True, False],
                'criterion': ['mse', 'mae']}


class XGBoost(BaseModel):
    def __init__(self, data, target):
        super().__init__(data, target)
        self.name = 'XGBoost Model'
        self.data = self.data_preparation(self.data)
        print(self.name)

    def data_preparation(self, data):
        categorical_columns = data.select_dtypes(include='category').columns
        data = pd.get_dummies(data, columns=categorical_columns)
        return data

    def get_x(self, data):
        columns_to_drop = ['datetime', 'date', 'tot_casual', 'tot_registered', 'tot_count']
        x = data.copy()
        x = self.data_preparation(x)
        for col in columns_to_drop:
            try:
                x = x.drop(columns=[col])
            except:
                continue
        return x

    def init_model(self, params):
        self.model = XGBRegressor(**params)

    @staticmethod
    def get_params_dict():
        return {'booster': ['gbtree', 'dart'],
                'eta': [0.003, 0.01, 0.1, 0.3, 0.5, 0.7],
                'min_child_weight': [1, 5, 10],
                'subsample': [0, 0.5, 1],
                'gamma': [0.5, 1, 1.5, 2, 5, 50, 100, 500],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [4, 6, 10, 13]}


class KNNRegressor(BaseModel):
    def __init__(self, data, target):
        super().__init__(data, target)
        self.name = 'KNN Model'
        self.data = self.data_preparation(self.data)
        print(self.name)
        self.scalers = {}

    def data_preparation(self, data):
        categorical_columns = data.select_dtypes(include='category').columns
        data = pd.get_dummies(data, columns=categorical_columns)
        return data

    def get_x(self, data):
        columns_to_drop = ['datetime', 'date', 'tot_casual', 'tot_registered', 'tot_count']
        x = data.copy()
        x = self.data_preparation(x)
        for col in columns_to_drop:
            try:
                x = x.drop(columns=[col])
            except:
                continue
        return x

    def fit_model(self, params, train=None, test=None):
        # self.train, self.test should be initiated before calling this method if train and test is not given as input
        self.init_model(params)
        if train is not None:
            self.train = train
            self.test = test

        X_train = self.get_x(self.train)
        y_train = self.get_y(self.train)
        X_test = self.get_x(self.test)
        self.input_features = X_train.columns

        X_train_scaled = pd.DataFrame()
        X_test_scaled = pd.DataFrame()
        for col in X_train.columns:
            self.scalers[col] = StandardScaler()
            self.scalers[col].fit(X_train[col].to_numpy().reshape(-1, 1))
            self.scalers[col].transform(X_train[col].to_numpy().reshape(-1, 1))
            X_train_scaled[col] = self.scalers[col].transform(X_train[col].to_numpy().reshape(-1, 1)).reshape(-1)
            X_test_scaled[col] = self.scalers[col].transform(X_test[col].to_numpy().reshape(-1, 1)).reshape(-1)

        self.model.fit(X_train_scaled, y_train)

        res = self.test.copy()
        # res['best_profit'] = res.apply(lambda x: utils.get_profit(x['tot_count'], x['tot_registered'], x['tot_casual']), axis=1)

        res['pred'] = self.model.predict(X_test_scaled)

        if self.target == 'tot_count':
            res['pred_profit'] = res.apply(
                lambda x: utils.get_profit(x['pred'], x['tot_registered'], x['tot_casual']), axis=1)

        profit = res['pred_profit'].mean()

        return -profit

    def init_model(self, params):
        self.model = KNeighborsRegressor(**params)

    @staticmethod
    def get_params_dict():
        return {'n_neighbors': [3, 5, 10, 20, 50],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [1, 2, 5, 10, 30, 50]}


class Naive(BaseModel):
    def __init__(self, data, target, window=7):
        super().__init__(data, target)
        self.name = 'Naive Model'
        self.window = window
        print(self.name)

    def predict(self, data):
        pred = data[self.target].shift(1).rolling(window=self.window).mean()
        return pred
