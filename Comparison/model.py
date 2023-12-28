import numpy as np
import pandas as pd
import copy
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf

from combiner import Combiner
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# Read Data
processor = Combiner('/Users/jaden/Desktop/BU MSMFT/MF703_Programming_for_Finance/Final_project/SONY')
train_start_date = datetime(2023, 7, 1)
train_end_date = datetime(2023, 8, 31)
data = processor.process_year_data(train_start_date, train_end_date)
data = data.drop(columns= ['ASK_Volume', 'MIDPOINT_Volume', 'BID_Volume', 'BID_ASK_Volume', 'IV_Volume']) # Volatility contain many nan
data = data.drop_duplicates(keep='last').reset_index(drop=True)
print(data)
data.to_csv('train.csv')

# Test Date
test_start_date = datetime(2023, 9, 1)
test_end_date = datetime(2023, 10, 30)
data = processor.process_year_data(test_start_date, test_end_date)
data = data.drop(columns= ['ASK_Volume', 'MIDPOINT_Volume', 'BID_Volume', 'BID_ASK_Volume', 'IV_Volume']) # Volatility contain many nan
data = data.drop_duplicates(keep='last').reset_index(drop=True)
print(data)
data.to_csv('test.csv')

train_data = pd.read_csv('train.csv')
train_data = train_data.drop(['Unnamed: 0', 'DateTime'], axis=1)  # Drop unwanted columns

test_data = pd.read_csv('test.csv')
simulate_data = copy.deepcopy(test_data)
# test_data = test_data.drop(['Unnamed: 0', 'DateTime'], axis=1)  # Drop unwanted columns
simulate_data = simulate_data.drop(['Unnamed: 0'], axis=1)  # Drop unwanted columns
test_data = test_data.drop(['Unnamed: 0', 'DateTime'], axis=1)  # Drop unwanted columns


# Normalize the train data except for TRADES_Volume
y_train = train_data['MIDPOINT_Close']
trades_volume = train_data['TRADES_Volume']
train_data = train_data.drop(['TRADES_Volume', 'MIDPOINT_Close'], axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
# train_data_scaled = scaler.fit_transform(train_data.drop('MIDPOINT_Close', axis=1))
train_data_scaled = scaler.fit_transform(train_data)
train_data_scaled = np.hstack((train_data_scaled, trades_volume.values.reshape(-1, 1)))
X_train = copy.deepcopy(train_data_scaled)

# Normalize the train data except for TRADES_Volume
trades_volume = test_data['TRADES_Volume']
test_data = test_data.drop(['TRADES_Volume', 'MIDPOINT_Close'], axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
# test_data_scaled = scaler.fit_transform(test_data.drop('MIDPOINT_Close', axis=1))
test_data_scaled = scaler.fit_transform(test_data)
test_data_scaled = np.hstack((test_data_scaled, trades_volume.values.reshape(-1, 1)))
X_test = copy.deepcopy(test_data_scaled)

# Elastic Net
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)
elastic_net_y_pred = elastic_net.predict(X_test)
pd.DataFrame(elastic_net_y_pred).to_csv('Comparison/elastic_net_pred.csv')

# # Random Forest
# random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
# random_forest.fit(X_train, y_train)
# random_forest_y_pred = random_forest.predict(X_test)
# pd.DataFrame(random_forest_y_pred).to_csv('Comparison/random_forest_pred.csv')

# XGBoost
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 100, seed = 42)
xg_reg.fit(X_train, y_train)
xg_reg_y_pred = xg_reg.predict(X_test)
pd.DataFrame(xg_reg_y_pred).to_csv('Comparison/xg_reg_pred.csv')

# Lightgbm
lgbm_reg = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgbm_reg.fit(X_train, y_train)
lgbm_y_pred = lgbm_reg.predict(X_test)
pd.DataFrame(lgbm_y_pred).to_csv('Comparison/lgbm_pred.csv')


# # LSTM
# X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
# model.add(LSTM(units=50))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
# LSTM_y_pred = model.predict(X_test)
# pd.DataFrame(LSTM_y_pred).to_csv('Comparison/LSTM_reg_pred.csv')


