import numpy as np
import pandas as pd
import copy

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from preprocessing.combiner import Combiner
from preprocessing.feature_engineering import Feature_Engineering
from model.GRU import GRUFeatureExtractor
from model.RL import DQN
from Algo_Trading.Agent import Trading_Agent

"""
Pipeline: Combiner -> Feature Engineering -> GRU (Dimension Reduction) -> DQN -> Trading_Agent 
-> Simulation Results
"""
def calculate_sortino_ratio(returns):
    rf = 0.001 / np.sqrt(4680) 
    negative_returns = [r for r in returns if r < 0]
    if len(negative_returns) == 0:
        return 0
    downside_deviation = np.std(negative_returns)
    return np.mean(returns) - rf / downside_deviation if downside_deviation != 0 else 0

def main():
    # Read Data
    processor = Combiner('/Users/jaden/Desktop/BU MSMFT/MF703_Programming_for_Finance/Final_project/JPM')
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
    
    # Feature Engineering
    transformer = Feature_Engineering('train.csv')
    transformer.load_and_prepare_data(['TRADES_Volume', 'Unnamed: 0'])
    transformer.apply_transformations('MIDPOINT')
    transformer.save_transformed_data('train.csv')

    transformer = Feature_Engineering('test.csv')
    transformer.load_and_prepare_data(['TRADES_Volume', 'Unnamed: 0'])
    transformer.apply_transformations('MIDPOINT')
    transformer.save_transformed_data('test.csv')

    train_data = pd.read_csv('train.csv')
    train_data = train_data.drop(['Unnamed: 0', 'DateTime'], axis=1)  # Drop unwanted columns

    test_data = pd.read_csv('test.csv')
    simulate_data = copy.deepcopy(test_data)
    # test_data = test_data.drop(['Unnamed: 0', 'DateTime'], axis=1)  # Drop unwanted columns
    simulate_data = simulate_data.drop(['Unnamed: 0'], axis=1)  # Drop unwanted columns
    test_data = test_data.drop(['Unnamed: 0', 'DateTime'], axis=1)  # Drop unwanted columns


    # Normalize the train data except for TRADES_Volume
    trades_volume = train_data['TRADES_Volume']
    train_data = train_data.drop('TRADES_Volume', axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # train_data_scaled = scaler.fit_transform(train_data.drop('MIDPOINT_Close', axis=1))
    train_data_scaled = scaler.fit_transform(train_data)
    train_data_scaled = np.hstack((train_data_scaled, trades_volume.values.reshape(-1, 1)))

    # Normalize the train data except for TRADES_Volume
    trades_volume = test_data['TRADES_Volume']
    test_data = test_data.drop('TRADES_Volume', axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # test_data_scaled = scaler.fit_transform(test_data.drop('MIDPOINT_Close', axis=1))
    test_data_scaled = scaler.fit_transform(test_data)
    test_data_scaled = np.hstack((test_data_scaled, trades_volume.values.reshape(-1, 1)))
    
    # Initialize GRU
    # time_steps = train_data_scaled.shape[0]-1
    time_steps = 24
    feature_dim = train_data_scaled.shape[1]
    gru_feature_extractor = GRUFeatureExtractor(time_steps, feature_dim)

    # Initialize DQN
    dqn_agent = DQN(state_size=16)

    # Train RL Model (For 1 month)
    # Initialize Setting
    time_steps = 24  # 120 seconds assuming 5-second intervals
    last_volatility = 0
    sortino_threshold = 1

    # Train RL Model (For 1 month)
    for i in range(time_steps, len(train_data_scaled), 120):  # 10 steps for 10 minutes prediction
        # Extract features with GRU (current_state)
        current_data = train_data_scaled[i - time_steps:i]
        current_data_reshaped = current_data.reshape((1, time_steps, feature_dim))
        current_features = gru_feature_extractor.extract_features(current_data_reshaped)

        # Extract features with GRU (next_state)
        next_data = train_data_scaled[i - time_steps+1:i+1]
        next_data_reshaped = next_data.reshape((1, time_steps, feature_dim))
        next_features = gru_feature_extractor.extract_features(next_data_reshaped)

        # Flatten the features for DQN
        flattened_current_features = current_features.flatten().reshape(1, -1)
        flattened_next_features = next_features.flatten().reshape(1, -1)

        returns = [((train_data['MIDPOINT_Close'][j] - train_data['MIDPOINT_Close'][j - 1]) / train_data['MIDPOINT_Close'][j - 1]) for j in range(1+ i - time_steps, i)]
        current_volatility = np.std(returns)
        sortino_ratio = calculate_sortino_ratio(returns)
        # Dynamically adjust sortino threshold
        if i != time_steps:
            if current_volatility >= last_volatility:
                sortino_threshold = sortino_threshold * 1.2
            else:
                sortino_threshold = sortino_threshold * 0.8
        
        if sortino_ratio < sortino_threshold : sortino_action = 2 # sell
        elif sortino_threshold <= sortino_ratio : sortino_action = 0 # buy
        dqn_agent.train(state = flattened_current_features, action = sortino_action, reward = abs(sortino_ratio), next_state= flattened_next_features)
        
        last_volatility = current_volatility

    # Reshape data for GRU input
    time_steps = 24  # 120 seconds assuming 5-second intervals
    feature_dim = test_data_scaled.shape[1]
    # gru_feature_extractor = GRUFeatureExtractor(time_steps, feature_dim)
    total_action = list()
    last_volatility = 0
    sortino_threshold = 1

    for i in range(time_steps, len(test_data_scaled), 120):  # 10 steps for 10 minutes prediction
        # Extract features with GRU (current_state)
        current_data = test_data_scaled[i - time_steps:i]
        current_data_reshaped = current_data.reshape((1, time_steps, feature_dim))
        current_features = gru_feature_extractor.extract_features(current_data_reshaped)

        # Extract features with GRU (next_state)
        next_data = test_data_scaled[i + 1 - time_steps:i + 1]
        next_data_reshaped = next_data.reshape((1, time_steps, feature_dim))
        next_features = gru_feature_extractor.extract_features(next_data_reshaped)

        # Flatten the features for DQN
        flattened_current_features = current_features.flatten().reshape(1, -1)
        flattened_next_features = next_features.flatten().reshape(1, -1)

        # DQN chooses an action
        returns = [((test_data['MIDPOINT_Close'][j] - test_data['MIDPOINT_Close'][j - 1]) / test_data['MIDPOINT_Close'][j - 1]) for j in range(1 + i - time_steps, i)]
        current_volatility = np.std(returns)
        sortino_ratio = calculate_sortino_ratio(returns)
        # Dynamically adjust sortino threshold
        if i != time_steps:
            if current_volatility >= last_volatility:
                sortino_threshold = sortino_threshold * 1.2
            else:
                sortino_threshold = sortino_threshold * 0.8
        
        if sortino_ratio < sortino_threshold : sortino_action = 2 # sell
        elif sortino_threshold <= sortino_ratio : sortino_action = 0 # buy
        dqn_agent.train(state = flattened_current_features, action = sortino_action, reward = abs(sortino_ratio), next_state= flattened_next_features)
        action = dqn_agent.choose_action(flattened_next_features)
        
        total_action.append(action)
        last_volatility = current_volatility

    print(total_action)

    # Initialize Trading Agent
    simulate_data = simulate_data.iloc[24:,:].reset_index(drop=True)
    trading_agent = Trading_Agent(simulate_data) # 2 min 15 secs later start trading

    # Final calculations and plot results
    trading_agent.simulate_trades(total_action)


if __name__ == '__main__':
    main()
