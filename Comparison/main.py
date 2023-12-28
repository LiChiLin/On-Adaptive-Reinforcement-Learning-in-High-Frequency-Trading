import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from comparison_agent import Trading_Agent


def determine_actions(prices, interval=120):
    actions = []
    time_steps = 24
    for i in range(time_steps, len(prices), 120):
        if i < interval:
            # Not enough data to compare for these entries
            actions.append(0)  # or -1 if you prefer to default to sell
        else:
            # Buy if price has gone up since 120 records ago, else sell
            actions.append(1 if prices[i] >= prices[i - interval] else -1)
    return actions

test_data = pd.read_csv('test.csv')
simulate_data = copy.deepcopy(test_data)
simulate_data = simulate_data.drop(['Unnamed: 0'], axis=1)  # Drop unwanted columns



elastic_net_y_pred = pd.read_csv('Comparison/elastic_net_pred.csv')
elastic_net_y_pred = elastic_net_y_pred.drop(columns=['Unnamed: 0'])
elastic_net_y_pred.columns = ['Price'] if len(elastic_net_y_pred.columns) == 1 else ['price' if i == 1 else col for i, col in enumerate(elastic_net_y_pred.columns)]

elastic_net_trading_actions = determine_actions(elastic_net_y_pred['Price'])
trading_agent = Trading_Agent(simulate_data, 'elastic_net', 'SONY') # 2 min 15 secs later start trading
trading_agent.simulate_trades(elastic_net_trading_actions)


xg_reg_y_pred = pd.read_csv('Comparison/xg_reg_pred.csv')
xg_reg_y_pred = xg_reg_y_pred.drop(columns=['Unnamed: 0'])
xg_reg_y_pred.columns = ['Price'] if len(xg_reg_y_pred.columns) == 1 else ['price' if i == 1 else col for i, col in enumerate(xg_reg_y_pred.columns)]

xg_reg_trading_actions = determine_actions(xg_reg_y_pred['Price'])
trading_agent = Trading_Agent(simulate_data, 'xg_reg', 'SONY') # 2 min 15 secs later start trading
trading_agent.simulate_trades(xg_reg_trading_actions)

lgbm_y_pred = pd.read_csv('Comparison/lgbm_pred.csv')
lgbm_y_pred = lgbm_y_pred.drop(columns=['Unnamed: 0'])
lgbm_y_pred.columns = ['Price'] if len(lgbm_y_pred.columns) == 1 else ['price' if i == 1 else col for i, col in enumerate(lgbm_y_pred.columns)]

lgbm_trading_actions = determine_actions(lgbm_y_pred['Price'])
trading_agent = Trading_Agent(simulate_data, 'lgbm', 'SONY') # 2 min 15 secs later start trading
trading_agent.simulate_trades(lgbm_trading_actions)

# Buy and Hold Strategy
trading_agent = Trading_Agent(simulate_data, 'buy_and_hold', 'SONY') # 2 min 15 secs later start trading
trading_agent.buy_and_hold_strat()


# Graph
df_buy_and_hold = pd.read_csv('/Users/jaden/Desktop/BU MSMFT/MF703_Programming_for_Finance/Final_project/Comparison/buy_and_hold_SONY_capital.csv')
df_elastic_net = pd.read_csv('/Users/jaden/Desktop/BU MSMFT/MF703_Programming_for_Finance/Final_project/Comparison/elastic_net_SONY_capital.csv')
df_xgb = pd.read_csv('/Users/jaden/Desktop/BU MSMFT/MF703_Programming_for_Finance/Final_project/Comparison/xg_reg_SONY_capital.csv')
df_lgbm = pd.read_csv('/Users/jaden/Desktop/BU MSMFT/MF703_Programming_for_Finance/Final_project/Comparison/lgbm_SONY_capital.csv')
df_gru_dqn = pd.read_csv('/Users/jaden/Desktop/BU MSMFT/MF703_Programming_for_Finance/Final_project/SONY result/SONY_capital.csv')

df_buy_and_hold = df_buy_and_hold.drop(columns=['Unnamed: 0'])
df_elastic_net = df_elastic_net.drop(columns=['Unnamed: 0'])
df_xgb = df_xgb.drop(columns=['Unnamed: 0'])
df_lgbm = df_lgbm.drop(columns=['Unnamed: 0'])
df_gru_dqn = df_gru_dqn.drop(columns=['Unnamed: 0'])
df_gru_dqn.columns = ['Capital Balance']

# Plot Capital Balance over time for each model
plt.figure(figsize=(15, 8))
plt.grid(True)

# Plot for buy and hold strategy
plt.plot(df_buy_and_hold['DateTime'][:140000], df_gru_dqn['Capital Balance'][:140000], color='green', linewidth=2, label='Adaptive GRU-DQN', zorder = 5)
plt.plot(df_buy_and_hold['DateTime'][:140000], df_buy_and_hold['Capital Balance'][:140000], label='Buy and Hold', color='skyblue', linewidth=1)
plt.plot(df_buy_and_hold['DateTime'][:140000], df_elastic_net['Capital Balance'][:140000], label='Elastic Net', color='steelblue', linewidth=1)
plt.plot(df_buy_and_hold['DateTime'][:140000], df_xgb['Capital Balance'][:140000], label='XGBoost', color='lightblue', linewidth=2)
plt.plot(df_buy_and_hold['DateTime'][:140000], df_lgbm['Capital Balance'][:140000], label='Light GBM', color='lightgrey', linewidth=1)


locator = mdates.AutoDateLocator(maxticks=10)
plt.gca().xaxis.set_major_locator(locator)
plt.title('SONY Capital Balance Trajectories', fontsize=16)
plt.xlabel('DateTime', fontsize=14)
plt.ylabel('Capital Balance', fontsize=14)
plt.ticklabel_format(style='plain', axis='y')  # Prevent scientific notation on y-axis
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
plt.legend(fontsize=12)
# Save the figure
plt.savefig('Comparison/Combined_Capital_Balance_Trend.png', dpi=300)

