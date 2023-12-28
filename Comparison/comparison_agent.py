import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

class Trading_Agent:
    def __init__(self, data, model_name, stock_name, initial_funds=1000000, transaction_cost=0.0005):
        self.data = data
        self.initial_funds = initial_funds
        self.capital_balance = initial_funds
        self.transaction_cost = transaction_cost
        self.position = 0
        self.funds_over_time = []
        self.model_name = model_name
        self.stock_name = stock_name

    def execute_trade(self, action, price, last_price):
        """
        Execute a trade based on the action and update the position and funds.
        action: -1 (sell), 0 (hold), 1 (buy)
        """
        returns = (price - last_price) / last_price
        cost = self.capital_balance * self.transaction_cost

        if action == 1: # Buy 
            if self.position == 0:
                self.position += 1
                self.capital_balance = self.capital_balance - cost
            elif self.position == -1:
                self.position += 1
                self.capital_balance = self.capital_balance * (1 - returns)
            elif self.position == 1:
                self.capital_balance = self.capital_balance * (1 + returns)
        elif action == -1: # Sell
            if self.position == 0:
                self.position -= 1
                self.capital_balance = self.capital_balance - cost
            elif self.position == 1:
                self.position -= 1
                self.capital_balance = self.capital_balance * (1 + returns)
            elif self.position == -1:
                self.capital_balance = self.capital_balance * (1 - returns)
        else: # Hold
            if self.position == 1:
                self.capital_balance = self.capital_balance * (1 + returns)
            elif self.position == -1:
                self.capital_balance = self.capital_balance * (1 - returns)

    def calculate_sharpe_ratio(self, returns):
        """ Calculate the Sharpe Ratio for the simulated trades. """
        if np.std(returns) == 0:
            return 0
        second_rf = (0.001 / np.sqrt(252)) / np.sqrt(4680)
        return (np.mean(returns) / np.std(returns)) * np.sqrt(4680) * np.sqrt(252)

    def calculate_max_drawdown(self, wealth_index):
        """ Calculate the Maximum Drawdown (MDD) for the simulated trades. """
        # Find the index of the first peak
        first_peak_index = np.argmax(wealth_index)

        # Calculate the traditional maximum drawdown
        traditional_max_drawdown = 0
        peak = wealth_index[0]
        for x in wealth_index:
            if x > peak:
                peak = x
            drawdown = (peak - x) / peak
            if drawdown > traditional_max_drawdown:
                traditional_max_drawdown = drawdown

        # Calculate the maximum drawdown before the first peak
        if first_peak_index == 0:
            before_peak_max_drawdown = 0
        else:
            trough_before_peak = np.min(wealth_index[:first_peak_index])
            before_peak_max_drawdown = (peak - trough_before_peak) / peak

        # Return the larger of the two drawdowns
        return max(traditional_max_drawdown, before_peak_max_drawdown)

    def calculate_total_returns(self):
        """ Calculate total percentage returns. """
        return (self.funds_over_time[-1] - self.initial_funds) / self.initial_funds * 100

    def calculate_volatility(self, returns):
        """ Calculate annualized volatility of returns. """
        return np.std(returns)

    def calculate_win_rate(self, returns):
        """ Calculate the win rate of trades. """
        wins = sum(np.array(returns) >= 0)
        return (wins / len(returns)) * 100

    def calculate_information_coefficient(self, benchmark_returns):
        """ Calculate the information coefficient. """
        returns = np.diff(self.funds_over_time) / self.funds_over_time[:-1]
        correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
        return correlation

    def calculate_information_ratio(self, returns, benchmark_returns):
        """ Calculate the information ratio. """
        return_difference = returns - benchmark_returns
        tracking_error = np.std(return_difference)
        information_ratio = np.mean(return_difference) / tracking_error
        return information_ratio

    def simulate_trades(self, actions):
        """ Simulate trades using the agent on the data. """
        # Initialize a list to hold datetime, capital balance, action, and midpoint close price
        self.actions_over_time = actions
        action_index = 0
        trading_records = []

        last_price = self.data['MIDPOINT_Close'][0]
        self.data = self.data.iloc[1:,:].reset_index(drop=True)
        for i, row in self.data.iterrows():
            current_datetime = row['DateTime']
            price = row['MIDPOINT_Close']
            # Execute trade at decision points (every 10 minutes)
            if i % 120 == 0:  # Assuming data is every 5 seconds, so 120 records per 10 minutes
                trading_action = self.actions_over_time[action_index]
                if trading_action == 1:
                    trading_action_symbol = 'Buy'
                elif trading_action == -1:
                    trading_action_symbol = 'Sell'
                else:
                    trading_action_symbol = 'Hold'
                action_index += 1
            else:
                trading_action = 0
                trading_action_symbol = 'N/A'

            self.execute_trade(trading_action, price, last_price)
            last_price = row['MIDPOINT_Close']
            # Update portfolio value even if there is no trade
            self.funds_over_time.append(self.capital_balance)

            # Record datetime, capital balance, action, and midpoint close price
            trading_records.append((current_datetime, self.capital_balance, trading_action_symbol, price))
        
        # Convert trading records to DataFrame for easier plotting
        trading_df = pd.DataFrame(trading_records, columns=['DateTime', 'Capital Balance', 'Action', 'MIDPOINT_Close'])

        # Calculate metrics
        total_returns = self.calculate_total_returns()
        funds_over_time=pd.DataFrame(self.funds_over_time)
        print(trading_df[['DateTime', 'Capital Balance']])
        trading_df[['DateTime', 'Capital Balance']].to_csv(f'Comparison/{self.model_name}_{self.stock_name}_capital.csv')
        returns= funds_over_time.pct_change(1)
        returns = returns.dropna()
        returns= np.array(returns)
        #fund_changes = np.diff(self.funds_over_time)
        #returns = np.divide(fund_changes, self.funds_over_time[:-1], out=np.zeros_like(fund_changes), where=self.funds_over_time[:-1] != 0)
        volatility = self.calculate_volatility(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        win_rate = self.calculate_win_rate(returns)
        max_drawdown = self.calculate_max_drawdown(np.array(self.funds_over_time))

        # Plot Capital Balance over time
        plt.figure(figsize=(12, 7))
        plt.grid()
        plt.plot(trading_df['DateTime'], trading_df['Capital Balance'], label='Capital Balance Over Time')
        # Set major locator to 6 ticks on the x-axis.
        locator = mdates.AutoDateLocator(maxticks=10)
        plt.gca().xaxis.set_major_locator(locator)
        plt.title('Capital Balance Trajectories')
        plt.xlabel('DateTime')
        plt.ylabel('Capital Balance')
        plt.ticklabel_format(style='plain', axis='y')  # Prevent scientific notation on y-axis
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
        plt.legend()
        plt.savefig(f'Comparison/{self.model_name}_{self.stock_name}_Trend.png')


        # Plot Actions over time
        action_mapping = {'Buy': 1, 'Hold': 0, 'Sell': -1, 'N/A': None}
        mapped_actions = [action_mapping[action] for action in trading_df['Action']]
        trading_df['Action'] = mapped_actions
        trading_df = trading_df.dropna()
        print(trading_df)
        plt.figure(figsize=(12, 7))
        plt.grid()
        plt.plot(trading_df['DateTime'], trading_df['Action'], label='Action', color='green')
        plt.scatter(trading_df['DateTime'], trading_df['Action'], color='red')
        locator = mdates.AutoDateLocator(maxticks=10)
        plt.gca().xaxis.set_major_locator(locator)
        plt.yticks(ticks=[-1, 0, 1], labels=['Sell', 'Hold', 'Buy'])
        plt.title('Actions Over Time')
        plt.xlabel('DateTime')
        plt.ylabel('Action')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'Comparison/{self.model_name}_{self.stock_name}_Action.png')

        # Print metrics
        metrics = {
            'Metric': [
                'Total Returns (%)',
                'Volatility',
                'Sharpe Ratio',
                'Win Rate (%)',
                'Maximum Drawdown (%)'
            ],
            'Value': [
                total_returns,
                volatility,
                sharpe_ratio,
                win_rate,
                max_drawdown * 100  # Convert to percentage
            ]
        }
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(f'Comparison/{self.model_name}_{self.stock_name}_result.csv')


    def buy_and_hold_strat(self):
        """ Simulate trades using the buy and hold strategy. """
        # Initialize a list to hold datetime, capital balance, and midpoint close price
        trading_records = []

        last_price = self.data['MIDPOINT_Close'][0]
        self.data = self.data.iloc[1:,:].reset_index(drop=True)

        # Buy once at the beginning
        self.execute_trade(1, self.data['MIDPOINT_Close'][0], last_price)
        last_price = self.data['MIDPOINT_Close'][0]

        for i, row in self.data.iterrows():
            self.funds_over_time.append(self.capital_balance)
            current_datetime = row['DateTime']
            price = row['MIDPOINT_Close']

            # Hold the position, no further buying or selling
            self.execute_trade(0, price, last_price)
            last_price = row['MIDPOINT_Close']

            # Record datetime, capital balance, and midpoint close price
            trading_records.append((current_datetime, self.capital_balance, price))
        
        # Convert trading records to DataFrame for easier plotting
        trading_df = pd.DataFrame(trading_records, columns=['DateTime', 'Capital Balance', 'MIDPOINT_Close'])

        # Calculate metrics
        total_returns = self.calculate_total_returns()
        funds_over_time=pd.DataFrame(self.funds_over_time)
        trading_df[['DateTime', 'Capital Balance']].to_csv(f'Comparison/{self.model_name}_{self.stock_name}_capital.csv')
        returns= funds_over_time.pct_change(1)
        returns = returns.dropna()
        returns= np.array(returns)
        #fund_changes = np.diff(self.funds_over_time)
        #returns = np.divide(fund_changes, self.funds_over_time[:-1], out=np.zeros_like(fund_changes), where=self.funds_over_time[:-1] != 0)
        volatility = self.calculate_volatility(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        win_rate = self.calculate_win_rate(returns)
        max_drawdown = self.calculate_max_drawdown(np.array(self.funds_over_time))

        # Plot Capital Balance over time
        plt.figure(figsize=(12, 7))
        plt.grid()
        plt.plot(trading_df['DateTime'], trading_df['Capital Balance'], label='Capital Balance Over Time')
        # Set major locator to 6 ticks on the x-axis.
        locator = mdates.AutoDateLocator(maxticks=10)
        plt.gca().xaxis.set_major_locator(locator)
        plt.title('Capital Balance Trajectories')
        plt.xlabel('DateTime')
        plt.ylabel('Capital Balance')
        plt.ticklabel_format(style='plain', axis='y')  # Prevent scientific notation on y-axis
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
        plt.legend()
        plt.savefig(f'Comparison/{self.model_name}_{self.stock_name}_Trend.png')

        # Print metrics
        metrics = {
            'Metric': [
                'Total Returns (%)',
                'Volatility',
                'Sharpe Ratio',
                'Win Rate (%)',
                'Maximum Drawdown (%)'
            ],
            'Value': [
                total_returns,
                volatility,
                sharpe_ratio,
                win_rate,
                max_drawdown * 100  # Convert to percentage
            ]
        }
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(f'Comparison/{self.model_name}_{self.stock_name}_result.csv')








