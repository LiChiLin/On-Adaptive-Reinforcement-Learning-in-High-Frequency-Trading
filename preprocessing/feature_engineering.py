import numpy as np
import pandas as pd
import copy
from scipy.stats import boxcox
from sklearn.preprocessing import power_transform

class Feature_Engineering:
    def __init__(self, input_path):
        self.input_path = input_path
        self.data = None
        self.transform_columns = None
        self.tmp_store = None

    def load_and_prepare_data(self, drop_columns):
        df = pd.read_csv(self.input_path)
        self.tmp_store = df['TRADES_Volume']
        df = df.drop(columns=drop_columns)
        self.data = copy.deepcopy(df)
        self.transform_columns = self.data.columns.drop(['DateTime'])

    def modified_boxcox(self, x, lmbda=0.5, shift=1):
        x_shifted = x + shift
        transformed = boxcox(x_shifted, lmbda=lmbda)
        return transformed

    def yeo_johnson(self, x):
        x_reshaped = x.values.reshape(-1, 1)
        transformed = power_transform(x_reshaped, method='yeo-johnson').flatten()
        return transformed

    def apply_transformations(self, indicator_column):
        transformations = {
            'log': lambda x: np.log(x + abs(x.min()) + 1),
            'sqrt': lambda x: np.sqrt(x + abs(x.min())),
            'inverse': lambda x: 1 / (x + abs(x.min()) + 1),
            'square': lambda x: (x + abs(x.min()))**2,
            'exponential': lambda x: np.exp(x + abs(x.min())),
            'modified_boxcox': self.modified_boxcox,
            'yeo_johnson': self.yeo_johnson
        }

        for col in self.transform_columns:
            for trans_name, trans_func in transformations.items():
                self.data[f'{trans_name}_{col}'] = trans_func(self.data[col])

            self.data[f'Rolling_Mean_{col}'] = self.data[col].rolling(window=5).mean()
            self.data[f'Rolling_Std_{col}'] = self.data[col].rolling(window=5).std()
            self.data[f'Lag_1_{col}'] = self.data[col].shift(1)

        self.add_tech_indicator(indicator_column)
        self.data.dropna(inplace=True)

    # Technical Indicator
    def add_sma(self, column, window=5):
        self.data[f'SMA_{window}_{column}'] = self.data[column].rolling(window=window).mean()

    def add_ema(self, column, span=12):
        self.data[f'EMA_{span}_{column}'] = self.data[column].ewm(span=span, adjust=False).mean()

    def add_rsi(self, column, window=14):
        delta = self.data[column].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.data[f'RSI_{window}_{column}'] = 100 - (100 / (1 + rs))

    def add_macd(self, column, span_short=12, span_long=26, signal=9):
        short_ema = self.data[column].ewm(span=span_short, adjust=False).mean()
        long_ema = self.data[column].ewm(span=span_long, adjust=False).mean()
        self.data[f'MACD_{column}'] = short_ema - long_ema
        self.data[f'MACD_Signal_{column}'] = self.data[f'MACD_{column}'].ewm(span=signal, adjust=False).mean()

    def add_bollinger_bands(self, column, window=20, no_of_std=2):
        self.data[f'Bollinger_Middle_{column}'] = self.data[column].rolling(window=window).mean()
        self.data[f'Bollinger_Upper_{column}'] = self.data[f'Bollinger_Middle_{column}'] + (self.data[column].rolling(window=window).std() * no_of_std)
        self.data[f'Bollinger_Lower_{column}'] = self.data[f'Bollinger_Middle_{column}'] - (self.data[column].rolling(window=window).std() * no_of_std)

    def add_atr(self, high, low, close, window=14):
        hl = self.data[high] - self.data[low]
        hc = abs(self.data[high] - self.data[close].shift())
        lc = abs(self.data[low] - self.data[close].shift())
        tr = pd.DataFrame({'HL': hl, 'HC': hc, 'LC': lc}).max(axis=1)
        self.data['ATR'] = tr.rolling(window=window).mean()

    def add_stochastic_oscillator(self, high, low, close, window=14, smooth_k=3, smooth_d=3):
        l14 = self.data[low].rolling(window=window).min()
        h14 = self.data[high].rolling(window=window).max()
        self.data['Stoch_K'] = ((self.data[close] - l14) / (h14 - l14)) * 100
        self.data['Stoch_D'] = self.data['Stoch_K'].rolling(smooth_k).mean()

    def add_obv(self, volume, close):
        self.data['OBV'] = (np.sign(self.data[close].diff()) * self.tmp_store).fillna(0).cumsum()

    def add_tech_indicator(self, indicator_column):
        self.add_sma(f'{indicator_column}_Close', window=10)
        self.add_ema(f'{indicator_column}_Close', span=12)
        self.add_rsi(f'{indicator_column}_Close', window=14)
        self.add_macd(f'{indicator_column}_Close')
        self.add_bollinger_bands(f'{indicator_column}_Close')
        self.add_atr(f'{indicator_column}_High', f'{indicator_column}_Low', f'{indicator_column}_Close')
        self.add_stochastic_oscillator(f'{indicator_column}_High', f'{indicator_column}_Low', f'{indicator_column}_Close')
        self.add_obv('TRADES_Volume', f'{indicator_column}_Close')

    def save_transformed_data(self, output_path):
        self.data['TRADES_Volume'] = self.tmp_store
        self.data.reset_index(inplace=True)
        self.data = self.data.drop(columns=['index'])
        print(self.data)
        self.data.to_csv(output_path)

    def get_nan_counts(self):
        self.data.isna().sum().to_csv('tmp5.csv')
        return self.data.isna().sum()
