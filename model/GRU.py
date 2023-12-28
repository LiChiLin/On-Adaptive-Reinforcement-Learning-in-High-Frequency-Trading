from tensorflow import keras
from keras.layers import GRU, Dropout, Dense
from keras.models import Sequential


class GRUFeatureExtractor:
    def __init__(self, input_size, feature_dim):
        self.input_size = input_size
        self.feature_dim = feature_dim  # input_shape e.g., (time_steps, num_features)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(GRU(64, input_shape=(self.input_size, self.feature_dim), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='linear'))

        model.compile(optimizer='adam', loss='mse')
        return model

    def extract_features(self, data):
        return self.model.predict(data)
