import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQN:
    def __init__(self, state_size, action_space=3, alpha=0.1, gamma=0.9, epsilon=1.0, min_epsilon=0.01, decay_rate=0.005):
        self.state_size = state_size
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.model = self._build_model()

    def choose_action(self, state):
        # ε-greedy strategy
        # action buy: 1, action hold: 0, and action sell: -1 
        if np.random.rand() <= self.epsilon:
            return np.random.choice([1, 0, -1])  # Choose randomly among -1, 0, 1
        else:
            q_values = self.model.predict(state)
            action_index = np.argmax(q_values[0])
            # Map the index 0, 1, 2 to actions -1, 0, 1
            action_map = {0: 1, 1: 0, 2: -1}
            return action_map[action_index]

    def _build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model


    def train(self, state, action, reward, next_state):
        print(self.model.predict(next_state))
        target = reward + 0.95 * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

        # Update epilison value
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_rate
