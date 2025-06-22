import tensorflow as tf
import numpy as np
import random
from collections import deque
import json
import os


class DQNAgent:
    def __init__(self, name, state_size, action_size):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01  #used to be 0.1
        self.epsilon_decay = 0.995  #used to be 0.99
        self.batch_size = 64
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(self.state_size,), activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    #save and load
    def save_state(self, filename):
        state = {
            "epsilon": self.epsilon,
        }
        with open(filename, "w") as f:
            json.dump(state, f)

    def load_state(self, filename):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                state = json.load(f)
                self.epsilon = state.get("epsilon", 1.0)

    def soft_update_target(self, tau=0.01):
        for target, main in zip(self.target_model.variables, self.model.variables):
            target.assign(tau * main + (1 - tau) * target)

    def update_target(self):
        τ = 0.01  # Small factor
        for target_param, param in zip(self.target_model.weights, self.model.weights):
            target_param.assign(τ * param + (1 - τ) * target_param)


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_values = self.model.predict(state[np.newaxis], verbose=0)[0]
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        print("Training...")
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state[np.newaxis], verbose=0)[0]
            if done:
                target[action] = reward
            else:
                future_q = np.amax(self.target_model.predict(next_state[np.newaxis], verbose=0)[0])
                target[action] = reward + self.gamma * future_q

            self.model.fit(state[np.newaxis], target[np.newaxis], verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
