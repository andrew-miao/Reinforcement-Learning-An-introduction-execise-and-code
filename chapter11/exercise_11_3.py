import numpy as np
import matplotlib.pyplot as plt

DASHED = 0
SOLID = 1
LOWER = 6
GAMMA = 0.99
LENGTH = 1000
DIM = 8

class Baird_MDP:
    def __init__(self):
        feature_matrix = [[2, 0, 0, 0, 0, 0, 0, 1],
                          [0, 2, 0, 0, 0, 0, 0, 1],
                          [0, 0, 2, 0, 0, 0, 0, 1],
                          [0, 0, 0, 2, 0, 0, 0, 1],
                          [0, 0, 0, 0, 2, 0, 0, 1],
                          [0, 0, 0, 0, 0, 2, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 2]]
        self.state_feature_matrix = np.array(feature_matrix, dtype=np.float64)
        self.n_state = self.state_feature_matrix.shape[0]
        self.lower = LOWER
        self.gamma = GAMMA
        self.action_space = np.array([DASHED, SOLID])
    
    def transit(self, action):
        if action == DASHED:
            next_state_id = np.random.randint(0, high=self.lower)
        else:
            next_state_id = self.lower
        return self.state_feature_matrix[next_state_id]
    
    def reward(self):
        return 0

def behaviour(MDP):
    action = np.random.choice(MDP.action_space, p=[6/7, 1/7])
    return action

def learn_weights(MDP, behaviour, lr=0.01, max_step=LENGTH):
    weights_record = []
    w = [1, 1, 1, 1, 1, 1, 10, 1]
    w = np.array(w, dtype=np.float64)
    state_id = np.random.randint(0, high=MDP.n_state)
    state = MDP.state_feature_matrix[state_id]
    for _ in range(max_step):
        weights_record.append(np.copy(w))
        action = behaviour(MDP)
        next_state = MDP.transit(action)
        delta = MDP.reward() + MDP.gamma * np.dot(next_state, w) - np.dot(state, w)
        rho = 0
        if action == SOLID:
            rho = 7
        w += lr * rho * delta * state
        state = next_state
    return np.array(weights_record)

def learn_weights_DP(MDP, lr=0.01, max_sweep=LENGTH):
    weigths_record = []
    w = [1, 1, 1, 1, 1, 1, 10, 1]
    w = np.array(w, dtype=np.float64)
    for _ in range(max_sweep):
        weigths_record.append(np.copy(w))
        gradient = 0
        for state in range(MDP.n_state):
            gradient += (MDP.reward() + MDP.gamma * np.dot(MDP.state_feature_matrix[LOWER], w) - np.dot(MDP.state_feature_matrix[state], w)) * MDP.state_feature_matrix[state]
        gradient = -gradient / MDP.n_state
        w -= lr * gradient
    return np.array(weigths_record)

def main():
    MDP = Baird_MDP()
    weights_record = learn_weights(MDP, behaviour)
    weights_record_dp = learn_weights_DP(MDP)
    x = np.arange(LENGTH)
    colors = ['black', 'red', 'darksalmon', 'tan',
              'darkgreen', 'blue', 'darkorchid', 'goldenrod']
    fig, (ax0, ax1) = plt.subplots(1, 2)
    axs = (ax0, ax1)
    for i, ax in enumerate(axs):
        for d in range(DIM):
            if i == 0:
                y = weights_record[:, d]
            else:
                y = weights_record_dp[:, d]
            ax.plot(x, y, label='w'+str(d + 1), color=colors[d])
    plt.legend(loc='upper left')
    print(weights_record_dp[LENGTH-1][6])
    plt.show()

if __name__ == '__main__':
    main()
    