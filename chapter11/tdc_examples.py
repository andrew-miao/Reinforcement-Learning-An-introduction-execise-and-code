from matplotlib import widgets
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
        self.state_feature = np.array(feature_matrix, dtype=np.float64)
        self.n_state = self.state_feature.shape[0]
        self.lower = LOWER
        self.gamma = GAMMA
        self.action_space = np.array([DASHED, SOLID])
    
    def transit(self, action):
        if action == DASHED:
            next_state_id = np.random.randint(0, high=self.lower)
        else:
            next_state_id = self.lower
        return self.state_feature[next_state_id]
    
    def reward(self):
        return 0

def behaviour(MDP):
    action = np.random.choice(MDP.action_space, p=[6/7, 1/7])
    return action

def compute_PBE(MDP, weights):
    gamma = MDP.gamma
    D_pi = [(1 - gamma)/7, (1 - gamma)/7, (1 - gamma)/7, (1 - gamma)/7,
            (1 - gamma)/7, (1 - gamma)/7, (1 - gamma)/7 + gamma]
    D_pi = np.diag(D_pi)
    state = MDP.state_feature
    v_w = np.matmul(state, weights)
    delta = MDP.reward() + gamma * np.dot(state[LOWER].T, weights) - v_w
    state_D_delta = np.matmul(np.matmul(state.T, D_pi), delta)
    state_D_state = np.matmul(np.matmul(state.T, D_pi), state)
    PBE = np.matmul(np.matmul(state_D_delta.T, np.linalg.pinv(state_D_state)), state_D_delta)
    return PBE

def compute_VE(MDP, weights):
    gamma = MDP.gamma
    D_pi = [(1 - gamma)/7, (1 - gamma)/7, (1 - gamma)/7, (1 - gamma)/7,
            (1 - gamma)/7, (1 - gamma)/7, (1 - gamma)/7 + gamma]
    D_pi = np.diag(D_pi)
    v_w = np.matmul(MDP.state_feature, weights)
    VE = np.matmul(np.matmul(v_w.T, D_pi), v_w)
    return VE

def TDC(MDP, alpha=0.005, beta=0.05, max_step=LENGTH):
    w = w = [1, 1, 1, 1, 1, 1, 10, 1]
    w = np.array(w, dtype=np.float64)
    u = np.random.rand(w.shape[0],)
    PBE_record, VE_record, w_record = [], [], []
    state_index = np.random.randint(0, high=MDP.n_state)
    state = MDP.state_feature[state_index]
    for _ in range(max_step):
        action = behaviour(MDP)
        next_state = MDP.transit(action)
        delta = MDP.reward() + MDP.gamma * np.dot(next_state, w) - np.dot(state, w)
        rho = 0
        if action == SOLID:
            rho = 7
        PBE_record.append(np.sqrt(compute_PBE(MDP, w)))
        VE_record.append(np.sqrt(compute_VE(MDP, w)))
        w_record.append(np.copy(w))
        u += beta * rho * (delta - np.dot(u.T, state)) * state
        w += alpha * rho * (delta * state - MDP.gamma * np.dot(state.T, u) * next_state)
        state = next_state
    return np.array(PBE_record), np.array(VE_record), np.array(w_record)

def expected_TDC(MDP, alpha=0.005, beta=0.05, max_sweep=LENGTH):
    w = [1, 1, 1, 1, 1, 1, 10, 1]
    w = np.array(w, dtype=np.float64)
    u = np.random.rand(w.shape[0],)
    PBE_record, VE_record, w_record = [], [], []
    for _ in range(max_sweep):
        u_increment, w_increment = 0, 0
        for state in MDP.state_feature:
            delta = MDP.gamma * np.dot(w.T, MDP.state_feature[LOWER]) - np.dot(w.T, state)
            u_increment += 7 * (delta - np.dot(u.T, state)) * state
            w_increment += 7 * (delta * state - MDP.gamma * np.dot(state.T, u) * MDP.state_feature[LOWER])
        PBE_record.append(np.sqrt(compute_PBE(MDP, w)))
        VE_record.append(np.sqrt(compute_VE(MDP, w)))
        w_record.append(np.copy(w))
        u += beta * u_increment / MDP.n_state
        w += alpha * w_increment / MDP.n_state
    return np.array(PBE_record), np.array(VE_record), np.array(w_record)

def main():
    MDP = Baird_MDP()
    PBE_record, VE_record, w_records = TDC(MDP)
    expected_PBE_record, expected_VE_record, expected_w_records = expected_TDC(MDP)
    x = np.arange(LENGTH)
    colors = ['black', 'red', 'darksalmon', 'tan',
              'darkgreen', 'blue', 'darkorchid', 'goldenrod', 
              'pink', 'cyan']
    plot_curves = [{'PBE': PBE_record, 'VE': VE_record, 'weights': w_records}, 
                    {'PBE': expected_PBE_record, 'VE': expected_VE_record, 'weights': expected_w_records}]
    title_list = ['TDC', 'Expected TDC']
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    for i, ax in enumerate(axs):
        ax.title.set_text(title_list[i])
        for d in range(DIM+2):
            if d < 8:
                y = plot_curves[i]['weights'][:, d]
                ax.plot(x, y, label='w'+str(d + 1), color=colors[d])
            elif d == 8:
                y = plot_curves[i]['PBE']
                ax.plot(x, y, label='$\sqrt{PBE}$', color=colors[d])
            else:
                y = plot_curves[i]['VE']
                ax.plot(x, y, label='$\sqrt{VE}$', color=colors[d])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()