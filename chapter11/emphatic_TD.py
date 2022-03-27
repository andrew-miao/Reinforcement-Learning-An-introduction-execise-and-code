from matplotlib import widgets
import numpy as np
import matplotlib.pyplot as plt

DASHED = 0
SOLID = 1
LOWER = 6
GAMMA = 0.99
LENGTH = 1000
DIM = 8
INTERSET = 1

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

def compute_VE(MDP, weights):
    D_mu = [1/7, 1/7, 1/7, 1/7,
            1/7, 1/7, 1/7]
    D_mu = np.diag(D_mu)
    v_w = np.matmul(MDP.state_feature, weights)
    VE = np.matmul(np.matmul(v_w.T, D_mu), v_w)
    return VE


def Emphatic_TD(MDP, alpha=0.03, max_sweep=LENGTH):
    w = [1, 1, 1, 1, 1, 1, 10, 1]
    w = np.array(w, dtype=np.float64)
    emphasis = 0
    VE_record, w_record = [], []
    for _ in range(max_sweep):
        w_increment, sum_emphasis = 0, 0
        for i, state in enumerate(MDP.state_feature):
            delta = MDP.reward() + MDP.gamma * np.dot(w, MDP.state_feature[LOWER]) - np.dot(w, state)
            # M_t := gamma * rho_{t - 1} * M_{t - 1} + I_t, rho_{t - 1} can be determined by the current state
            # if the current state is LOWER state, then rho_{t - 1} = 1 / (1/7) = 7; otherwise, rho_{t - 1} = 0 / (6/7) = 0
            rho = 0
            if i == LOWER:
                rho = 7
            temp_emphasis = MDP.gamma * rho * emphasis + INTERSET
            sum_emphasis += temp_emphasis
            # delta_w := M_t * E_b[rho_t * E_pi[delta_t]|S_t=s] * grad(v(s; w))
            w_increment += temp_emphasis * delta * state
        VE_record.append(np.sqrt(compute_VE(MDP, w)))
        w_record.append(np.copy(w))
        w += alpha * w_increment / MDP.n_state
        emphasis = sum_emphasis / MDP.n_state
    return np.array(VE_record), np.array(w_record)

def main():
    MDP = Baird_MDP()
    VE_record, w_record = Emphatic_TD(MDP)
    x = np.arange(LENGTH)
    colors = ['black', 'red', 'darksalmon', 'tan',
              'darkgreen', 'blue', 'darkorchid', 'goldenrod', 'pink']
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Sweeps')
    plt.title('Emphatic TD')
    for d in range(DIM+1):
        if d < 8:
            y = w_record[:, d]
            plt.plot(x, y, label='w'+str(d + 1), color=colors[d])
        else:
            y = VE_record
            plt.plot(x, y, label='$\sqrt{VE}$', color=colors[d])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()