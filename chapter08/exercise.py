import numpy as np
import matplotlib.pyplot as plt

MAX_STEPS = 200000
EPSILON = 0.1
TERMINAL_PROB = 0.1
ACTIONS = [0, 1]
N_ACTIONS = 2

def argmax(value):
    max_q = np.max(value)
    action = np.random.choice([a for a, q in enumerate(value) if q == max_q])
    return action

class Env:
    def __init__(self, n_states, n_branch):
        self.n_states = n_states
        self.n_branch = n_branch
        self.dynamic = np.random.randint(n_states, size=(n_states, N_ACTIONS, n_branch))  # dynmaic = P(s' | s, a)
        self.reward = np.random.randn(n_states, N_ACTIONS, n_branch)  # reward = R(s, a)

    def step(self, state, action):
        if np.random.rand() < TERMINAL_PROB:
            return 0, self.n_states
        random_id = np.random.randint(self.n_branch)
        reward = self.reward[state, action, random_id]
        next_state = self.dynamic[state, action, random_id]
        return reward, next_state

# Using Monte-Carlo method to evaluate the v_pi(s0).
def evaluation(env, q_value):
    n_runs = 1000
    returns = []
    for _ in range(n_runs):
        state = 0
        rewards = 0
        while state < env.n_states:
            action = argmax(q_value[state])
            reward, next_state = env.step(state, action)
            rewards += reward
            state = next_state
        returns.append(rewards)
    return np.mean(returns)

# uniform method
def uniform(env, eval_step, max_iter=MAX_STEPS):
    Q = np.zeros((env.n_states, N_ACTIONS))
    results = []
    for t in range(max_iter):
        state = np.random.randint(env.n_states)
        action = np.random.randint(2)
        all_possible_rewards = env.reward[state, action]
        all_possible_next_states = env.dynamic[state, action]
        max_next_Q = np.max(Q[all_possible_next_states, :], axis=1)
        Q[state, action] = (1 - TERMINAL_PROB) * np.mean(all_possible_rewards + max_next_Q)
        if t % eval_step == 0:
            V = evaluation(env, Q)
            results.append(V)
            print('t = %d and V(s0) = %.3f' % (t, V))
    return results

# on-policy method
def on_policy(env, eval_step, max_iter=MAX_STEPS):
    Q = np.zeros((env.n_states, N_ACTIONS))
    results = []
    state = 0
    for t in range(max_iter):
        if np.random.rand() < EPSILON:
            action = np.random.choice(ACTIONS)
        else:
            action = argmax(Q[state])
        all_possible_rewards = env.reward[state, action]
        all_possible_next_states = env.dynamic[state, action]
        max_next_Q = np.max(Q[all_possible_next_states, :], axis=1)
        Q[state, action] = (1 - TERMINAL_PROB) * np.mean(all_possible_rewards + max_next_Q)

        _, next_state = env.step(state, action)
        if next_state == env.n_states:
            next_state = 0
        state = next_state

        if t % eval_step == 0:
            V = evaluation(env, Q)
            results.append(V)
            print('t = %d and V(s0) = %.3f' % (t, V))
    return results

def main():
    n_states = 1000
    branch_list = [1, 3, 10]
    eval_step = 200
    n_runs = 10  # number of experiments
    method_list = [on_policy, uniform]
    for n_branch in branch_list:
        env = Env(n_states, n_branch)
        for method in method_list:
            value = []
            for i in range(n_runs):
                print("------------------%d-th experiment for branch: %d-------------------" % (i + 1, n_branch))
                results = method(env, eval_step)
                value.append(results)
            value = np.mean(value, axis=0)
            steps = np.arange(0, MAX_STEPS, eval_step)
            plt.plot(steps, value, label=f'b = {n_branch}, {method.__name__}')

    plt.legend()
    plt.xlabel('num of iterations (1000 states)')
    plt.ylabel('$V(s_0)$')
    plt.show()

if __name__ == '__main__':
    main()