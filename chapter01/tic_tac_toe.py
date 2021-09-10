"""
Tic-Tac-Toe example from Reinforcement Learning: An Introduction Chapter 1.
Author: Yanting Miao
"""

import torch
import numpy as np
import time
import math

N_ROW = 3
N_COL = 3
BOARD_SIZE =  N_ROW * N_COL

class State:
    """
    Two players game. 1 represents player 1; -1 represents player 2.
    Hence, if one row/col/diagonal reach 3 player 1 wins, or reach -3 player 2 wins;
    or tie if there is no aviable positions for playing.
    """
    def __init__(self):
        self.board = np.zeros((N_ROW, N_COL))
        self.winner = None  # 1: player 1 wins; -1: player 2 wins; 0: tie; None: game is not done yet.
        self.hash = None

    def get_hash(self):
        # hash value := ternary of "board" (Imagine board be a sequence, which element
        # is 0 or 1 or 2, and thus, each board/sequence can have a unique hash value)
        if self.hash is None:
            self.hash = 0
            seq = self.board.reshape(N_ROW * N_COL)
            for data in seq:
                self.hash = 3 * self.hash + (data + 1)
            return self.hash
        return self.hash

    def check_winner(self):
        results = []
        # check each row
        for i in range(N_ROW):
            results.append(np.sum(self.board[i, :]))
        
        # check each col
        for i in range(N_COL):
            results.append(np.sum(self.board[:, i]))

        # check each diagonal
        diagonal_1 = 0
        diagonal_2 = 0
        for i in range(N_ROW):
            diagonal_1 += self.board[i, i]
            diagonal_2 += self.board[i, N_ROW - 1 - i]

        results.append(diagonal_1)
        results.append(diagonal_2)
        for result in results:
            if result == 3:
                self.winner = 1
                return self.winner
            if result == -3:
                self.winner = -1
                return self.winner
        
        tie = np.sum(np.abs(self.board))
        if tie == BOARD_SIZE:
            self.winner = 0
            return self.winner
        return self.winner

    def get_nextState(self, action):
        """
        action = (i, j, symbol)
        (i, j): (row index, col index)
        symbol: 1: player 1; -1: player 2
        """
        (i, j, symbol) = action
        next_state = State()
        next_state.board = np.copy(self.board)
        next_state.board[i][j] = symbol
        return next_state
    
    def show_board(self):
        for i in range(N_ROW):
            print('-------------')
            out = '| '
            for j in range(N_COL):
                if self.board[i, j] == 1:
                    token = '*'
                elif self.board[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')

def get_all_states_recurrsion(all_states, current_state, current_symbol):
    """
    This function refers to https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter01/tic_tac_toe.py
    """
    for i in range(N_ROW):
        for j in range(N_COL):
            if current_state.board[i][j] == 0:
                new_state = current_state.get_nextState((i, j, current_symbol))
                new_state_hash = new_state.get_hash()
                if new_state_hash not in all_states:
                    winner = new_state.check_winner()
                    all_states[new_state_hash] = (new_state, winner)
                    if winner is None:
                        get_all_states_recurrsion(all_states, new_state, -current_symbol)

def get_all_state():
    current_state = State()
    current_symbol = 1
    all_states = {current_state.get_hash(): (current_state, current_state.check_winner())}
    get_all_states_recurrsion(all_states, current_state, current_symbol)
    return all_states

all_states = get_all_state()

# AI player
class Player:
    """
    The core idea of this AI player is to estimate all states that happen in the Tic-Tac-Toe game.
    """
    def __init__(self, symbol, step_size=0.1, epsilon=0.1, random_seed=42):
        """
        symbol: 1: player 1; -1: player 2
        step_size: step-size parameter
        epsilon: the probability of exploration
        random_seed: random seed
        """
        self.symbol = symbol
        self.step_size = step_size
        self.epislon = epsilon
        self.value = {}  # state-value pair
        self.happend_states = []  # store state during a one game
        self.greedy = []  # greedy[i] = True: greedy action; greedy[i] = False: exploration
        # np.random.seed(random_seed)
        self.value_init()

    def value_init(self):
        # value initialization
        for state_hash in all_states:
            if all_states[state_hash][1] is None:
                self.value[state_hash] = 0.5
            else:
                if all_states[state_hash][1] == self.symbol:
                    self.value[state_hash] = 1
                elif all_states[state_hash][1] == -self.symbol:
                    self.value[state_hash] = 0
                else:
                    self.value[state_hash] = 0.5
    
    def reset(self):
        """
        Start a new game
        """
        self.happend_states, self.greedy = [], []
    
    def set_state(self, state):
        self.happend_states.append(state.get_hash())
        self.greedy.append(True)

    def act(self, state):
        possible_actions = []
        possible_next_states = []
        possible_values = []
        for i in range(N_ROW):
            for j in range(N_COL):
                if state.board[i][j] == 0:
                    possible_actions.append((i, j, self.symbol))
                    lookahead_state = state.get_nextState((i, j, self.symbol))
                    possible_next_states.append(lookahead_state)
                    possible_values.append(self.value[lookahead_state.get_hash()])

        index = None
        if np.random.rand() > self.epislon:
            # to select one of the actions of equal value at random due to Python's sort is stable
            value_action_pairs = []
            for value, action in zip(possible_values, possible_actions):
                value_action_pairs.append((value, action))
            np.random.shuffle(value_action_pairs)
            value_action_pairs.sort(key=lambda x:x[0], reverse=True)
            action = value_action_pairs[0][1]
        else:
            index = np.random.randint(len(possible_actions))
            action = possible_actions[index]
            self.greedy[-1] = False
        return action

    def backup(self):
        """
        V(S_t) := V(S_t) + alpha * [V(S_t+1) - V(S_t)]
        """
        # print(self.happend_states)
        # print(self.greedy)
        e = self.happend_states[-1]
        b = all_states[e]
        for i in range(len(self.happend_states) - 2, 0, -1):
            current_state = self.happend_states[i]
            next_state = self.happend_states[i + 1]
            c = self.value[current_state]
            n = self.value[next_state]
            b = all_states[next_state]
            self.value[current_state] += self.greedy[i] * self.step_size * (self.value[next_state] - self.value[current_state])
    
    def generate_path(self):
        player = 'first' if self.symbol == 1 else 'second'
        PATH = './player_' + player +'.pt'
        return PATH

    def save_policy(self):
        PATH = self.generate_path()
        torch.save(self.value, PATH)
    
    def load_policy(self):
        PATH = self.generate_path()
        self.value = torch.load(PATH)

class HumanPlayer:
    def __init__(self, symbol):
        self.symbol = symbol
        if self.symbol == 1:
            print('Your symbol: *')
        else:
            print('Your symbol: x')
        print('Board')
        print('------')
        print('q|w|e')
        print('a|s|d')
        print('z|x|c')
        print('------')
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None
    
    def reset(self):
        pass

    def set_state(self, state):
        pass
    
    def act(self, state):
        choice = input('Enter your action:')
        idx = self.keys.index(choice)
        i = math.floor(idx / N_ROW)
        j = idx - N_ROW * i
        return (i, j, self.symbol)

class Tic_Tac_Toe:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
    
    def reset(self):
        self.player1.reset()
        self.player2.reset()

    def altenate(self):
        while True:
            yield self.player1
            yield self.player2

    def play(self, show_board=False):
        """
        show_board=True: show board after each action
        """
        altenator = self.altenate()
        state = State()
        if show_board:
            state.show_board()
        self.reset()
        self.player1.set_state(state)
        self.player2.set_state(state)
        while True:
            current_player = next(altenator)
            action = current_player.act(state)
            next_state = state.get_nextState(action)
            state = next_state
            self.player1.set_state(state)
            self.player2.set_state(state)
            if show_board:
                state.show_board()
            if state.check_winner() is not None:
                return state.winner

def timeSince(start):
    end = time.time()
    s = end - start
    m = math.floor(s / 60)
    s -= m * 60
    return m, s

def train(player1, player2, n_epochs=1e5, print_every=500):
    win1, win2, tie = 0, 0, 0
    tic_tac_toe = Tic_Tac_Toe(player1, player2)
    n_epochs = int(n_epochs)
    start = time.time()
    for i in range(n_epochs):
        winner = tic_tac_toe.play()
        player1.backup()
        player2.backup()
        if winner == 1:
            win1 += 1
        elif winner == -1:
            win2 += 1
        else:
            tie += 1
        if (i + 1) % print_every == 0:
            m, s = timeSince(start)
            print('=' * 128)
            print('Spend %dm, %ds' % (m, s))
            print('Epoch: %d' % (i + 1))
            print('Player 1 win rate = %.2f%%, Player 2 win rate = %.2f%%, Tie rate = %.2f%%'
                  % (win1 * 100 / (i + 1), win2 * 100 / (i + 1), tie * 100 / (i + 1)))

    player1.save_policy()
    player2.save_policy()
    print('Two players polices are successfully saved!')
    print('Training is done!')

def twoAIcompete(player1, player2, rounds=1000):
    win1, win2, tie = 0, 0, 0
    tic_tac_toe = Tic_Tac_Toe(player1, player2)
    player1.load_policy()
    player2.load_policy()
    for _ in range(rounds):
        winner = tic_tac_toe.play()
        if winner == 1:
            win1 += 1
        elif winner == -1:
            win2 += 1
        else:
            tie += 1
    print('Player 1 win rate = %.2f%%, Player 2 win rate = %.2f%%, Tie rate = %.2f%%'
          % (win1 * 100 / rounds, win2 * 100/ rounds, tie * 100/ rounds))

def playWithHuman():
    flag = True
    while flag:
        human_choice = input('Do you want play first (y/n)')
        if human_choice == 'y':
            player1 = HumanPlayer(symbol=1)
            human_symbol = 1
            player2 = Player(symbol=-1)
            player2.load_policy()
        else:
            player1 = Player(symbol=1)
            player1.load_policy()
            player2 = HumanPlayer(symbol=-1)
            human_symbol = -1
        tic_tac_toe = Tic_Tac_Toe(player1, player2)
        winner = tic_tac_toe.play(show_board=True)
        if winner == human_symbol:
            print('You win!')
        elif winner == -human_symbol:
            print('You lose.')
        else:
            print('Tie!')
        human_choice = input('Play again? (y/n)')
        if human_choice == 'n':
            flag = False
    
if __name__ == '__main__':
    epsilon = 0.01
    player1 = Player(symbol=1, epsilon=epsilon)
    player2 = Player(symbol=-1, epsilon=epsilon)
    train(player1, player2)
    player1 = Player(symbol=1, epsilon=0)
    player2 = Player(symbol=-1, epsilon=0)
    twoAIcompete(player1, player2)
    playWithHuman()