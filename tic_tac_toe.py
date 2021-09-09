"""
Tic-Tac-Toe example from Reinforcement Learning: An Introduction Chapter 1.
Author: Yanting Miao
"""

import torch
import numpy as np

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
        self.hash = self.get_hash()
        self.winner = None  # 1: player 1 wins; -1: player 2 wins; 0: tie; None: game is not done yet.

    def get_hash(self):
        # hash value := ternary of "board" (Imagine board be a sequence, which element
        # is 0 or 1 or 2, and thus, each board/sequence can have a unique hash value)
        hash_value = 0
        for i in range(N_ROW):
            for j in range(N_COL):
                hash_value += 3 * hash_value + (self.board[i][j] + 1)
        return hash_value

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
            if results == -3:
                self.winner = -1
                return self.winner
        
        tie = np.sum(np.abs(self.board))
        if tie == BOARD_SIZE:
            self.winner = 0
            return self.winner
        return self.winner

    def get_nextState(self, i, j, symbol):
        """
        (i, j): (row index, col index)
        symbol: 1: player 1; -1: player 2
        """
        next_state = State()
        next_state.board = self.board
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
                new_state = current_state.get_nextState(i, j, current_symbol)
                if new_state.hash not in all_states:
                    winner = new_state.check_winner
                    all_states[new_state.hash] = (new_state, winner)
                    if winner is None:
                        get_all_states_recurrsion(all_states, new_state, -current_symbol)

def get_all_state():
    all_states = {}
    current_state = State()
    current_symbol = 1
    get_all_states_recurrsion(all_states, current_state, current_symbol)
    return all_states


class Player:
    def __init__(self, symbol, step_size=0.1, epsilon=0.1):
        """
        symbol: 1: player 1; -1: player 2
        step_size: step-size parameter
        epsilon: the probability of exploration
        """
        self.symbol = symbol
        self.step_size = step_size
        self.epislon = epsilon
        
