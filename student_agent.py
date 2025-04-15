# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
from board_2048 import board, learning, pattern  

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)













class TD_MCTS_Node:
    def __init__(self, board_state, score, parent=None, action=None):
        self.state = board_state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if self.is_move_legal(a)]

    def is_move_legal(self, action):
        sim = board(self.state)
        return sim.move(action) != -1

    def fully_expanded(self):
        return len(self.untried_actions) == 0

class TD_MCTS:
    def __init__(self, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def select_child(self, node):
        best_val = -float("inf")
        best_child = None
        for child in node.children.values():
            if child.visits == 0:
                UCT_val = float("inf")
            else:
                Q = child.total_reward / child.visits
                UCT_val = Q + self.c * math.sqrt(math.log(node.visits) / child.visits)
            if UCT_val > best_val:
                best_val = UCT_val
                best_child = child
        return best_child

    def evaluate_afterstate(self, b):
        max_value = float("-inf")
        for action in range(4):
            after = board(b)
            score = after.move(action)
            if score == -1:
                continue
            val = score + self.approximator.estimate(after)
            if val > max_value:
                max_value = val
        return max_value if max_value != float("-inf") else 0

    def rollout(self, b, score, depth):
        total_reward = 0
        gamma_factor = 1
        current_board = board(b)
        current_score = score

        for _ in range(depth):
            legal_actions = [a for a in range(4) if board(current_board).move(a) != -1]
            if not legal_actions:
                break
            a = random.choice(legal_actions)
            after = board(current_board)
            reward = after.move(a)
            if reward == -1:
                break
            total_reward += gamma_factor * reward
            gamma_factor *= self.gamma
            current_board = board(after)
            current_board.popup()
        estimate = self.approximator.estimate(current_board)
        total_reward += gamma_factor * estimate
        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_board = board(node.state)
        sim_score = node.score

        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            reward = sim_board.move(node.action)
            if reward == -1:
                break
            sim_score += reward
            sim_board.popup()

        if not node.fully_expanded():
            a = random.choice(node.untried_actions)
            new_board = board(sim_board)
            reward = new_board.move(a)
            if reward == -1:
                return
            new_board.popup()
            new_score = sim_score + reward
            child = TD_MCTS_Node(new_board, new_score, parent=node, action=a)
            node.children[a] = child
            node.untried_actions.remove(a)
            node = child

        rollout_reward = self.rollout(node.state, node.score, self.rollout_depth)
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution


def np_to_board(np_board: np.ndarray) -> board:
    b = board()
    for i in range(4):
        for j in range(4):
            val = np_board[i, j]
            b.set(i * 4 + j, int(np.log2(val)) if val != 0 else 0)
    return b

def load_approximator_from_bin(filename='2048.bin'):
    print("loading...")
    board.lookup.init()
    approximator = learning()
    approximator.add_feature(pattern([0, 1, 2, 3, 4, 5]))
    approximator.add_feature(pattern([4, 5, 6, 7, 8, 9]))
    approximator.add_feature(pattern([0, 1, 2, 4, 5, 6]))
    approximator.add_feature(pattern([4, 5, 6, 8, 9, 10]))
    approximator.load(filename)
    print("loaded!!!")
    return approximator



approximator = load_approximator_from_bin()

def get_action(state, score):
    env = Game2048Env()
    #return random.choice([0, 1, 2, 3]) # Choose a random action
    global approximator
    # You can submit this random agent to evaluate the performance of a purely random strategy.
    if approximator is None:
        approximator = load_approximator_from_bin()
    
    bitboard = np_to_board(state)
    root = TD_MCTS_Node(bitboard, score)
    td_mcts = TD_MCTS(approximator, iterations=500, exploration_constant=1.41, rollout_depth=20, gamma=0.99)

    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    best_act, _ = td_mcts.best_action_distribution(root)
    print(score,best_act)
    return best_act


