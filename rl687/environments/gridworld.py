import numpy as np
from .skeleton import Environment

class Gridworld(Environment):
    """
    The Gridworld as described in the lecture notes of the 687 course material.

    Actions: up (0), down (1), left (2), right (3)

    Environment Dynamics: With probability 0.8 the robot moves in the specified
        direction. With probability 0.05 it gets confused and veers to the
        right -- it moves +90 degrees from where it attempted to move, e.g.,
        with probability 0.05, moving up will result in the robot moving right.
        With probability 0.05 it gets confused and veers to the left -- moves
        -90 degrees from where it attempted to move, e.g., with probability
        0.05, moving right will result in the robot moving down. With
        probability 0.1 the robot temporarily breaks and does not move at all.
        If the movement defined by these dynamics would cause the agent to
        exit the grid (e.g., move up when next to the top wall), then the
        agent does not move. The robot starts in the top left corner, and the
        process ends in the bottom right corner.

    Rewards: -10 for entering the state with water
            +10 for entering the goal state
            0 everywhere else



    """

    def __init__(self, startState=0, endState=24, shape=(5,5), obstacles=[12, 17], waterStates=[6, 18, 22]):
        self.startState = startState
        self.endState = endState
        self.shape = shape
        self.obstacles = obstacles
        self.waterStates = waterStates
        self.currentState = startState
        self.lastActiontaken = None
        self.lastReward  = None
        self.grid = np.arange(25).reshape(shape)
        self.UP, self.DOWN, self.LEFT, self.RIGHT = [0 , 1,  2, 3]

        print(self.grid)
    @property
    def name(self):
        return "687-Gridworld"

    @property
    def reward(self):
        return self.lastReward

    @property
    def action(self):
        return self.lastActiontaken

    @property
    def isEnd(self):
        print((self.currentState == self.endState))
        return (self.currentState == self.endState)
    @property
    def state(self):
        return self.currentState

    @property
    def gamma(self):
        pass

    def step(self, action):
        c_index = np.where(self.grid == self.currentState)
        new_indices = np.array(c_index).T + np.array([[-1, 0], [+1, 0], [0, -1], [0, 1]])
        if action == self.UP:

            if new_indices[self.UP, 0] > 0 and ~(self.grid[tuple(new_indices[self.UP, :])] in self.obstacles):
               up_state  = self.grid[tuple(new_indices[self.UP, :])]
            else:
                up_state = self.currentState
            #left state reward
            if new_indices[self.LEFT, 1] > 0 and ~(self.grid[tuple(new_indices[self.LEFT, :])] in self.obstacles):
               left_state  = self.grid[tuple(new_indices[self.LEFT, :])]
            else:
                left_state = self.currentState

            if new_indices[self.RIGHT, 1] < 5 and ~(self.grid[tuple(new_indices[self.RIGHT, :])] in self.obstacles):
               right_state  = self.grid[tuple(new_indices[self.RIGHT, :])]
            else:
                right_state = self.currentState

            r = 0.9*self.R(up_state)
            #staying there  reward
            r += 0.1*self.R(self.currentState)
            r += 0.05*self.R(left_state)
            r += 0.05*self.R(right_state)
            p =[0.9, 0.05, 0.05, 0.1]
            next_state_idx = np.where(np.random.multinomial(1, p)==1)[0][0]
            next_states = [up_state, left_state, right_state, self.currentState]


            self.currentState = next_states[next_state_idx]

        elif action == self.DOWN:

            if new_indices[self.DOWN, 0] < 5 and ~(self.grid[tuple(new_indices[self.DOWN, :])] in self.obstacles):
               down_state  = self.grid[tuple(new_indices[self.DOWN, :])]
            else:
                down_state = self.currentState
            #left state reward
            if new_indices[self.RIGHT, 1] < 5 and ~(self.grid[tuple(new_indices[self.RIGHT, :])] in self.obstacles):
               left_state  = self.grid[tuple(new_indices[self.RIGHT, :])]
            else:
                left_state = self.currentState

            if new_indices[self.LEFT, 1] > 0 and ~(self.grid[tuple(new_indices[self.LEFT, :])] in self.obstacles):
               right_state  = self.grid[tuple(new_indices[self.LEFT, :])]
            else:
                right_state = self.currentState

            r = 0.9*self.R(down_state)
            #staying there  reward
            r += 0.1*self.R(self.currentState)
            r += 0.05*self.R(left_state)
            r += 0.05*self.R(right_state)
            p =[0.9, 0.05, 0.05, 0.1]
            next_state_idx = np.where(np.random.multinomial(1, p)==1)[0][0]
            next_states = [down_state, left_state, right_state, self.currentState]


            self.currentState = next_states[next_state_idx]
        elif action == self.LEFT:
            if new_indices[self.LEFT, 1] > 0 and ~(self.grid[tuple(new_indices[self.LEFT, :])] in self.obstacles):
               left_state  = self.grid[tuple(new_indices[self.LEFT, :])]
            else:
                left_state = self.currentState
            #left state reward
            if new_indices[self.UP, 0] > 0 and ~(self.grid[tuple(new_indices[self.UP, :])] in self.obstacles):
               up_state  = self.grid[tuple(new_indices[self.UP, :])]
            else:
                up_state = self.currentState

            if new_indices[self.DOWN, 0] < 5 and ~(self.grid[tuple(new_indices[self.DOWN, :])] in self.obstacles):
               down_state  = self.grid[tuple(new_indices[self.DOWN, :])]
            else:
                down_state = self.currentState

            r = 0.05*self.R(down_state)
            #staying there  reward
            r += 0.1*self.R(self.currentState)
            r += 0.9*self.R(left_state)
            r += 0.05*self.R(up_state)
            p =[0.9, 0.05, 0.05, 0.1]
            next_state_idx = np.where(np.random.multinomial(1, p)==1)[0][0]
            next_states = [left_state, down_state, up_state, self.currentState]


            self.currentState = next_states[next_state_idx]
        elif action == self.RIGHT:
            if new_indices[self.RIGHT, 1] < 5 and ~(self.grid[tuple(new_indices[self.RIGHT, :])] in self.obstacles):
               right_state  = self.grid[tuple(new_indices[self.RIGHT, :])]
            else:
                right_state = self.currentState
            #left state reward
            if new_indices[self.UP, 0] > 0 and ~(self.grid[tuple(new_indices[self.UP, :])] in self.obstacles):
               down_state  = self.grid[tuple(new_indices[self.UP, :])]
            else:
                down_state = self.currentState

            if new_indices[self.DOWN, 0] < 5 and ~(self.grid[tuple(new_indices[self.DOWN, :])] in self.obstacles):
               up_state  = self.grid[tuple(new_indices[self.DOWN, :])]
            else:
                up_state = self.currentState

            r = 0.9*self.R(right_state)
            #staying there  reward
            r += 0.1*self.R(self.currentState)
            r += 0.05*self.R(up_state)
            r += 0.05*self.R(down_state)
            p =[0.9, 0.05, 0.05, 0.1]
            next_state_idx = np.where(np.random.multinomial(1, p)==1)[0][0]
            next_states = [right_state, down_state, up_state, self.currentState]


            self.currentState = next_states[next_state_idx]
        else:
            print("UNKOWN ACTION !!")

        return r, (self.currentState == self.endState)
    def reset(self):
        self.currentState = self.startState
        pass

    def R(self, _state):
        """
        reward function

        output:
            reward -- the reward resulting in the agent being in a particular state
        """
        if _state in self.waterStates:
            return -10
        elif _state == self.endState:
            return 10
        else:
            return 0
