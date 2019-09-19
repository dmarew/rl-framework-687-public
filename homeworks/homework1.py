import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl687.environments.gridworld import Gridworld

def problemA():
    """
    Have the agent uniformly randomly select actions. Run 10,000 episodes.
    Report the mean, standard deviation, maximum, and minimum of the observed
    discounted returns.
    """
    grid_world = Gridworld()
    rewards = []
    for episod in range(10000):
        is_end  = False
        grid_world.reset()
        r = 0

        while ~is_end:
            action = np.random.randint(4)
            r_, is_end = grid_world.step(action)
            r += r_
        rewards.append(r)
        print(episod, r, is_end)
    rewards = np.array(rewards)
    print(rewards.mean(), rewards.std(), rewards.max(),rewards.min())

def problemB():
    """
    Run the optimal policy that you found for 10,000 episodes. Repor the
    mean, standard deviation, maximum, and minimum of the observed
    discounted returns
    """
    pass  # TODO

def main():
    print("Hello world")
    problemA()
main()
