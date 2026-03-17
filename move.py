
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

custom_map = [
    "SFFF",
    "FHHF",
    "FFFH",
    "FHFG"
]

class EnvironmentWrapper(FrozenLakeEnv):
    def __init__(self):
        super.__init__()

        

env = FrozenLakeEnv(desc=custom_map, is_slippery=True)

