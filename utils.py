import random
import numpy as np


def get_random_state(rad):
    random.seed(rad)
    np.random.seed(rad)
