import numpy as np

class CrossEntropyCost:
    @staticmethod
    def compute_cost(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(a, y, z):
        return a - y
