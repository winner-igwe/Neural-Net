import numpy as np

class LogCost():
    def compute_cost(a,y):
        return np.sum(-y * np.log(a))
    
    @staticmethod
    def delta(a,y,z):
        return (a-y)
