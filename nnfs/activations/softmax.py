import numpy as np

class Softmax:
    @staticmethod
    def fn(z):
        # Numerically stable softmax
        shift_z = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(shift_z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    @staticmethod
    def prime(z):
        # Usually not used with cross-entropy loss
        s = Softmax.fn(z)
        return s * (1 - s)
