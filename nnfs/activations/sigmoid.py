import numpy as np
class Sigmoid:
    @staticmethod
    def fn(z):
        z = np.clip(z, -500, 500)  # Add this
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def fn_prime(z):
        sig = Sigmoid.fn(z)
        return sig * (1 - sig)