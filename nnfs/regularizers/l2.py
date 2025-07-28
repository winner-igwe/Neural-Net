import numpy as np

class L2:
    @staticmethod
    def cost(lmbda, size, weights):
        return 0.5 * (lmbda / size) * sum(np.linalg.norm(w)**2 for w in weights)

    @staticmethod
    def reg_weights(lmbda, w):
        return lmbda * w
