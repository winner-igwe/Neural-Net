import numpy as np
class L1():
    @staticmethod
    def cost(lmbda,size,weights):
        return (lmbda/size) * sum(np.sum(np.abs(w)) for w in weights)
    
    @staticmethod
    def reg_weights(lmbda, w):
        return lmbda * np.sign(w)
