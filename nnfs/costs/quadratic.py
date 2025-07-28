import  numpy as np
class QuadraticCost:

    @staticmethod
    def compute_cost(a, y):
        """Compute the quadratic cost: 0.5 * ||a - y||^2"""
        return 0.5 * np.sum((a - y) ** 2)

    @staticmethod
    def delta(a, y, z):
        """Compute the error delta using the derivative of the sigmoid."""
        return (a - y) * QuadraticCost.sigmoid_prime(z)

    @staticmethod
    def sigmoid(z):
        """Standard sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        s = QuadraticCost.sigmoid(z)
        return s * (1 - s)

