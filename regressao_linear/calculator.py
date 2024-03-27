import numpy as np
from dataset import add_ones_to_matrix

class ThetaIteration:
    
    def __init__(self, alpha: float, initialTheta: np.array) -> None:
        self.alpha = alpha
        self.theta = initialTheta

    @add_ones_to_matrix
    def nextTheta(self, X: np.matrix, y: np.array) -> np.array:
        N = X.shape[0]
        sum_therm = [ sum(self.theta * X[:,j] - y) for j in range(X.shape[1]) ]
        self.theta = self.theta - ((self.alpha / N) * sum_therm)
        return self.theta
    
    @add_ones_to_matrix
    def cost(self, X: np.matrix, y: np.array):
        N = X.shape[0]
        sum_therm = [ sum((self.theta * X[:,j] - y) ^ 2) for j in range(X.shape[1]) ]
        return sum_therm / (2 * N)
    