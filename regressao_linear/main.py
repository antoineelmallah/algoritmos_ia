from calculator import ThetaIteration
import numpy as np
from dataset import get_dataset

alpha = .001

dataset = get_dataset('/home/mallah/Downloads/ice_cream_sales-temperatures.csv')

x = dataset[:,0]
y = dataset[:,1]

iterator = ThetaIteration(alpha=alpha, initialTheta=np.array([0]))

while iterator.cost(x, y) < alpha:
    iterator.nextTheta(x, y)

print(iterator.theta)