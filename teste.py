import numpy as np

a = np.matrix([[1,2,3],[4,5,6],[7,8,9]])

#ones = np.ones(shape=(3,1))
#print(a)
#print(np.hstack((ones,a)))

def add_ones_column(X:np.matrix) -> np.matrix:
    return np.hstack((np.ones(shape=(X.shape[0],1)), X))

def add_ones_to_matrix(function):
    def wrapper(*args, **kwargs):
        new_args = [ add_ones_column(arg) if type(arg) == np.matrix else arg for arg in args ]
        return function(*new_args, **kwargs)
    return wrapper

@add_ones_to_matrix
def print_x(X:np.matrix):
    print(X)


print_x(a)