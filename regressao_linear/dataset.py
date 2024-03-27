import numpy as np

def get_dataset(path: str):
    result = np.genfromtxt(path, delimiter=',')
    return result

def add_ones_column(X:np.matrix) -> np.matrix:
    return np.hstack((np.ones(shape=(X.shape[0],1)), X))

def add_ones_to_matrix(function):
    def wrapper(*args, **kwargs):
        new_args = [ add_ones_column(arg) if type(arg) == np.matrix else arg for arg in args ]
        return function(*new_args, **kwargs)
    return wrapper

if __name__ == '__main__':
    result = get_dataset('/home/mallah/Downloads/ice_cream_sales-temperatures.csv')
    print(result)
    print(type(result[0,0]), type(result[0,1]))