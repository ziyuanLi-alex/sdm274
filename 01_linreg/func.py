import numpy as np

class AdvancedFuncs():
    def __init__(self):
        pass

    def mean_squared_error(true, pred):
        squared_error = np.square(true - pred)
        summed = np.sum(squared_error)
        return summed / true.size 

    