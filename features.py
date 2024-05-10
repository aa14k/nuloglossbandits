from itertools import product
import numpy as np

def basis(dim, order):
    return np.array(list(product(np.arange(order+1), repeat=dim)))

# min=array([-4.8,-5,-.418,-5]), spread=array([9.6,10,.836,10])):
def fourier(states, basis, min, spread):
    return np.cos(np.pi*(states-min)/spread@basis.T)
