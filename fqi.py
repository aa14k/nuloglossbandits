from tqdm import tqdm
from scipy.optimize import minimize
import numpy as np

def sigmoid(z): return 1/(1+np.exp(-np.clip(z, -36, 36)))

def lsq(w,x,y):
    return np.sum((sigmoid(x@w) - y)**2)

def dlsq(w,x,y):
    p = sigmoid(x@w)
    return 2 * (p - y) * (p * (1-p))@x
 
def llog(w,x,y):
    p = sigmoid(x@w)
    return -sum(y*np.log(p) + (1-y)*np.log(1-p))

def dllog(w,x,y): return (sigmoid(x@w)-y)@x

def fqi(loss, inputs, cs, s_s, gamma, rounds):
    l, dl = (llog, dllog) if loss=='log' else (lsq, dlsq)
    shape = (inputs.shape[1]//s_s.shape[1], s_s.shape[1])
    w = np.zeros(shape)
    for _ in tqdm(range(rounds)):
        # this target computation assumes (nonzero cost => done)
        targets = np.maximum(cs, gamma * sigmoid(w@s_s.T).min(axis=0))
        w = minimize(fun=l, x0=np.zeros(w.size), method='L-BFGS-B',
                     jac=dl, args=(inputs, targets),
                     options={'gtol': 1e-4}).x.reshape(shape)
    return w
