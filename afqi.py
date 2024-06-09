from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from scipy.special import log_expit 
import numpy as np
from tqdm import tqdm

def lsq(w,x,y):
    return np.sum(np.power(sigmoid(x @ w) - y,2))

def dlsq(w,x,y):
    p = sigmoid(x @ w)
    return 2 * (p - y) * (p * (1 - p)) @ x


def hlsq(w,x,y):
    p = sigmoid(x@w)
    return 2 * (p - y) * (p * (1 - p)) @ x
 
def llog(w,x,y):
    xw = x @ w
    return -1.0 * np.sum(y * log_expit(xw) + (1-y) * log_expit(-xw))

def dllog(w,x,y): return (sigmoid(x @ w) - y) @ x

def hllog(w,x,_):
    p = sigmoid(x @ w)
    D = np.diag(p * (1 - p))
    return x.T @ D @ x

def fqi(loss, inputs, cs, s_s, gamma, rounds):
    l, dl = (llog, dllog) if loss=='log' else (lsq, dlsq)
    shape = (inputs.shape[1] // s_s.shape[1], s_s.shape[1])
    w = np.zeros(shape)
    for _ in tqdm(range(rounds)):
        # this target computation assumes (nonzero cost => done)
        x0 = np.random.uniform(low = -0.2,high = 0.2,size = w.size)
        targets = np.where(cs==-1, cs,
                           gamma *(sigmoid(w @ s_s.T).min(axis = 0)-1))+1
        w = minimize(fun = l, x0 = x0, method='l-bfgs-b',
                    jac=dl, args = (inputs, targets),
                    ).x.reshape(shape)
    return w
