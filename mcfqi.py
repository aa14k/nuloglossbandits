from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from scipy.special import log_expit 
import numpy as np
from tqdm import tqdm

def lsq(w,x,y): return np.sum(np.power(sigmoid(x @ w) - y,2))

def dlsq(w,x,y):
    p = sigmoid(x @ w)
    return 2 * (p - y) * (p * (1 - p)) @ x

def llog(w,x,y):
    xw = x @ w
    return -1.0 * np.sum(y * log_expit(xw) + (1-y) * log_expit(-xw))

def dllog(w,x,y): return (sigmoid(x @ w) - y) @ x

# assumes 3 actions
def fqi(loss, ss, a_s, cs, s_s, H):
    featsize = ss.shape[-1]
    w = np.zeros((H,3,featsize))
    l, dl = (llog, dllog) if loss=='log' else (lsq, dlsq)
    for h in tqdm(range(H-1,-1,-1)):
        for a in range(3):
            if h+1<H:
                targets=sigmoid(w[h+1]@(s_s[h][a_s[h]==a]).T).min(axis=0)
            else:
                targets = cs[h][a_s[h]==a]
            w[h,a] = minimize(l, 0.0*np.random.uniform(-.2,.2,featsize),
                              method='bfgs', jac=dl,
                              args = (ss[h][a_s[h]==a], targets)).x
    return w
