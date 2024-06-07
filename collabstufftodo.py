# cell 1
import numpy as np
from datetime import datetime as dt

# everything is vectorised by width, works with 1 but has dim=2
# state space [-.07,.07] x [-1.2,.5]
# action space [0,1,2]

class mountain_car(object):

  def __init__(self):
      pass

  def init(self, width):
      return np.vstack((np.random.uniform(-.5,-.5,width),
                        np.zeros(width))).T

  # the done signal is scalar
  def step(self, s, a, h, H=800):
      F=.001;      G=.0025
      p, dp = s.T
      dp = np.clip((dp*(p!=-1.2) + (a-1)*F - np.cos(3*p)*G)*(p!=.5),
                  -.07,.07)
      p = np.clip(p + dp, -1.2, .5)
      return np.vstack((p,dp)).T, (p!=.5).astype(np.float32)*(h==H-1), (h==H-1)

# cell 2
def data(n, env, nactions, horizon=800):
    # sample data
    # NB. the data is only stored as tuples for saving it
    data = np.zeros((horizon,n,6))
    s = env.init(n)
    for h in range(horizon):
        s_, c, _ = env.step(s, a:=np.random.randint(nactions,size=n),h,horizon)
        data[h] = np.concatenate((s,np.vstack((a,c)).T,s:=s_),1)
    
    # np.savetxt(str:=dt.now().strftime(f'{name}-{n}-%H.%M.%S-%d.%m'),
    #            data)
    # prepare data for fqi
    #nfeatures = features(data[0][0]).size
    #inputs = np.zeros((n, nactions*nfeatures))
    #cs = np.zeros(n)
    #s_s = np.zeros((n,nfeatures))
    #for i, (s,a,c,s_) in enumerate(data):
    #    inputs[i, a*nfeatures:(a+1)*nfeatures] = features(s)
    #    cs[i] = c
    #    s_s[i] = features(s_)
    return data

# cell 3
def foo(n, env, nsucc, nactions, features, horizon=800):
  succs = 0
  fails = 0
  buf = np.zeros((horizon,n,6))
  while succs < nsucc or fails < n - nsucc:
    d = data(n, env, nactions, horizon)
    succidx = d[horizon-1,:,3] == 0
    dsucc = d[:,succidx][:,:nsucc-succs]
    dfail = d[:,~succidx][:,:n-nsucc-fails]
    buf[:,succs:(succs:=succs+dsucc.shape[1])] = dsucc
    buf[:,nsucc+fails:nsucc+(fails:=fails + dfail.shape[1])] = dfail
  dsucc = dfail = None# garbage collect
  # prepare data for fqi
  # assumes states are size 2
  nfeatures = features(buf[0][0][:2]).size
  inputs = np.zeros((horizon, nactions, n, nfeatures))
  cs = np.zeros((horizon, nactions, n))
  s_s = np.zeros((horizon, nactions, n,nfeatures))
  for i in range(horizon):
    for a in range(nactions):
      idx = buf[i,:,2]==a
      inputs[i,a,idx] = features(buf[i,:,:2])[idx]
      cs[i,a,idx] = buf[i,:,3][idx]
      s_s[i,a,idx] = features(buf[i,:,-2:])[idx]
  return inputs, cs, s_s
    
n=1000
env=mountain_car()
MIN = np.array([-.07,-1.2])
SPREAD = np.array([.14,1.7])
ORD = 3
b = basis(MIN.size, ORD)
features = lambda x: fourier(x, b, MIN, SPREAD)
_ = foo(n,env,1,3,features)

# cell 5
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from scipy.special import log_expit 
import numpy as np

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

def finitehorizonfqi(loss, inputs, cs, s_s, gamma, rounds):
    l, dl = (llog, dllog) if loss=='log' else (lsq, dlsq)
    shape = (inputs.shape[1] // s_s.shape[1], s_s.shape[1])
    w = np.zeros(inputs.shape[0], shape)
    for h in range(rounds-1,-1,-1):
      for a in range(3): # assume 3 actions
        # this target computation assumes (nonzero cost => done)
        x0 = np.random.uniform(low = -0.2,high = 0.2,size = w.size) # should be outside the for loop
        targets = np.maximum(cs[h,a], gamma * sigmoid(w @ s_s[h,a].T).min(axis = 0))
        w[h,a] = minimize(fun = l, x0 = x0, method='bfgs',
                        jac=dl, args = (inputs[h,a], targets[h,a]),
                        ).x.reshape(shape)
    return w
