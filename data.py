import numpy as np
from datetime import datetime as dt

def data(n, env, nactions, features, save=True):
    # sample data
    # NB. the data is only stored as tuples for saving it
    data = []; done = True
    for _ in range(n):
        if done: s = env.init()
        s_, done = env.step(s, a:=np.random.randint(nactions))
        data.append((s,a,done,s:=s_))
    # save s, a, c, s_
    np.savetxt(str:=dt.now().strftime(f'data/{n}-%H.%M.%S-%d.%m'),
               [np.concatenate((s,[a,c],s_)) for (s,a,c,s_) in data])
    # prepare data for fqi
    nfeatures = features(data[0][0]).size
    inputs = np.zeros((n, nactions*nfeatures))
    cs = np.zeros(n)
    s_s = np.zeros((n,nfeatures))
    for i, (s,a,c,s_) in enumerate(data):
        inputs[i, a*nfeatures:(a+1)*nfeatures] = features(s)
        cs[i] = c
        s_s[i] = features(s_)
    return inputs, cs, s_s
