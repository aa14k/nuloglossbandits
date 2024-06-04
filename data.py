import numpy as np
from datetime import datetime as dt

def data(n, env, nactions, features):
    # sample data
    # NB. the data is only stored as tuples for saving it
    data = []; done = True
    for _ in range(n):
        if done: s = env.init()
        s_, c, done = env.step(s, a:=np.random.randint(nactions))
        data.append((s,a,c,s:=s_))
    # save s, a, c, s_
    np.savetxt(str:=dt.now().strftime(f'data/invpen-{n}-%H.%M.%S-%d.%m'),
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

# todo make this less shitty
# note that this is working with n*H experiences
def data_mountaincar(n, env, nactions, features, H, nsucc=1, width=100, name=''):
    succs = []
    fails = []
    # collect trajectories
    while len(succs) < nsucc*H or len(fails)<(n-nsucc)*H:
        traj=[]
        s = env.init(width)
        for h in range(H):
            a = np.random.randint(nactions,size=width)
            s_, c, _ = env.step(s, a, h+1, H)
            traj.append((s,a,c,s:=s_))
        succmask = c==0
        for i in range(width):
            store = succs if succmask[i] else fails
            for (s,a,c,s_) in traj: store.append((s[i],a[i],c[i],s_[i]))
        succs = succs[:nsucc*H] # naive lack of short circuit is slow
        fails = fails[:(n-nsucc)*H]
    data = succs + fails
    # save s, a, c, s_
    np.savetxt(dt.now().strftime(f'data/mountc-{name}-{n}-%H.%M.%S-%d.%m'),
               [np.concatenate((s,[a,c],s_)) for (s,a,c,s_) in data], '%.4e')
    # prepare data for fqi
    nfeatures = features(data[0][0]).size
    inputs = np.zeros((n*H, nactions*nfeatures))
    cs = np.zeros(n*H)
    s_s = np.zeros((n*H,nfeatures))
    for i, (s,a,c,s_) in enumerate(data):
        inputs[i, a*nfeatures:(a+1)*nfeatures] = features(s)
        cs[i] = c
        s_s[i] = features(s_)
    return inputs, cs, s_s

