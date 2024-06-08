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

# n is output data size
# ns is number of successes
# H is horizon
# w is simulation width
# get rid of 1-index bs later
def mcdata(n, env, nactions, features, ns, H):
    sidx, fidx = 0, ns
    data = np.zeros((H,n,6))
    while sidx < ns or fidx < n:
        d = np.zeros((H,n,6))
        s = env.init(n)
        for h in range(1, H+1):
            s_, c = env.step(s,a:=np.random.randint(nactions,size=n),h,H)
            d[h-1] = np.concatenate((s,np.vstack((a,c)).T,s:=s_),1)
        succ = d[:,d[H-1,:,3]==0][:,:ns-sidx]
        fail = d[:,d[H-1,:,3]==1][:,:n-fidx]
        data[:,sidx:sidx+succ.shape[1]] = succ; sidx += succ.shape[1]
        data[:,fidx:fidx+fail.shape[1]] = fail; fidx += fail.shape[1]
    # prepare data for fqi
    # at each step indices stores which data to use for a given action
    featsize = features(data[0][0][:2]).size
    ss = np.zeros((H, n, featsize))
    a_s = np.zeros((H,n))
    cs = np.zeros((H, n))
    s_s = np.zeros((H, n,featsize))
    # indices = np.zeros((H,nactions,n), dtype=bool)
    for h in range(H):
        ss[h] = features(data[h,:,:2])
        a_s[h] = data[h,:,2]
        cs[h] = data[h,:,3]
        s_s[h] = features(data[h,:,-2:])
        # for a in range(nactions): indices[h,a] = data[h,:,2]==a
    return ss, a_s, cs, s_s

def mcdata2(n, env, nactions, features, ns, H):
    data = np.zeros((H,n,6))
    while True:
        data = np.zeros((H,n,6))
        s = env.init(n)
        for h in range(1, H+1):
            s_, c = env.step(s,a:=np.random.randint(nactions,size=n),h,H)
            data[h-1] = np.concatenate((s,np.vstack((a,c)).T,s:=s_),1)
        if (data[H-1,:,3]==0).sum() >= ns:
            break
    # print(data.shape, sum(data[H-1,:,3]))
    # prepare data for fqi
    featsize = features(data[0][0][:2]).size
    ss = np.zeros((H, n, featsize))
    a_s = np.zeros((H,n))
    cs = np.zeros((H, n))
    s_s = np.zeros((H, n,featsize))
    # indices = np.zeros((H,nactions,n), dtype=bool)
    for h in range(H):
        ss[h] = features(data[h,:,:2])
        a_s[h] = data[h,:,2]
        cs[h] = data[h,:,3]
        s_s[h] = features(data[h,:,-2:])
        # for a in range(nactions): indices[h,a] = data[h,:,2]==a
    return ss, a_s, cs, s_s
        
    
