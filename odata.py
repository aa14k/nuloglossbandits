import numpy as np
from datetime import datetime as time
from env import *
import gym

# function that collects a (fixed horizon) dataset
# assumes action space is [0..nactions]
def getdata(env,size,nsuc,H):
    init,step,nactions=[(acrobotinit,acrobotstep,3),
                        (mountaincarinit,mountaincarstep,3)][env]
    fs, ss = [], []
    i = 0#
    while len(fs)<size-nsuc or len(ss)<nsuc:
        t = []; s = init()
        for h in range(H):
            a = np.random.randint(nactions)
            s,r = step(s,a,h,H)
            t.append((s,a,r))
        if len(fs)<size-nsuc and not r: fs.append(t)
        elif len(ss)<nsuc and r: ss.append(t)
        print(len(ss),len(fs),i:=i+1)
    return np.array(ss+fs)

# gets a cartpole dataset by running trajectories
# size is number of timesteps
def getdatacartpoletraj(size):
    data=np.zeros((size,10))
    s=cartpoleinit()
    for t in range(size):
        data[t,:4]=s
        data[t,4]=(a:=np.random.randint(2))
        s,r=cartpolestep(s,a)
        data[t,5],data[t,6:]=r,s
        #if r!=0: traj=False
        if r!=0: s=cartpoleinit()
    np.save(time.now().strftime(f'data/{size}-%H.%M.%S-%d.%m'), data)


# why?
# def getdatacartpolegym(size):
#     env = gym.make('CartPole-v1')
#     data=np.zeros((size,10))
#     s,_ = env.reset()
#     traj = True
#     for t in range(size):
#         a = env.action_space.sample()
#         s_,r,terminated,_,_ = env.step(a)
#         print(s,a,r,s_)
#         data[t,:4], data[t,4], data[t,5],data[t,6:] = s, a, r, s_
#         if terminated: s = env.reset()
#     np.save(time.now().strftime(f'data/{size}-%H.%M.%S-%d.%m'), data)
