from envs import invpendulum as env
from features import basis, fourier
from eval import eval
from nufqi import fqi
import matplotlib.pyplot as plt
import numpy as np
PI = np.pi

# experiment
GAMMA = 0.95
ROUNDS = 100 # rounds for fqi
NS = [500 * (i+1) for i in range(10)] # input data sizes
EVALCOUNT = 20 # how many trials to evaluate performance
MAXEVALSTEPS = 3000
MODELCOUNT = 5 # how many models to check for each n value
ORD = 3
# environment
MIN = np.array([-PI/2,-5])
SPREAD = np.array([PI,10])
ACTIONSIZ = 3
DIM = 2

resultslog = np.zeros((len(NS),MODELCOUNT))
resultssq = np.zeros((len(NS),MODELCOUNT))
for i, n in enumerate(NS):
    print(n)
    for j in range(MODELCOUNT):
        # collect dataset
        # NB the data is only stored as tuples for saving it
        data = []; done = True
        for _ in range(n):
            if done: s = env.init()
            s_, done = env.step(s, a:=np.random.randint(ACTIONSIZ))
            data.append((s,a,done,s_))
            s = s_
        # fit model
        b = basis(DIM, ORD)
        features = lambda x: fourier(x, b, MIN, SPREAD)
        fsize = (ORD+1)**DIM
        inputs = np.zeros((n, ACTIONSIZ*fsize))
        cs = np.zeros(n)
        s_s = np.zeros((n,fsize))
        for k, (s,a,c,s_) in enumerate(data):
            fourier(s, b, MIN, SPREAD)
            inputs[k,a*fsize:(a+1)*fsize] = features(s)
            cs[k], s_s[k] = c, features(s_)
        w = fqi('log', inputs, cs, s_s, GAMMA, ROUNDS)
        results, lens = eval(w, env, features, MAXEVALSTEPS, EVALCOUNT)
        resultslog[i][j] = sum(results)/len(results)
        print('log ', lens)
        w = fqi('sqr', inputs, cs, s_s, GAMMA, ROUNDS)
        results, lens = eval(w, env, features, MAXEVALSTEPS, EVALCOUNT)
        resultssq[i][j] = sum(results)/len(results)
        print('sqr ', lens)
        # save results

plt.xlabel('Training data size (timesteps)')
plt.ylabel('Average success rate')#r'$\bar{c} \pm 2\sigma$')
plt.plot(NS, resultslog.mean(1), '.--', label='Log Loss')
plt.plot(NS, resultssq.mean(1), '.--', label='Squared Loss')
plt.legend()
plt.savefig('out.png')
