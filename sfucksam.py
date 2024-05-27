from envs import invpendulum as env
from features import basis, fourier
from data import data
from fqi import fqi
from eval import eval
from plot import plot

import datetime

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

# experiment
NJOBS = 1 # parallelism
GAMMA = 0.95
ROUNDS = 400 # rounds for fqi
NS = [2000 * (i+1) for i in range(5)] # input data sizes
MAXEVALSTEPS = 3000
NRUNS = 9 # how many models to check for each n value (should be a multiple/divisor of NJOBS)
ORD = 4 # order of basis
# environment
MIN = np.array([-np.pi/2,-5])
SPREAD = np.array([np.pi,10])
NACTIONS = 3

def exp(i):
    # collect dataset
    b = basis(MIN.size, ORD)
    features = lambda x: fourier(x, b, MIN, SPREAD)
    inputs, cs, s_s = data(NS[i//NRUNS], env, NACTIONS, features)
    # fit models
    w = fqi('log', inputs, cs, s_s, GAMMA, ROUNDS)
    log, _ = eval(w, env, features, MAXEVALSTEPS)
    w = fqi('sqr', inputs, cs, s_s, GAMMA, ROUNDS)
    sqr, _ = eval(w, env, features, MAXEVALSTEPS)
    return log, sqr

if __name__=='__main__':
    log, sqr = np.zeros((len(NS),NRUNS)), np.zeros((len(NS),NRUNS))
    now = datetime.datetime.now()
    print('Start Time: ', now.time())
    results = Parallel(n_jobs=NJOBS)(delayed(exp)(i)
                                     for i in tqdm(range(log.size)))
    for i, (l,s) in enumerate(results):
        log[i//NRUNS,i%NRUNS] = l
        sqr[i//NRUNS,i%NRUNS] = s
    plot(NS, log, sqr, 'plot.pdf')

# for stochastic environments:
'''
results, lens = eval(w, env, features, MAXEVALSTEPS, EVALCOUNT)
resultslog[i][j] = sum(results)/len(results)
print('log ', lens)
w = fqi('sqr', inputs, cs, s_s, GAMMA, ROUNDS)
results, lens = eval(w, env, features, MAXEVALSTEPS, EVALCOUNT)
resultssq[i][j] = sum(results)/len(results)
print('sqr ', lens)
'''
