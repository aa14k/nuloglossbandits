from envs import mountaincar as env
from features import basis, fourier
from data import data_mountaincar as data
from afqi import fqi
from eval import mc_eval as eval
from plot import plot

from joblib import Parallel, delayed
import numpy as np

import datetime


# experiment
NJOBS = 15 # parallelism (15 should be memory safe)
GAMMA = 1 # mc with 1
ROUNDS = 800 # rounds for fqi
NS = [1000, 2000, 3000, 4000, 5000]#, 15000]
      # 18000, 21000, 24000, 27000, 30000] # input data sizes
MAXEVALSTEPS = 800 # should match H
NRUNS = 5 # how many models to check for each n value
ORD = 4 # order of basis
# environment
WIDTH = 10000
NSUCC = 5 # 1, 5, 30
H = 800
MIN = np.array([-.07,-1.2])
SPREAD = np.array([.14,1.7])
NACTIONS = 3

def exp(i):
    # collect dataset
    b = basis(MIN.size, ORD)
    features = lambda x: fourier(x, b, MIN, SPREAD)
    inputs, cs, s_s = data(NS[i//NRUNS], env, NACTIONS, features, H, NSUCC, WIDTH,i)
    # fit models
    w_log = fqi('log', inputs, cs, s_s, GAMMA, ROUNDS)
    log, _ = eval(w_log, env, features, MAXEVALSTEPS)
    w_sq = fqi('sqr', inputs, cs, s_s, GAMMA, ROUNDS)
    sqr, _ = eval(w_sq, env, features, MAXEVALSTEPS)
    print(f'{cs.size/H}:', log, sqr)
    print('Job ' + str(i) + ' completed')
    return log, sqr

if __name__=='__main__':
    log, sqr = np.zeros((len(NS),NRUNS)), np.zeros((len(NS),NRUNS))
    now = datetime.datetime.now()
    print('Start Time: ', now.time())
    results = Parallel(n_jobs=NJOBS)(delayed(exp)(i)
                                     for i in range(log.size))
    print('Done')
    for i, (l,s) in enumerate(results):
        log[i//NRUNS,i%NRUNS] = l
        sqr[i//NRUNS,i%NRUNS] = s
    plot(NS, log, sqr, 'mc_plot_order_4.pdf')

# # for stochastic environments:
# '''
# results, lens = eval(w, env, features, MAXEVALSTEPS, EVALCOUNT)
# resultslog[i][j] = sum(results)/len(results)
# print('log ', lens)
# w = fqi('sqr', inputs, cs, s_s, GAMMA, ROUNDS)
# results, lens = eval(w, env, features, MAXEVALSTEPS, EVALCOUNT)
# resultssq[i][j] = sum(results)/len(results)
# print('sqr ', lens)
# '''
