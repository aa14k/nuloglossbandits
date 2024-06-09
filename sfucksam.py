from envs import discountmountaincar as env
from features import basis, fourier
from data import mcdataconc as data
from afqi import fqi
from eval import eval
from plot import plot

from joblib import Parallel, delayed
import numpy as np

import datetime


# experiment
NJOBS = 45 # parallelism
GAMMA = 0.95
ROUNDS = 600 # rounds for fqi
NS = [5000, 10000, 20000] # input data sizes
MAXEVALSTEPS = 2000
NRUNS = 9 # how many models to check for each n value
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
    w_log = fqi('log', inputs, cs, s_s, GAMMA, ROUNDS)
    log, hlog = eval(w_log, env, features, MAXEVALSTEPS)
    # w_sq = fqi('sqr', inputs, cs, s_s, GAMMA, ROUNDS)
    # sqr, hsqr = eval(w_sq, env, features, MAXEVALSTEPS)
    print(hlog, cs.size)#
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
    plot(NS, log, sqr, 'plot_order_4.pdf')

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
