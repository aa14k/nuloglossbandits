from envs import mountaincar as env
from features import basis, fourier
from data import mcdata2 as data
from mcfqi import fqi
from eval import mc_eval as eval
from plot import plot

from joblib import Parallel, delayed
import numpy as np

import datetime


# experiment
NJOBS = 15 # parallelism (15 should be memory safe)
NS = [9000, 21000, 30000]
# [1000, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]
NRUNS = 15 # how many models to check for each n value
ORD = 6 # order of basis
# environment
NSUCC = 5 # 1, 5, 30
H = 800
MIN = np.array([-.07,-1.2])
SPREAD = np.array([.14,1.7])
NACTIONS = 3

def exp(i):
    # collect dataset
    b = basis(MIN.size, ORD)
    features = lambda x: fourier(x, b, MIN, SPREAD)
    ss,a_s,cs,s_s = data(NS[i//NRUNS], env, NACTIONS, features, NSUCC, H)
    # fit models
    w_log = fqi('log', ss, a_s, cs, s_s, H)
    log = eval(w_log, env, features, H)
    # w_sq = fqi('sqr', ss, a_s, cs, s_s, H)
    # sqr = eval(w_sq, env, features, H)
    # print(f'{cs.size/H}:', log, sqr)
    print(f'{cs.size/H}:', log)
    # print('Job ' + str(i) + ' completed')
    return log

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
