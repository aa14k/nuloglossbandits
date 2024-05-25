from envs import invpendulum as env
from features import basis, fourier
from data import data
from fqi import fqi
from eval import eval
from plot import plot
from joblib import Parallel, delayed
import numpy as np

# experiment
NJOBS = 38 # parallelism
GAMMA = 0.95
ROUNDS = 100 # rounds for fqi
NS = [500 * (i+1) for i in range(10)] # input data sizes
MAXEVALSTEPS = 3000
NMODELS = 90 # how many models to check for each n value
ORD = 3 # order of basis
# environment
MIN = np.array([-np.pi/2,-5])
SPREAD = np.array([np.pi,10])
NACTIONS = 3

def exp(i):
    for i, n in enumerate(NS):
        print(n)
        for j in range(NMODELS):
            # collect dataset
            b = basis(MIN.size, ORD)
            features = lambda x: fourier(x, b, MIN, SPREAD)
            inputs, cs, s_s = data(n, env, NACTIONS, features)
            # fit model
            w = fqi('log', inputs, cs, s_s, GAMMA, ROUNDS)
            log[i][j] = eval(w, env, features, MAXEVALSTEPS)
            w = fqi('sqr', inputs, cs, s_s, GAMMA, ROUNDS)
            sqr[i][j] = eval(w, env, features, MAXEVALSTEPS)
            # save results
            plot(NS, log, sqr, 'out.png')

if __name__=='__main__':
    log, sqr = np.zeros((len(NS),NMODELS)), np.zeros((len(NS),NMODELS))
    for i in range(-(len(NS)*NMODELS//-NJOBS)  # todo1 parallelism
    exit()
    # Parallel(n_jobs=40)(delayed(run_experiment)(H, num_trials, phi, d, num_success) for j in tqdm(range(runs)))

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
