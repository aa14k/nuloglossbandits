from scipy.optimize import minimize, least_squares
from itertools import product
from numpy import zeros, exp, log, sum, cos, min, clip, array, arange, pi
from scipy.special import expit as sigmoid
import numpy as np
from env import *
from tqdm import tqdm


def makebasis(statedim, order):
    return array(list(product(arange(order+1), repeat=statedim)))

def fourierfeat(states, basis, min=array([-4.8,-5,-.418,-5]),
                spread=array([9.6,10,.836,10])):
    return cos(pi*(states-min)/spread@basis.T)# scaling

#def sigmoid(x): return 1/(1+(1+1e-7)*exp(clip(-x,-36,36)))


def lsq(th,x,y):
    y_pred = sigmoid(x@th)
    return np.sum(y_pred-y)

def dlsq(th,x,y):
    p = sigmoid(x @ th)
    scalar = 2 * (p - y) * (p * (1-p))
    return scalar.T@x # why would there be a transpose here?

def llog(th,x,y):
    # bce = keras.losses.BinaryCrossentropy(from_logits=True)
    # return bce(y,x@th)*len(y) 

    p=sigmoid(x@th)#clip(sigmoid(x@th),1e-16,1-1e-16)
    return -sum(y*log(p) + (1-y)*log(1-p))


def dllog(th,x,y):# todo check signs (why would there be a transpose here?)
    return (sigmoid(x@th)-y).T@x

# def hllog(th,x,y):
#     p = sigmoid(x@th)
#     return  (p*(1-p)*x.T)@x


def eval(th,n,gamma=.95):
    basis = makebasis(4,4)
    cost=np.zeros(n)
    s = cartpoleinit()
    for i in range(n):
        #a=th@fourierfeat(s,basis)
        s,c=cartpolestep(s,np.argmin(th@fourierfeat(s,basis)))
        if c!=0: return c*gamma**(i), i+1


# data, statedim, iters, loss func, loss deriv, actionsize, order, discount
# l and dl should take (theta,x,y)
def fqilog(D,d,k,l,dl,A=2,o=4,gamma=.95):
    print('fqi-log')
    basis = makebasis(d,o)
    ss=D[:,:d]; As=D[:,d]; rs=D[:,-d]; Ss=D[:,-d:]#(s,a,r,s') in each row
    ss=fourierfeat(ss,basis); Ss=fourierfeat(Ss,basis)
    exit()
    th=zeros((A,(o+1)**d))
    for i in tqdm(range(k)):
        if not i%10: print(f'eval: {i}', eval(th,1000))#
        target=clip(rs + gamma*min(sigmoid(th@Ss.T), axis=0),0,1)
        for a in range(A):
            sol=minimize(fun=l,
                           x0=zeros(d),# x0=th[a],
                           args=(ss[As==a],target[As==a]),
                           method='newton-cg',
                           # hess = hl,
                           jac=dl)
            th[a] = sol.x
    return th


# this is the bad kind of duplicate code
# def fqi_sq(D,d,k,l,dl,A=2,o=4,gamma=.99):
#     print('fqi-sq')
#     basis = makebasis(d,o)
#     ss=D[:,:d]; As=D[:,d]; rs=D[:,-d]; Ss=D[:,-d:]#(s,a,r,s') in each row
#     ss=fourierfeat(ss,basis); Ss=fourierfeat(Ss,basis)
#     th=zeros((A,(o+1)**d))
#     # target=zeros(ss.size)
#     for i in tqdm(range(k)):
#         if not i%5: print(f'eval: {i}', eval(th,1000))#
#         target=clip(rs + gamma*min(th@Ss.T, axis=0),0,1)
#         for a in range(A):
#             #sol=least_squares(fun=l,
#             #               x0=zeros(len(th[a])),# x0=th[a],
#             #               args=(ss[As==a],target[As==a]),
#             #               jac = dl)
#             #th[a] = sol.x
#             # 'OLS: Linear model'
#             features = ss[As==a]
#             th[a] = np.linalg.solve(features.T @ features, features.T @ target[As==a])
#             print(np.linalg.norm((features @ th[a] - target[As==a]).T@features))
            
            
#     return th
