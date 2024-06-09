import numpy as np

# everything is vectorised by width, works with 1 but has dim=2
# state space [-.07,.07] x [-1.2,.5]
# action space [0,1,2]

# gym version would initialise in (-.6,-.4) instead of (-.5,-.5)
def init():
    return np.array([-.5, 0])

def step(s, a):
    F=.001;      G=.0025
    p, dp = s
    if p == -1.2: dp = 0
    dp += (a-1)*F - np.cos(3*p)*G
    if p >= .5: dp = 0
    dp = np.clip(dp,-.07,.07)
    p = np.clip(p + dp, -1.2, .6)
    return np.array([p,dp]), -float(p>=.5), p>=.5
