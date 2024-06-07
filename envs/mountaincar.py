import numpy as np

# everything is vectorised by width, works with 1 but has dim=2
# state space [-.07,.07] x [-1.2,.5]
# action space [0,1,2]


def init(width=1):
    return np.vstack((np.random.uniform(-.6,-.4,width),
                      np.zeros(width))).T

# the done signal is scalar
def step(s, a, h, H=800):
    F=.001;      G=.0025
    p, dp = s.T
    dp = np.clip((dp*(p!=-1.2) + (a-1)*F - np.cos(3*p)*G)*(p!=.5),
                 -.07,.07)
    p = np.clip(p + dp, -1.2, .5)
    return np.vstack((p,dp)).T, (p!=.5).astype(np.float32)*(h==H), (h==H)
