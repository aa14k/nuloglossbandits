import numpy as np

# everything is vectorised by width, works with 1 but has dim=2
# state space [-.07,.07] x [-1.2,.5]
# action space [0,1,2]

# gym version would initialise in (-.6,-.4) instead of (-.5,-.5)
def init(width=1):
    return np.vstack((np.random.uniform(-.5,-.5,width) * 0.0 - 0.5,
                      np.zeros(width))).T

# horizon is 1-indexed
def step(s, a, h, H):
    pos = s[:,0]
    vel = s[:,1]
    vel = vel + (a-1)*0.001 + (-0.0025) * np.cos(3*pos)
    vel = np.clip(vel,-0.07,0.07)

    pos = np.where(pos>=0.6, 0.6, np.clip(pos+vel,-1.2,0.6))
    return np.vstack((pos,vel)).T, (pos<.6).astype(np.float32)*(h==H-1)
