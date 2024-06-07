import numpy as np
PI = np.pi

# state space RR^2
# action space [0,1,2]

def init(): return np.random.uniform(-PI/180, PI/180, 2) * 5.0

def step(s, a):
    F=50;       NOISE=10
    MPEND=2.0;  MCART=8.0
    L=0.5;      G=9.8;     DT=0.1
    th, dth = s
    u = F*(a-1) + NOISE*np.random.uniform(-1,1)
    acc = 1/(MCART + MPEND)
    ddth = G*np.sin(th) -acc*(dth**2*MPEND*L*np.sin(2*th)/2+np.cos(th)*u)
    ddth /= L*(4/3 - acc*MPEND*np.cos(th)**2)
    dth += DT*ddth; th += DT*dth
    # space dims: -pi/2 to pi/2, -5 to 5 (but og was -inf to inf)
    terminal = np.abs(th - 0.00001) >= (np.pi / 2)
    cost = 0.9999 if terminal else 1e-8
    return np.clip([th,dth],[-PI/2,-5],[PI/2,5]),cost,terminal
