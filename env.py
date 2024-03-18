import numpy as np
from numpy import pi as pi

#def fourier(inputs, order)

# acrobot states are numpy 4 vectors, actions in [0,1,2]
# initialization in [-0.1,0.1)**4 as in gyn
def acrobotinit():
    return np.random.random(4)*.2-.1

# use a timestep of 0.2 like gym (book pg 271 does 4 steps at 0.05)
# rn reward is 1 for being at height of 1 above pivot
def acrobotstep(s, a):
    def dynamics(s):
        t1, t2, dt1, dt2 = s
        d1 = cos(t2)+3.5
        d2 = cos(t2)/2 + 1.25
        phi2 = 4.9*cos(t1 + t2 - pi/2)
        phi1 = -(dt2/2+dt1)*dt2*sin(t2) + 14.7*cos(t1 - pi/2) + phi2
        ddt2 = ((a-1) + d2/d1*phi1 - dt1**2*sin(t2)/2 - phi2) / (1.25 - d2**2 / d1)
        ddt1 = -(d2*ddt2 + phi1) / d1
        return np.array([dt1,dt2,ddt1,ddt2])
    def rk4(x, h, dx=dynamics):
        g1=dx(x); g2=dx(x+(h/2)*g1); g3=dx(x+(h/2)*g2); g4=dx(x+h*g3)
        return x + h*(g1/6 + g2/3 + g3/3 + g4/6)
    s=rk4(s,.2);s[:2]%=2*pi
    np.clip(s,[0,0,-4*pi,-9*pi],[2*pi,2*pi,4*pi,9*pi],s)
    return s, int(-np.cos(s[0]) - np.cos(s[0]+s[1]) >= 1)
