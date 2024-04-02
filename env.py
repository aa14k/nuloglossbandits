import numpy as np
from numpy import pi, cos, sin, clip

#def fourier(inputs, order)
# acrobot states are numpy 4 vectors, actions in [0,1,2]
# initialization in [-0.1,0.1)**4 as in gym
def acrobotinit():
    return np.random.random(4)*.2-.1

# use a timestep of 0.2 like gym (book pg 271 does 4 steps at 0.05)
# rn reward is 1 for being at height of 1 above pivot
def acrobotstep(s, a, h, H):
    goal = lambda s: int(-cos(s[0]) - cos(s[0]+s[1]) >= 1)
    if goal(s): return s, int(h==H)
    def dynamics(s):
        t1, t2, dt1, dt2 = s
        d1=cos(t2)+3.5
        d2=cos(t2)/2 + 1.25
        phi2=4.9*cos(t1+t2-pi/2)
        phi1=-(dt2/2+dt1)*dt2*sin(t2)+14.7*cos(t1-pi/2)+phi2
        ddt2=((a-1)+d2/d1*phi1-dt1**2*sin(t2)/2-phi2)/(1.25-d2**2/d1)
        ddt1=-(d2*ddt2+phi1)/d1
        return np.array([dt1,dt2,ddt1,ddt2])
    def rk4(x, h, dx=dynamics):
        g1=dx(x); g2=dx(x+h/2*g1); g3=dx(x+h/2*g2); g4=dx(x+h*g3)
        return x + h*(g1/6 + g2/3 + g3/3 + g4/6)
    s=rk4(s,.2);s[:2]%=2*pi
    clip(s,[0,0,-4*pi,-9*pi],[2*pi,2*pi,4*pi,9*pi],s)
    return s, goal(s)*(h==H)

# todo either this freezes in place on success or it requires the agent to stay upright through to the end

def mountaincarinit():
    return (np.random.random(2)/5 - .6) * [1,0]

def mountaincarstep(s, a, h, H):
    if s[0]==.5: return s, int(h==H)
    elif s[0]==-1.2: s[1]=0
    s=clip([s.sum(),s[1]+.001*a-.0025*cos(3*s[0])],[-1.2,-.07],[.5,.07])
    return s, (s[0]==.5)*(h==H)
    # x := clip(x + x', -1.2, .5)
    # x' := clip(x' + .001a - .0025cos(3x),-.07,.07)
    # hit left and reset velocity to 0
    # hit right and stay

# todo from http://incompleteideas.net/book/code/pole.c
def cartpoleinit():
    pass

def cartpolestep(s, a):
    pass
