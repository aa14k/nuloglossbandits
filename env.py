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
    if goal(s): return s, int(h==H) # currently this freezes in place upon a success
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

# everything is vectorised by width, works with 1 but has dim=2
# the done signal is scalar
def mountaincarinit(width=1):
    return np.vstack((np.random.uniform(-.6,-.4,width), np.zeros(width))).T
    # return (np.random.random(2)/5 - .6) * [1,0]

def mountaincarstep(s, a, h, H=800):
    F=.001;      G=.0025
    dp, p = s.T
    dp = np.clip((dp*(p!=-1.2) + (a-1)*F - np.cos(3*p)*G)*(p!=.5),
                 -.07,.07)
    p = np.clip(p + dp, -1.2, .5)
    s = np.vstack((dp,p)).T
    return s, (p==.5).astype(np.float32)*(h==H), (h==H)
    # if s[0]==.5: return s, int(h==H)
    # elif s[0]==-1.2: s[1]=0
    # s=clip([s.sum(),s[1]+.001*a-.0025*cos(3*s[0])],[-1.2,-.07],[.5,.07])
    # return s, (s[0]==.5)*(h==H)
    # x := clip(x + x', -1.2, .5)
    # x' := clip(x' + .001a - .0025cos(3x),-.07,.07)
    # hit left and reset velocity to 0
    # hit right and stay
    # A = [0, 1, 2], H = 800

# http://incompleteideas.net/book/code/pole.c
def cartpoleinit():
    return np.zeros(4) # gym does np.random.random(4)*.1 - .05

def cartpolestep(s, a):
    done = lambda s: abs(s[0]) >= 4.8 or abs(s[2]) >= .418
    if done(s): return s, 0# don't reset
    x, dx, t, dt = s
    f = 10*(2*a-1) # a in {0,1}
    g = 9.8; mc = 1; mp = .1; l = .5; tau = .02
    temp = (f + (mp*l)*dt**2*sin(t))/(mp+mc)
    thetaacc=(g*sin(t) - cos(t)*temp)/(l*(4/3 - mp * cos(t)**2/(mp+mc)))
    xacc = temp - mp*l*thetaacc*cos(t)/(mp+mc)
    # update
    x += tau*dx
    dx += tau*xacc
    t += tau*dt
    dt += tau*thetaacc
    # end
    f32max = np.finfo(np.float32).max
    bound=np.array([4.8,f32max,2*pi,f32max],np.float32)
    s = clip([x,dx,t,dt],-bound,bound)
    return s, float(done(s))


'''
cart pos is in -+4.8
angle is in -+.418 rads
pos,dpos,ang,dang
velocities can by -+inf

the "epsiode terminates"
if pos is not in -+2.4
if ang is not in -+.2095

gym starts w/ all obs in -+.05, so we'll start at *0*

'''
