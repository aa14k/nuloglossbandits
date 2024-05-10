import numpy as np
import os
from env import *
from data import *
from fqi import *


if __name__=='__main__':
    # getdatacartpoletraj(2000000)
    print('data/'+os.listdir('data')[0])
    D=np.load('data/'+os.listdir('data')[0])
    print('fitting')
    th = fqilog(D,4,10000,llog,dllog)
    #th = fqi_sq(D,4,10000,lsq,dlsq)
    
    print('eval')
    print(eval(th,1000))
