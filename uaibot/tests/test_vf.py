import uaibot as ub
import numpy as np
from utils_test import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') 

curve = []
Npoints = 2000
dt = 0.01
Tmax = 6

for i in range(Npoints):
    s = 2*np.pi*i/(Npoints-1)
    x = np.cos(s)+0.2*np.sin(2*s)
    y = np.sin(s)+0.1*np.cos(3*s)
    # curve.append(np.matrix([x,y]).T)
    if i%2==0:
        curve.append([x,y])
    else:
        curve.append(np.array([x,y]))
    
    
q = np.matrix([0.5,0.]).T
hist_q = []

for k in range(round(Tmax/dt)):
    
    hist_q.append(np.matrix(q))
    dotq, dist, _ = ub.Robot.vector_field(q,curve, alpha = 2, mode='python')
    
    q+= dotq*dt

    
    
plt.plot([ub.Utils.cvt(qh)[0,0] for qh in curve], [ub.Utils.cvt(qh)[1,0] for qh in curve],color='red')
plt.plot([qh[0,0] for qh in hist_q], [qh[1,0] for qh in hist_q])

plt.show()
    


    
    

    
    





    