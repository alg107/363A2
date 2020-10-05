import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from stochastic.noise import BrownianNoise
bm = BrownianNoise()

#bm_samples=bm.sample(2000)*15+1
#plt.plot(bm_samples)
plt.show()

InfDays = 10;
R0 = [0.5, 0.9, 1.5]

def sir(yn,tn,betan, gamman):
    S,I,R=yn
    #betan = betan*bm_samples[int(np.floor(tn*10))]
    dydt = [
            -betan*I*S,
            betan*I*S - gamman*I,
            gamman*I
            ]
    return dydt

def sirv(y,t,beta, gamma):
    S,I,R=y
    dydt = [
            -beta*I*S-b*S+b*(1-m)*(S+R),
            beta*I*S - (pd+r)*I,
            r*I-b*R+mb*(S+R)
            ]
    return dydt

for R in R0:

    y0 = [1.0,0.05,0.0]
    beta = R/InfDays
    t = np.linspace(0,100,1000)
    sol = odeint(sir, y0, t, args=(beta, 0.1))
    plt.plot(t, sol[:,1])
    #plt.plot(t, sol[:,1])
    #plt.plot(t, sol[:,2])
plt.show()

