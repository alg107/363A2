import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

plt.style.use('Solarize_Light2')

default_params = {
        "gamma": 0.1,
        "InfDays": 10,
        "R0": [0.5, 1.0, 2.0, 3.0, 4.0],
        "intervals": [0,50,120,200],
        "y0": [1.0,0.0,0.0],
        "import_packet": 6e-7 # 3 of 5,000,000
        }


def sir(y,t,beta, gamma):
    S,I,R=y
    #betan = betan*bm_samples[int(np.floor(tn*10))]
    dydt = [
            #(np.sin(t)*0.1+1)
            -beta*I*S,
            beta*I*S - gamma*I,
            gamma*I
            ]
    return dydt

def sirv(y,t,beta, gamma):
    # Unfinished
    S,I,R=y
    dydt = [
            -beta*I*S-b*S+b*(1-m)*(S+R),
            beta*I*S - (pd+r)*I,
            r*I-b*R+mb*(S+R)
            ]
    return dydt

def runsir(params, ti, tf, fn):
    gamma, R0, InfDays, y0 = params
    beta = R0/InfDays
    t = np.linspace(ti,tf,1000)
    sol = odeint(fn, y0, t, args=(beta, gamma))
    return t, sol

def gen_intervals(tf):
    pass

def runsim(p, fn):
    gamma = p["gamma"]
    R0 = p["R0"]
    InfDays = p["InfDays"]
    intervals = p["intervals"]
    y0 = p["y0"]
    import_packet = p["import_packet"]
    
    for R in R0:
        final_t = np.array([])
        final_sol = np.array([])
        y0_int = np.copy(y0)
        for i, interval in enumerate(intervals[:-1]):
            params = [gamma, R, InfDays, y0_int]
            t, sol = runsir(params,intervals[i], intervals[i+1], fn)
            final_t = np.concatenate((final_t, t)) if final_t.size else t
            final_sol = np.concatenate((final_sol, sol)) if final_sol.size else sol
            y0_int = sol[-1]
            y0_int[1]+=import_packet # Imported COVID
            y0_int = y0_int/np.sum(y0_int) #Normalise
        plt.figure()
        plt.plot(final_t, final_sol[:,0])
        plt.plot(final_t, final_sol[:,1])
        plt.plot(final_t, final_sol[:,2])
        plt.legend(["S","I","R"])

runsim(default_params, sir)
plt.show()

