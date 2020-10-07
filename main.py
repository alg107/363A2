import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import UnivariateSpline

plt.style.use('Solarize_Light2')
from scipy.stats import norm
from scipy import signal

vaccine_functions = [
         lambda x: 0.5*(signal.square(x/(2*np.pi)/3)+1)*0.002*(1/4)
        ]



default_params = {
        "gamma": 0.1,
        "nu": (1/(30*4)),
        "InfDays": 10,
        "R0": [0.5, 1.0, 2.0, 3.0, 4.0],
        "intervals": [0,50,120,200, 300, 400, 450, 500],
        "y0": [1.0,0.0,0.0,0.0],
        "import_packet": 6e-7 # 3 of 5,000,000
        }


def sir(y,t,beta, gamma, spl):
    S,I,R=y
    #betan = betan*bm_samples[int(np.floor(tn*10))]
    dydt = [
            #(np.sin(t)*0.1+1)
            -beta*spl(t)*I*S,
            beta*spl(t)*I*S - spl(t)*gamma*I,
            spl(t)*gamma*I
            ]
    return dydt

def sirv(y,t,beta, gamma, nu, spl, vaccf):
    S,I,R,V=y
    #betan = betan*bm_samples[int(np.floor(tn*10))]
    
    dydt = [
            #(np.sin(t)*0.1+1)
            -beta*spl(t)*I*S -vaccf(t)*S + nu*V,
            beta*spl(t)*I*S - spl(t)*gamma*I,
            spl(t)*gamma*I,
            vaccf(t)*S-nu*V 
            ]
    return dydt


def runsir(params, ti, tf, fn, vaccf):
    gamma, nu, R0, InfDays, y0 = params
    beta = R0/InfDays
    t = np.linspace(ti,tf,1000)
    sol = odeint(fn, y0, t, args=(beta, gamma, nu, spl, vaccf))
    return t, sol

def gen_intervals(tf):
    pass

def gen_noisefn(length, pnts=1000):
    # Process parameters
    delta = 0.2
    dt = 0.1

    x = 1.0
    xs = np.array([1.0])

    # Iterate to compute the steps of the Brownian motion.
    for k in range(pnts):
        xs = np.append(xs, xs[-1] + norm.rvs(scale=delta**2*dt))
    x = np.linspace(0, length,len(xs))
    spl = UnivariateSpline(x, xs, k=1, s=0)
    return x, spl



def runsim(p, fn):
    gamma = p["gamma"]
    R0 = p["R0"]
    InfDays = p["InfDays"]
    intervals = p["intervals"]
    y0 = p["y0"]
    import_packet = p["import_packet"]
    nu = p["nu"]
    vaccf = lambda x: 0.5*(signal.square(x/(2*np.pi)/3)+1)*0.002*(1/4)
    
    for R in R0:
        final_t = np.array([])
        final_sol = np.array([])
        y0_int = np.copy(y0)
        for i, interval in enumerate(intervals[:-1]):
            params = [gamma, nu, R, InfDays, y0_int]
            t, sol = runsir(params,intervals[i], intervals[i+1], fn, vaccf)
            final_t = np.concatenate((final_t, t)) if final_t.size else t
            final_sol = np.concatenate((final_sol, sol)) if final_sol.size else sol
            y0_int = sol[-1]
            y0_int[1]+=import_packet # Imported COVID
            y0_int = y0_int/np.sum(y0_int) #Normalise
        plt.figure()
        plt.plot(final_t, final_sol[:,0])
        plt.plot(final_t, final_sol[:,1])
        plt.plot(final_t, final_sol[:,2])
        plt.plot(final_t, final_sol[:,3])
        plt.legend(["S","I","R","V"])

x, spl = gen_noisefn(500) # Arg must be as long as needed
plt.plot(x, spl(x))

runsim(default_params, sirv)
#vaccf = lambda x: 0.5*(signal.square(x/(2*np.pi)/3)+1)*0.01
#y = np.linspace(0,500,100000)
#plt.plot(y,vaccf(y))
plt.show()



