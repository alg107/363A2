import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import UnivariateSpline

plt.style.use('Solarize_Light2')
from scipy.stats import norm
from scipy import signal

def find_nearest_idx(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

vaccine_functions = [
         lambda x: 0.5*(signal.square(x/(2*np.pi)/3)+1)*0.002*(1/4)
        ]

"""
Gathering Parameters:
    - Infected days
    - R0
    - Vaccine cost
    - Vaccine capabilities (How many can we vaccinate)
"""

default_params = {
        "nu": (1/(30*4)),
        "InfDays": 10,
        "R0": [1.8],
        "intervals": [0,50,120,200, 300, 400, 450, 365*3],
        "y0": [1.0,0.0,0.0,0.0,0.0],
        "import_packet": 6e-7, # 3 of 5,000,000
        "vacc_cost": 10.0*5e6
        }


class Model:
    def __init__(self, modelfn, vacc_function, params, name=""):
        self.modelfn = modelfn
        self.vacc_function = vacc_function
        self.params = params
        self.name = name
        self.gen_noisefn(500)

    def run(self, vacc=True):
        fn = self.modelfn
        p = self.params
        R0 = p["R0"]
        InfDays = p["InfDays"]
        gamma = 1/InfDays
        intervals = p["intervals"]
        y0 = p["y0"]
        import_packet = p["import_packet"]
        nu = p["nu"]
        vaccf = self.vacc_function
        
        for R in R0:
            final_t = np.array([])
            final_sol = np.array([])
            y0_int = np.copy(y0)
            for i, interval in enumerate(intervals[:-1]):
                t, sol = self.runsir(y0_int,R,intervals[i], intervals[i+1], fn, vaccf)
                final_t = np.concatenate((final_t, t)) if final_t.size else t
                final_sol = np.concatenate((final_sol, sol)) if final_sol.size else sol
                y0_int = sol[-1]
                y0_int[1]+=import_packet # Imported COVID
                y0_int[:4] = y0_int[:4]/np.sum(y0_int[:4]) #Normalise
            plt.figure()
            if self.name: plt.title(self.name)
            plt.plot(final_t, final_sol[:,0])
            plt.plot(final_t, final_sol[:,1])
            plt.plot(final_t, final_sol[:,2])
            plt.plot(final_t, final_sol[:,3])
            plt.legend(["$S$","$I$","$R$","$V$"])
            plt.figure()
            plt.plot(final_t, final_sol[:,4])#self.params["vacc_cost"]
            plt.legend(["Proportion of pop. vacc"])
        self.print_model_stats(final_t, final_sol)

    def print_model_stats(self, final_t, final_sol):
        vcost = self.params["vacc_cost"]
        max_I_index = np.argmax(final_sol[:,1])
        max_I = final_sol[:,1][max_I_index]
        max_I_day = final_t[max_I_index]
        final_S = final_sol[:,0][-1]
        final_I = final_sol[:,1][-1]
        final_R = final_sol[:,2][-1]
        y1c = final_sol[:,4][find_nearest_idx(final_t, 365)]*vcost
        y2c = final_sol[:,4][find_nearest_idx(final_t, 365*2)]*vcost-y1c
        y3c = final_sol[:,4][find_nearest_idx(final_t, 365*3)]*vcost-y1c-y2c
         
        # Violated hospital capacity?
        print("Final Recovered/Deceased:", round(final_R, 3))
        print("Max Infected:", round(max_I, 3))
        print("Day of Max Infection:", round(max_I_day))
        print("1st Year Vacc Cost:", f'${round(y1c, 2):,}')
        print("    ", f'{round(y1c*100/vcost, 2):,}', "% Vaccinated this year")
        print("2nd Year Vacc Cost:", f'${round(y2c, 2):,}')
        print("    ", f'{round(y2c*100/vcost, 2):,}', "% Vaccinated this year")
        print("3rd Year Vacc Cost:", f'${round(y3c, 2):,}')
        print("    ", f'{round(y3c*100/vcost, 2):,}', "% Vaccinated this year")

    def runsir(self,y0, R, ti, tf, fn, vaccf):
        nu = self.params["nu"]
        beta = R/self.params["InfDays"]
        gamma = 1/self.params["InfDays"]
        t = np.linspace(ti,tf,1000)
        sol = odeint(fn, y0, t, args=(beta, gamma, nu, self.spl, self.vacc_function, self.params["vacc_cost"]))
        return t, sol

    def gen_intervals(self, tf):
        pass

    def gen_noisefn(self, length, pnts=1000):
        # Process parameters
        delta = 0.2
        dt = 0.01

        x = 1.0
        xs = np.array([1.0])

        # Iterate to compute the steps of the Brownian motion.
        for k in range(pnts):
            xs = np.append(xs, xs[-1] + norm.rvs(scale=delta**2*dt))
        x = np.linspace(0, length,len(xs))
        spl = UnivariateSpline(x, xs, k=1, s=0)
        self.x = x
        self.spl = spl


def sirv(y,t,beta, gamma, nu, spl, vaccf, vacc_cost):
    S,I,R,V,C=y
    #betan = betan*bm_samples[int(np.floor(tn*10))]
    
    dydt = [
            #(np.sin(t)*0.1+1)
            -beta*spl(t)*I*S -vaccf(t)*S + nu*V,
            beta*spl(t)*I*S - spl(t)*gamma*I,
            spl(t)*gamma*I,
            vaccf(t)*S-nu*V,
            vaccf(t)*S
            ]
    return dydt

