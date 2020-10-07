import SIRV
import numpy as np
import scipy.signal 
import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')
from scipy import signal


periodic_vaccf = lambda x: 0.5*(signal.square(x/(2*np.pi)/3)+1)*0.002
const_vaccf = lambda x: 0.001
no_vaccf = lambda x: 0

m1 = SIRV.Model(SIRV.sirv, periodic_vaccf, SIRV.default_params, name="Periodic")
print("\nPeriodic\n")
m1.run()

m2 = SIRV.Model(SIRV.sirv, const_vaccf, SIRV.default_params, name="Constant")
print("\nConstant\n")
m2.run()

m3 = SIRV.Model(SIRV.sirv, no_vaccf, SIRV.default_params, name="None")
print("\nNone\n")
m3.run()

plt.show()
