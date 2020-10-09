import SIRV
import numpy as np
import scipy.signal 
import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')
from scipy import signal


multiplier = 0.003
periodic_vaccf = lambda x: 0.5*(signal.square(x/(2*np.pi)/3)+1)*multiplier*2
const_vaccf = lambda x: multiplier
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


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

ax1.plot(m1.x, [periodic_vaccf(a)/multiplier for a in m1.x])
ax2.plot(m1.x, [const_vaccf(a)/multiplier for a in m1.x])

#ax1.axes.get_yaxis().set_visible(False)
#ax2.axes.get_yaxis().set_visible(False)
#ax.set_ylabel("Arbitrary Units")

plt.show()
