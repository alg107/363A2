import SIRV
import numpy as np
import scipy.signal 
import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')
from scipy import signal


vaccf = lambda x: 0.5*(signal.square(x/(2*np.pi)/3)+1)*0.002
m = SIRV.Model(SIRV.sirv, vaccf, SIRV.default_params)
m.gen_noisefn(500) # Arg must be as long as needed
plt.plot(m.x, m.spl(m.x))
m.run()
plt.show()
