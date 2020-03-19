import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from length_of_stay_model.prob_hospitalisation import project_path

mpl.rc('font', family = 'serif', size = 14)
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

x = np.linspace(0, 20, 1000)
y = stats.norm.pdf((x-10)/3.5)
plt.plot(x, y, label='Normal(10.0, 3.5)')
plt.xlim(0, 20)
plt.ylim(0, 0.5)
plt.xlabel('Days delay to symptom onset')
_ = plt.ylabel('Density')
plt.savefig(project_path + '/figs/delay_to_symptoms.png', dpi=300)
plt.show()
x = np.linspace(0, 20, 1000)
rv = stats.gamma(8)
plt.plot(x, rv.pdf(x), label='Normal(10.0, 3.5)')
plt.xlim(0, 20)
plt.ylim(0, 0.2)
plt.yticks([0.0, 0.05, 0.10, 0.15, 0.20])
plt.xlabel('Length of stay in the ICU')
_ = plt.ylabel('Density')
plt.savefig(project_path + '/figs/los_dist.png', dpi=300)

