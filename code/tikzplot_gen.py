import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

conf = np.array([0.3,0.6,0.8,0.85])
avg_err = np.array([2.746977049,	2.144839681,	1.59813396,	2.056172204])
plt.plot(conf,avg_err, "o-", lw = 4)
plt.xlabel("Confidence threshold")
plt.ylabel("RMS OD error (m)")
plt.show()
