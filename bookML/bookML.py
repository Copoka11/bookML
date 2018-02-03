#test
"""
import timeit
normal_py_sec = timeit.timeit('sum(x*x for x in range(1000))', number=10000)
naive_np_sec = timeit.timeit('sum(na*na)', 
                             setup="import numpy as np; na=np.arange(1000)",
                             number=10000)
good_np_sec = timeit.timeit('na.dot(na)',
                            setup="import numpy as np; na=np.arange(1000)",
                            number=10000)
print ("Normal Python: %f sec" % normal_py_sec)
print ("Naive NumPy: %f sec" % naive_np_sec)
print ("Good NumPy: %f sec" % good_np_sec)
"""

import scipy as sp
import matplotlib.pyplot as plt

def error (f, x, y):
    return sp.sum((f(x)-y)**2)

data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")

print(data[:10])
print(data.shape)

x = data[:,0]       #нумерация записей
y = data[:,1]       #данные

print("количество NaN",sp.sum(sp.isnan(y)))

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

inflection = int(3.5*7*24)
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)

fax = sp.linspace(0, xa[-1], 1000)
fbx = sp.linspace(0, xb[-1], 1000)

print("Error inflections=%f" % (fa_error+fb_error))

plt.scatter(x, y, s=10)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range (10)],
           ['week %i' % w for w in range(10)])
plt.autoscale(tight=True)
plt.grid(True, linestyle='-', color='0.75')
###################################################################
plt.plot(fax, fa(fax), linewidth=3, color='0.25')
plt.plot(fbx, fb(fbx), linewidth=3, color='0.45')

plt.show()

