import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

a,b, N = 0, 10, 1000        #Boundaries, datapoints
shift = -3                  #Shift, note 3/10 of L = b-a

x = np.linspace(a,b,N)
x1 = 1*x + shift
time = np.arange(1-N,N)     #Theoritical definition, time is centered at 0

y1 = sum([np.sin(2*np.pi*i*x/b) for i in range(1,5)])
y2 = sum([np.sin(2*np.pi*i*x1/b) for i in range(1,5)])

#Really only helps with large irregular data, try it
# y1 -= y1.mean()
# y2 -= y2.mean()
# y1 /= y1.std()
# y2 /= y2.std()

cross_correlation = correlate(y1,y2)
shift_calculated = time[cross_correlation.argmax()] *1.0* b/N
y3 = sum([np.sin(2*np.pi*i*(x1-shift_calculated)/b) for i in range(1,5)])
print("Preset shift: ", shift, "\nCalculated shift: ", shift_calculated)



plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.legend(("Regular", "Shifted", "Recovered"))
plt.show()
