import matplotlib.pyplot as plt
import numpy as np
y = np.linspace(-np.pi,np.pi, 200)
x = np.sin(y)


plt.plot(x, y, label="first", color='r')
plt.legend()
plt.xlabel("x_values")
plt.ylabel("y_values")
plt.title("this a graph\npavan")
plt.show()