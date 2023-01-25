
import matplotlib.pyplot as plt
import numpy as np

xmin = 0
xmax = 5
N = []
MErr = []
n = 20
#for n in range(2,50, 1):
N.append(n)
X = np.linspace(xmin,xmax, n)
h = xmax/n 

# definition of the DE
def f(y):
    a = 1 
    return -a*y

#Implementation of Euler method
y = 10
Y = [y]
for ii in range(n-1):
    y = y + h*f(y)
    Y.append(y)
    
# true analytical function
def fth(x, y0):
    a = 1
    return y0*np.exp(-a*x)

# error function
Err = np.abs(Y-fth(X, 10))
#MErr.append(np.max(Err))
    
plt.figure(figsize=(5, 2.7), layout='constrained')
plt.plot(X, fth(X, 10))
plt.plot(X, Y)
plt.plot(X, Err)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Euler")
#plt.plot(N, MErr)