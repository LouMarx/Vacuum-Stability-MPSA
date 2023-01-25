import numpy as np
import matplotlib.pyplot as plt
from math import *

# Méthode de Runge-Kutta/Euler

def G(t, g):
    return((1/16*pi**2)*41/10*g**3)

def F(t, u):
    return u

def euler(F, t0, tmax, u0, N ):
    h =(tmax-t0)/N
    u = u0
    t = t0
    U = [u0]
    T = [t0]
    for i in range (1,N+1):
        
        u = u + F(t,u)*h
        t = t + h
        U.append(u)
        T.append(t)
    return (T, U)

def runge_kutta(F, t0, tmax, u0, N ) :
    h =(tmax-t0)/N
    u = u0
    t = t0
    U = [u0]
    T = [t0]
 
    
    for i in range (1,N+1):
        k1 = F(t, u)
        k2 = F(t + h/2, u + k1*(h/2))
        k3 = F(t + h/2, u + k2*(h/2))     
        k4 = F(t + h, u + k3*h)
        u = u + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h
        U.append(u)
        T.append(t)
    return (T, U)

"Conditions Initiales"
u0 = 1
t0 = 0.5
tmax = 1
N = 1000

"Fonction exponentielle"
x = np.linspace(t0, tmax, N+1)
y = np.exp(x)

"Fonction approché"
Tr, Ur = runge_kutta(F, t0, tmax, u0, N)
Te, Ue = euler(F, t0, tmax, u0, N)
Tr_g, Ur_g = runge_kutta(G, t0, tmax, u0, N)

"Calcul de l'erreur pour différents nombre de points"
Err_r=[]
Err_e=[]
I=[]
for i in range(1,N+1):
    
    I.append(i)
    Tr, Ur = runge_kutta(F, t0, tmax, u0, i)
    Te, Ue = euler(F, t0, tmax, u0, i)
    x = np.linspace(t0, tmax, i+1)
    y = np.exp(x)
    erreur_r = max(abs(Ur-y))
    erreur_e = max(abs(Ue-y))
    Err_r.append(erreur_r)
    Err_e.append(erreur_e)

"Figures"
fig, axs=plt.subplots(3,1)
axs[0].plot(Tr, Ur, "b.-", label='runge-kutta')
axs[0].plot(x,y, "r.-", label='exp(x)')
axs[0].plot(Te, Ue, "g.-", label='euler')
axs[1].plot(I,Err_r,"b-", label ='err_runge')
axs[1].plot(I,Err_e, "g-", label ='err_euler')
axs[2].plot(Tr_g, Ur_g)

plt.ylabel("erreur")
plt.show()

