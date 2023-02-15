#Resolution numérique
import numpy as np
from class_used import *
import matplotlib.pyplot as plt


class Integer():
    
    def __init__(self, methode):
        self.methode = methode
        
    def applique(self):
        return(self.methode.applique())
    
    def change(self, methode):
        self.methode = methode

class Euler(Integer):
    
    def __init__(self, function, t0, tmax, h, u0):
        self.h = h
        self.t = t0
        self.tmax = tmax
        self.function = function
        self.u = u0
        self.U = [np.copy(self.u)] 
        self.T = [t0]
        
    def applique(self):
        
        
        while self.t < self.tmax:
            self.h = min(np.abs(self.h), np.abs(self.tmax-self.t))
            self.u += self.h*self.function(self.t, self.u)
            self.t += self.h
        
            self.T.append(self.t)
            self.U.append(np.copy(self.u))
            
        
        return (np.array(self.T), np.array(self.U))


class RK4(Integer):
    
    def __init__(self, function, t0, tmax, h, u0):
        self.h = h
        self.t = t0
        self.tmax = tmax
        self.function = function
        self.u = u0
        self.U = [np.copy(self.u)] 
        self.T = [t0]
        
    def applique(self):
        
        while self.t < self.tmax:
            
            self.h = min(np.abs(self.h), np.abs(self.tmax-self.t))
            k0 = self.h * self.function(self.t, self.u)
            k1 = self.h * self.function(self.t + self.h/2, self.u + k0/2)
            k2 = self.h * self.function(self.t + self.h/2, self.u + k1/2)
            k3 = self.h * self.function(self.t + self.h, self.u + k2)
            self.t += self.h
            self.u += 1/6 * (k0 + 2*k1 + 2*k2 + k3)
            self.T.append(self.t)
            self.U.append(np.copy(self.u))
        
        return(np.array(self.T), np.array(self.U))


class RK5_adaptatif(Integer):
    def __init__(self, function, t0, tmax, h, u0, tol):
        self.h = h
        self.t = t0
        self.tmax = tmax
        self.function = function
        self.u = u0
        self.tol = tol
        self.u = u0
        self.U = [np.copy(self.u)] 
        self.T = [t0]
        self.H = [h]
        self.Err = []
        
    def applique(self):
        
        a1 = 0.2; a2 = 0.3; a3 = 0.8; a4 = 8/9; a5 = 1.0
        a6 = 1.0
        c0 = 35/384; c2 = 500/1113; c3 = 125/192
        c4 = -2187/6784; c5 = 11/84
        d0 = 5179/57600; d2 = 7571/16695; d3 = 393/640
        d4 = -92097/339200; d5 = 187/2100; d6 = 1/40
        b10 = 0.2
        b20 = 0.075; b21 = 0.225
        b30 = 44/45; b31 = -56/15; b32 = 32/9
        b40 = 19372/6561; b41 = -25360/2187; b42 = 64448/6561
        b43 = -212/729
        b50 = 9017/3168; b51 =-355/33; b52 = 46732/5247
        b53 = 49/176; b54 = -5103/18656
        b60 = 35/384; b61 = 0 ; b62 = 500/1113; b63 = 125/192;
        b64 = -2187/6784; b65 = 11/84

        k0 = self.h * self.function(self.t, self.u)
        while self.t < self.tmax:
            
            k1 = self.h * self.function(self.t + a1*self.h, self.u + b10*k0)
            k2 = self.h * self.function(self.t + a2*self.h, self.u + b20*k0 \
            + b21*k1)
            k3 = self.h * self.function(self.t + a3*self.h, self.u + b30*k0 \
            + b31*k1 + b32*k2)
            k4 = self.h * self.function(self.t + a4*self.h, self.u + b40*k0 \
            + b41*k1 + b42*k2 + b43*k3)
            k5 = self.h * self.function(self.t + a5*self.h, self.u + b50*k0 \
            + b51*k1 + b52*k2 + b53*k3 + b54*k4)
            k6 = self.h * self.function(self.t + a6*self.h, self.u + b60*k0 \
            + b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5)
            
            
            E = (c0 - d0)*k0 + (c2 - d2)*k2 + (c3 - d3)*k3 + (c4 - d4)*k4 \
            + (c5 - d5)*k5 - d6*k6
            e = np.sqrt(np.sum(E**2))/len(self.u)
            
            h2 = 0.9*self.h*(self.tol/e)**(0.2)
            
            if e<= self.tol : 
                
                self.u += c0*k0 + c2*k2 + c3*k3 + c4*k4 + c5*k5
                self.t += self.h
                self.T.append(self.t)
                self.U.append(np.copy(self.u))
                self.H.append(self.h)
                self.Err.append(e)
                
                k0 = h2/self.h * k6
                if h2 > 10*self.h : h2 = 10*self.h
            
            else : 
                
                if h2 < 0.1*self.h : h2 = 0.1*self.h
                k0 = h2/self.h*k0
            self.h = h2
            
        return (np.array(self.T), np.array(self.U), np.array(self.H), np.array(self.Err))
