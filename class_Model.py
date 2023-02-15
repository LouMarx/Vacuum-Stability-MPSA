import numpy as np
from variables import *
from Runge_Kutta_class import *
import matplotlib.pyplot as plt
from sympy.interactive import printing
printing.init_printing(use_latex = True)


class Model:
    
    def __init__(self, b, tmax = np.log10(mgut), h = 0.01, tol = 10**-13):
        
        self.b = b
        self.tmax = tmax
        self.h = h
        self.tol = tol
        
        
        self.__t0 = t0
        self.__mz = mz
        self.__mgut = mgut
        self.__mh = mh
        self.__mt = mt
        self.__v = v
        self.__lambda0 = lambda0
        self.__y0 = y0
        self.__b_standarModel = b_standardModel
        self.__b_SUSYModel = b_SUSYModel
        self.__alphaQED = alphaQED
        self.__sin2thetaW =sin2thetaW
        self.__alpha1 = alpha1
        self.__alpha2 = alpha2
        self.__alpha3 = alpha3
        self.__alphaGlobal = np.array([alpha1, alpha2, alpha3])
        self.__g0 = np.array([np.sqrt(4*np.pi*alpha1), np.sqrt(4*np.pi*alpha2), np.sqrt(4*np.pi*alpha3), y0, lambda0])
        self.__C = 1/(self.__alphaGlobal) + (np.log(10)*(self.b)/(2*np.pi))*self.__t0
        self.T = np.arange(self.__t0, self.tmax, self.h)
        
    def __F(self, t,u):
        __F = np.zeros(5)
        __F[0] = np.log(10)*self.b[0] * 1/(16*np.pi**2)*u[0]**3
        __F[1] = np.log(10)*self.b[1] * 1/(16*np.pi**2)*u[1]**3
        __F[2] = np.log(10)*self.b[2] * 1/(16*np.pi**2)*u[2]**3
        __F[3] = (np.log(10)/(4*np.pi)**2) * u[3] * ((9/2)*u[3]**2 - (9/4)*u[1]**2\
              -(17/12)*(3/5)*u[0]**2 - 8*u[2]**2)
        __F[4] = (np.log(10)/(4*np.pi)**2) * (24*u[4]**2 - 6*u[3]**4 + \
              (3/8)*(2*u[1]**4 + (u[1]**2 + (3/5)*u[0]**2)**2) + \
              u[4] * (-9*u[1]**2 - 3*(3/5)*u[0]**2 + 12*u[3]**2))
        return (__F)        
    
    def get_Model(self):
        """
        Cette méthode permet de savoir à l'aide du paramètre b (array) représentant la 
        constante bi si le modèle utilisé est celui du modèle 
        standard, du modèle SUSY ou d'un autre modèle non identifié'

        Returns
        -------
        None.

        """
        if np.array_equal(self.b, self.__b_standarModel):
             print("Standard Model")
        elif np.array_equal(self.b, self.__b_SUSYModel):
             print("SUSY Model")
        else :
             print("Unknown Model")
    
    
    def set_tmax(self, tmax):
        """
        Cette méthode permet de modifier la valeur de tmax, la valeur final de 
        T. Elle modifie également les valeurs de T (array).

        Parameters
        ----------
        tmax : int ou float.

        Returns
        -------
        None.

        """
        self.tmax = tmax
        self.T = np.arange(self.__t0, self.tmax, self.h)
        
    def set_h(self, h):
        """
        Cette méthode permet de modifier la valeur du pas h. Elle modifie 
        également les valeurs de T (array).

        Parameters
        ----------
        h : int ou float.

        Returns
        -------
        None.

        """
        self.h = h
        self.T = np.arange(self.__t0, self.tmax, self.h)
    
    def set_tol(self, tol):
        """
        Cette méthode permet de modifier la valeur de la tolérance tol pour 
        l'application de la méthode de Runge-Kutta d'ordre 5 adaptative. 

        Parameters
        ----------
        tol : int ou float.

        Returns
        -------
        None.

        """
        self.tol = tol

    def get_alpha_th(self):
        """
        Cette méthode permet d'obtenir les différentes constantes de couplage 
        alpha calculé analytiquement. Elle renvoie un tableau dont la première 
        ligne correspond à la constonte de couplage de l'interraction electromagnétique,
        la deuxième, la constante de couplage de l'interraction faible et la troisième, 
        la constante de couplage de l'interraction forte.

        Returns
        -------
        TYPE : array

        """
        self.__Alpha = Calcul_coupling_constant(self.__C, self.b, self.T)
        
        return (self.__Alpha.get_alpha())
    
    def get_g_th(self):
        """
        Cette méthode permet d'obtenir les différentes constantes de couplage 
        g calculé analytiquement. Elle renvoie un tableau dont la première 
        ligne correspond à la constonte de couplage de l'interraction electromagnétique,
        la deuxième, la constante de couplage de l'interraction faible et la troisième, 
        la constante de couplage de l'interraction forte.

        Returns
        -------
        TYPE : array

        """
        self.__Alpha = Calcul_coupling_constant(self.__C, self.b, self.T)
        
        return(self.__Alpha.get_g())
    

    def plot_inv_alpha1_th(self):
        """
        Cette méthode trace la constante de couplage électromagnétique 1/alpha1 
        en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__alpha_array = self.get_alpha_th()
        self.__Alpha_plot = Plot(self.T, 1/self.__alpha_array[0])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{1,th}$')
        return(self.__Alpha_plot.plot_list())
    
    def plot_inv_alpha1_euler(self):
        """
        Cette méthode trace la constante de couplage électromagnétique 1/alpha1 
        numériquement par la méthode d'Euler (explicite) en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__Euler = Integer(Euler(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__Euler.applique()    
        self.__plot_Euler = Plot(self.__abscisse, (4*np.pi)/self.__ordonnée[:,0]**2)
        
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{1,euler}$')
        return(self.__plot_Euler.plot_list())
    
    def plot_inv_alpha1_rgk4(self):
        """
        Cette méthode trace la constante de couplage électromagnétique 1/alpha1 
        numériquement par la méthode de Runge-Kutta d'ordre 4 en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK4(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, (4*np.pi)/self.__ordonnée[:,0]**2)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{1,rgk4}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_inv_alpha1_rgk5(self):
        """
        Cette méthode trace la constante de couplage électromagnétique 1/alpha1 
        numériquement par la méthode de Runge-Kutta d'ordre 5 adaptative en 
        fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK5_adaptatif(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0), self.tol))
        self.__abscisse, self.__ordonnée, self.pas, self.err = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, (4*np.pi)/self.__ordonnée[:,0]**2)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{1,rgk5}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_inv_alpha2_th(self):
        """
        Cette méthode trace la constante de couplage faible 1/alpha2
        en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__alpha_array = self.get_alpha_th()
        self.__Alpha_plot = Plot(self.T, 1/self.__alpha_array[1])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{2,th}$')
        return(self.__Alpha_plot.plot_list())
   
    def plot_inv_alpha2_euler(self):
        """
        Cette méthode trace la constante de couplage faible 1/alpha2 
        numériquement par la méthode d'Euler (explicite) en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__Euler = Integer(Euler(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__Euler.applique()    
        self.__plot_Euler = Plot(self.__abscisse, (4*np.pi)/self.__ordonnée[:,1]**2)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{2,euler}$')
        return(self.__plot_Euler.plot_list())
    
    def plot_inv_alpha2_rgk4(self):
        """
        Cette méthode trace la constante de couplage faible 1/alpha2 
        numériquement par la méthode de Runge-Kutta d'ordre 4 en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK4(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, (4*np.pi)/self.__ordonnée[:,1]**2)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{2,rgk4}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_inv_alpha2_rgk5(self):
        """
        Cette méthode trace la constante de couplage faible 1/alpha2 
        numériquement par la méthode de Runge-Kutta d'ordre 5 adaptative en 
        fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK5_adaptatif(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0), self.tol))
        self.__abscisse, self.__ordonnée, self.pas, self.err = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, (4*np.pi)/self.__ordonnée[:,1]**2)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{2,rgk5}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_inv_alpha3_th(self):
        """
        Cette méthode trace la constante de couplage forte 1/alpha3
        en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__alpha_array = self.get_alpha_th()
        self.__Alpha_plot = Plot(self.T, 1/self.__alpha_array[2])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{3,th}$')
        return(self.__Alpha_plot.plot_list())
    
    def plot_inv_alpha3_euler(self):
        """
        Cette méthode trace la constante de couplage forte 1/alpha3 
        numériquement par la méthode d'Euler (explicite) en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__Euler = Integer(Euler(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__Euler.applique()    
        self.__plot_Euler = Plot(self.__abscisse, (4*np.pi)/self.__ordonnée[:,2]**2)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{3,euler}$')
        return(self.__plot_Euler.plot_list())
    
    def plot_inv_alpha3_rgk4(self):
        """
        Cette méthode trace la constante de couplage forte 1/alpha3 
        numériquement par la méthode de Runge-Kutta d'ordre 4 en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK4(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, (4*np.pi)/self.__ordonnée[:,2]**2)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{3,rg4}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_inv_alpha3_rgk5(self):
        """
        Cette méthode trace la constante de couplage forte 1/alpha3 
        numériquement par la méthode de Runge-Kutta d'ordre 5 adaptative en 
        fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK5_adaptatif(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0), self.tol))
        self.__abscisse, self.__ordonnée, self.pas, self.err = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, (4*np.pi)/self.__ordonnée[:,2]**2)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{3,rgk5}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_inv_alpha_th_all(self):
        """
        Cette méthode trace toutes les constantes de couplage 1/alpha 
        (électromagnétique, faible, forte) en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__alpha_array = self.get_alpha_th()
        self.__Alpha_plot = Plot(self.T, 1/self.__alpha_array)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{th}$')
        return(self.__Alpha_plot.plot_array())
    
    def plot_inv_alpha_all_euler(self):
        """
        Cette méthode trace toutes les constantes de couplage 1/alpha 
        (électromagnétique, faible, forte) numériquement par la méthode 
        d'Euler (explicite), en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__Euler = Integer(Euler(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__Euler.applique()    
        self.__plot_Euler = Plot(self.__abscisse, (4*np.pi)/(self.__ordonnée.T[0:3,:])**2)
        #la matrice des ordonnées doit être transposée avant d'être tracé par 
        # la méthode de plot.array()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{euler}$')
        return(self.__plot_Euler.plot_array())
    
    def plot_inv_alpha_all_rgk4(self):
        """
        Cette méthode trace toutes les constantes de couplage 1/alpha 
        (électromagnétique, faible, forte) numériquement par la méthode 
        de Runge-Kutta d'ordre 4, en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK4(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, (4*np.pi)/(self.__ordonnée.T[0:3,:])**2)
        #la matrice des ordonnées doit être transposée avant d'être tracé par 
        # la méthode de plot.array()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{rgk4}$')
        return(self.__plot_RGK.plot_array())
    
    def plot_inv_alpha_all_rgk5(self):
        """
        Cette méthode trace toutes les constantes de couplage 1/alpha 
        (électromagnétique, faible, forte) numériquement par la méthode 
        de Runge-Kutta d'ordre 5 adaptative en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK5_adaptatif(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0), self.tol))
        self.__abscisse, self.__ordonnée, self.pas, self.err = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, (4*np.pi)/(self.__ordonnée.T[0:3,:])**2)
        #la matrice des ordonnées doit être transposée avant d'être tracé par 
        # la méthode de plot.array()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$1/\alpha_{rgk5}$')
        return(self.__plot_RGK.plot_array())
    
    def plot_g1_th(self):
        """
        Cette méthode trace la constante de couplage électromagnétique g1 
        en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__g_array = self.get_g_th()
        self.__G_plot = Plot(self.T, self.__g_array[0])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{1,th}$')
        return(self.__G_plot.plot_list())
    
    def plot_g1_euler(self):
        """
        Cette méthode trace la constante de couplage électromagnétique g1 
        numériquement par la méthode d'Euler (explicite), en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__Euler = Integer(Euler(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__Euler.applique()    
        self.__plot_Euler = Plot(self.__abscisse, self.__ordonnée[:,0])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{1,euler}$')
        return(self.__plot_Euler.plot_list())
    
    def plot_g1_rgk4(self):
        """
        Cette méthode trace la constante de couplage électromagnétique g1 
        numériquement par la méthode de Runge-Kutta d'ordre 4, en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK4(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, self.__ordonnée[:,0])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{1,rgk4}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_g1_rgk5(self):
        """
        Cette méthode trace la constante de couplage électromagnétique g1 
        numériquement par la méthode de Runge-Kutta d'ordre 5 adaptative en 
        fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK5_adaptatif(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0), self.tol))
        self.__abscisse, self.__ordonnée, self.pas, self.err = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, self.__ordonnée[:,0])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{1,rgk5}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_g2_th(self):
        """
        Cette méthode trace la constante de couplage faible g2 
        en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__g_array = self.get_g_th()
        self.__G_plot = Plot(self.T, self.__g_array[1])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{2,th}$')
        return(self.__G_plot.plot_list())
    
    def plot_g2_euler(self):
        """
        Cette méthode trace la constante de couplage faible g2 
        numériquement par la méthode d'Euler (explicite), en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        
        self.__Euler = Integer(Euler(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))       
        self.__abscisse, self.__ordonnée = self.__Euler.applique()
        self.__plot_Euler = Plot(self.__abscisse, self.__ordonnée[:,1])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{2,euler}$')
        return(self.__plot_Euler.plot_list())
    
    def plot_g2_rgk4(self):
        """
        Cette méthode trace la constante de couplage faible g2 numériquement 
        par la méthode de Runge-Kutta d'ordre 4, en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK4(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, self.__ordonnée[:,1])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{2,rgk4}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_g2_rgk5(self):
        """
        Cette méthode trace la constante de couplage faible g2
        numériquement par la méthode de Runge-Kutta d'ordre 5 adaptative en 
        fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK5_adaptatif(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0), self.tol))
        self.__abscisse, self.__ordonnée, self.pas, self.err = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, self.__ordonnée[:,1])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_[2,rgk5}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_g3_th(self):
        """
        Cette méthode trace la constante de couplage forte g3 
        en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__g_array = self.get_g_th()
        self.__G_plot = Plot(self.T, self.__g_array[2])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{3,th}$')
        return(self.__G_plot.plot_list())
    
    def plot_g3_euler(self):
        """
        Cette méthode trace la constante de couplage forte g3
        numériquement par la méthode d'Euler (explicite), en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__Euler = Integer(Euler(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__Euler.applique()
        self.__plot_Euler = Plot(self.__abscisse, self.__ordonnée[:,2])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{3,euler}$')
        return(self.__plot_Euler.plot_list())
    
    def plot_g3_rgk4(self):
        """
        Cette méthode trace la constante de couplage forte g3 numériquement 
        par la méthode de Runge-Kutta d'ordre 4, en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK4(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, self.__ordonnée[:,2])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{3,rgk4}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_g3_rgk5(self):
        """
        Cette méthode trace la constante de couplage forte g3 
        numériquement par la méthode de Runge-Kutta d'ordre 5 adaptative en 
        fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK5_adaptatif(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0), self.tol))
        self.__abscisse, self.__ordonnée, self.pas, self.err = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, self.__ordonnée[:,2])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{3,rgk5}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_g_th_all(self):
        """
        Cette méthode trace toutes les constantes de couplage g 
        (électromagnétique, faible, forte) en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__g_array = self.get_g_th()
        self.__G_plot = Plot(self.T, self.__g_array)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{th}$')
        return(self.__G_plot.plot_array())
    
    def plot_g_all_euler(self):
        """
        Cette méthode trace toutes les constantes de couplage g 
        (électromagnétique, faible, forte) numériquement par la méthode d'Euler
        (explicite), en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__Euler = Integer(Euler(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__Euler.applique()
        self.__plot_Euler = Plot(self.__abscisse, self.__ordonnée.T[0:3,:]) 
        #la matrice des ordonnées doit être transposée avant d'être tracé par 
        # la méthode de plot.array()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{euler}$')
        return(self.__plot_Euler.plot_array())
    
    def plot_g_all_rgk4(self):
        """
        Cette méthode trace toutes les constantes de couplage g 
        (électromagnétique, faible, forte) numériquement par la méthode de 
        Runge-Kutta d'ordre 4, en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK4(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, self.__ordonnée.T[0:3,:])
        #la matrice des ordonnées doit être transposée avant d'être tracé par 
        # la méthode de plot.array()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{rgk4}$')
        return(self.__plot_RGK.plot_array())
    
    def plot_g_all_rgk5(self):
        """
        Cette méthode trace toutes les constantes de couplage g 
        (électromagnétique, faible, forte) numériquement par la méthode de 
        Runge-Kutta d'ordre 5 adaptative, en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK5_adaptatif(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0), self.tol))
        self.__abscisse, self.__ordonnée, self.pas, self.err = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, self.__ordonnée.T[0:3,:])
        #la matrice des ordonnées doit être transposée avant d'être tracé par 
        # la méthode de plot.array()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$g_{rgk5}$')
        return(self.__plot_RGK.plot_array())
       
    def plot_Yukawa_euler(self):
        """
        Cette méthode trace le terme de Yukawa du quark top numériquement 
        par la méthode d'Euler (explicite), en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__Euler = Integer(Euler(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__Euler.applique()
        self.__plot_Euler = Plot(self.__abscisse, self.__ordonnée[:,3]) 
        
        plt.xlabel(r'$t$')
        plt.ylabel(r'$Yt_{euler}$')
        return(self.__plot_Euler.plot_list())
    
    def plot_Yukawa_rgk4(self):
        """
        Cette méthode trace le terme de Yukawa du quark top numériquement 
        par la méthode de Runge-Kutta d'ordre 4, en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK4(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, self.__ordonnée[:,3])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$Yt_{rgk4}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_Yukawa_rgk5(self):
        """
        Cette méthode trace le terme de Yukawa du quark top numériquement 
        par la méthode de Runge-Kutta d'ordre 5 adaptative, en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK5_adaptatif(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0), self.tol))
        self.__abscisse, self.__ordonnée, self.pas, self.err = self.__RGK.applique()    
        self.__plot_RGK = Plot(self.__abscisse, self.__ordonnée[:,3])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$Yt_{rgk5}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_lambda_euler(self):
        """
        Cette méthode trace le paramètre lambda de Higgs numériquement 
        par la méthode d'Euler (explicite), en fonction du paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__Euler = Integer(Euler(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__Euler.applique()
        self.__plot_Euler = Plot(self.__abscisse, self.__ordonnée[:,4]) 
        
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\lambda_{euler}$')
        return(self.__plot_Euler.plot_list())
    
    def plot_lambda_rgk4(self):
        """
        Cette méthode trace le paramètre lambda de Higgs numériquement 
        par la méthode de Runge-Kutta d'ordre 4, en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK4(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0)))
        self.__abscisse, self.__ordonnée = self.__RGK.applique()
        self.__plot_RGK = Plot(self.__abscisse, self.__ordonnée[:,4]) 
        
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\lambda_{rgk4}$')
        return(self.__plot_RGK.plot_list())
        
    def plot_lambda_rgk5(self):
        """
        Cette méthode trace le paramètre lambda de Higgs numériquement 
        par la méthode de Runge-Kutta d'ordre 5 adaptative, en fonction du 
        paramètre t = log(Q)

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK5_adaptatif(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0), self.tol))
        self.__abscisse, self.__ordonnée, self.pas, self.err = self.__RGK.applique()
        self.__plot_RGK = Plot(self.__abscisse, self.__ordonnée[:,4]) 
        
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\lambda_{rgk5}$')
        return(self.__plot_RGK.plot_list())
    
    def plot_pas_rgk5(self):
        """
        Cette méthode trace l'évolution du pas en fonction du 
        paramètre t = log(Q) lors de l'utilisitation de la méthode Runge-Kutta
        d'ordre 5 adaptative

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK5_adaptatif(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0), self.tol))
        self.__abscisse, self.__ordonnée, self.__pas, self.__err = self.__RGK.applique()    
        
        self.__plot_RGK = Plot(self.__abscisse, self.__pas)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$pas$')
        return(self.__plot_RGK.plot_list())
    
    def plot_erreur_rgk5(self):
        """
        Cette méthode trace l'évolution de l'erreur calculé en utilisant une norme
        Root Mean Square en fonction du paramètre t = log(Q) 
        lors de l'utilisitation de la méthode Runge-Kutta d'ordre 5 adaptative

        Returns
        -------
        None.

        """
        self.__RGK = Integer(RK5_adaptatif(self.__F, self.__t0, self.tmax, self.h, np.copy(self.__g0), self.tol))
        self.__abscisse, self.__ordonnée, self.__pas, self.__err = self.__RGK.applique()    
        
        self.__plot_RGK = Plot(self.__abscisse[1:], self.__err)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$erreur$')
        return(self.__plot_RGK.plot_list())
    
    def new_plot(self):
        try :
            self.__plot_Euler.new_plot()
            self.__plot_RGK.new_plot()
            self.__Alpha_plot.new_plot()
            self.__G_plot.new_plot()
            
        except :
            pass

class Calcul_coupling_constant :
   
    def __init__(self, C, b, T):
        self.__T = T
        self.__C = C
        self.__b = b
        
        
        self.__Alpha = np.zeros((3, self.__T.size))
    
    def get_alpha(self):
        """
        Cette méthode calcule analytiquement les constantes de couplage alpha 
        des différentes intéractions (électromagnétique, faible et forte) 
        et renvoie un tableau dont la première ligne correspond à la constonte 
        de couplage de l'interraction electromagnétique,la deuxième, 
        la constante de couplage de l'interraction faible et la troisième, 
        la constante de couplage de l'interraction forte

        Returns
        -------
        TYPE : array

        """
        self.__Alpha[0] = 1/(self.__C[0]-(self.__b[0]*np.log(10)/(2*np.pi)) * self.__T)
        self.__Alpha[1] =  1/(self.__C[1]-(self.__b[1]*np.log(10)/(2*np.pi)) * self.__T)
        self.__Alpha[2] = 1/(self.__C[2]-(self.__b[2]*np.log(10)/(2*np.pi)) * self.__T)
        return (self.__Alpha)
    
    def get_g(self):
        """
        Cette méthode calcule analytiquement les constantes de couplage g 
        des différentes intéractions (électromagnétique, faible et forte) 
        et renvoie un tableau dont la première ligne correspond à la constonte 
        de couplage de l'interraction electromagnétique,la deuxième, 
        la constante de couplage de l'interraction faible et la troisième, 
        la constante de couplage de l'interraction forte

        Returns
        -------
        TYPE : array.

        """
        self.__g = np.sqrt(4*np.pi*self.get_alpha())
        return(self.__g)

class Plot :
    def __init__(self, T, U):
        self.__T = T
        self.__U = U
        plt.grid(True)
        
    def plot_list(self):
        """
        Cette méthode trace une liste

        Returns
        -------
        None.

        """
        plt.plot(self.__T, self.__U)
    
    def plot_array(self):
        """
        Cette méthode trace un tableau

        Returns
        -------
        None.

        """
        self.__row = self.__U.shape[0]
        for k in range (0, self.__row):
            plt.plot(self.__T, self.__U[k])
    
    def new_plot(self):
        return (plt.close())

