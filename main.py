# main

# Veuiller lancer le code par section
from class_Model import *
import numpy as np
import matplotlib.pyplot as plt
from sympy.interactive import printing
printing.init_printing(use_latex = True)
#%% Comparaison des différentes méthodes d'intégration sur une fonction exponentiel simple

# Définition de la fonction exponentielle
def function_exp(t,u):
    return(u)

# Condition initial
t_initial = 0
t_final = 1
u_initial = 1
pas = 0.1
tolerance = 10**-10


# Calcul et trace la courbe en utilisant la méthode d'Euler explicite
Methode = Integer(Euler(function_exp, t_initial, t_final, pas, u_initial))
abscisse, ordonnée = Methode.applique()
Plot1 = Plot(abscisse, ordonnée)
Plot1.plot_list()
# Calcul et trace la courbe en changeant la méthode d'Euler pour celle de Runge-Kutta d'ordre 4
Methode.change(RK4(function_exp, t_initial, t_final, pas, u_initial)) 
abscisse, ordonnée = Methode.applique()  
Plot2 = Plot(abscisse, ordonnée)             
Plot2.plot_list()
# Trace la vrai courbe exponentielle
abscisse = np.arange(t_initial, t_final+pas, pas)
ordonnée = np.exp(abscisse)
plt.plot(abscisse, ordonnée)

 
#%% Création du Model 
Plot1.new_plot()
Plot2.new_plot()
plt.cla()

# Création du Model
b_standard = np.array([41/10, -19/6, -7])    
b_susy = np.array([33/5, 1, -3])
Model_Standard = Model(b_standard)
Model_Susy = Model(b_susy)

# Etude du Model
Model_Standard.get_Model() #affiche le type de modèle que nous avons
Model_Susy.get_Model()
#%% Etude des constantes de couplage g (calcul analytique et courbe en fonction de t = log(Q))
Model_Standard.new_plot()

coupling_constant_g = Model_Standard.get_g_th()

print("Constante de couplage electromagnétique g1 = ",coupling_constant_g[0,:])
print("Constante de couplage faible g2 =", coupling_constant_g[1,:])
print("Constante de couplage forte g3 =", coupling_constant_g[2,:])

Model_Standard.plot_g_th_all()

#%% Etude des constantes de couplage alpha (calcul analytique et courbe de 1/alpha en fonction de t)
Model_Standard.new_plot()

coupling_constant_alpha = Model_Standard.get_alpha_th()

print("Constante de couplage electromagnétique alpha_1 = ",coupling_constant_alpha[0,:])
print("Constante de couplage faible alpha2 =", coupling_constant_alpha[1,:])
print("Constante de couplage forte alpha3 =", coupling_constant_alpha[2,:])

Model_Standard.plot_inv_alpha_th_all()
Model_Susy.plot_inv_alpha_th_all()
#%% Comparaison des constantes de couplage alpha3 avec les différentes méthodes (peut être fait pour chaque alpha et g)

Model_Standard.new_plot()
Model_Susy.new_plot()

# Changement du pour différencier Euler et Runge-Kutta
Model_Standard.set_h(1)

# Trace les courbes de 1/alpha3 calculé analytiquement et numériquement par 
#les différentes méthodes d'intégration

Model_Standard.plot_inv_alpha3_th()
Model_Standard.plot_inv_alpha3_euler()
Model_Standard.plot_inv_alpha3_rgk4()
Model_Standard.plot_inv_alpha3_rgk5()

#%% Comparaison des constantes de couplage g avec les différentes méthodes d'intégration

Model_Standard.new_plot()

# Trace les courbes de g calculé analytiquement et numériquement par 
#les différentes méthodes d'intégration

Model_Standard.plot_g_th_all()
Model_Standard.plot_g_all_euler()
Model_Standard.plot_g_all_rgk4()
Model_Standard.plot_g_all_rgk5()

#%% Comparaison du terme de Yukawa pour le quark top en fonction du Model

Model_Standard.new_plot()

Model_Standard.set_h(0.01)

Model_Standard.plot_Yukawa_rgk5()
Model_Susy.plot_Yukawa_rgk5()
#%% Même chose pour le paramètre lambda de Higgs

Model_Standard.new_plot()
Model_Susy.new_plot()

Model_Standard.plot_lambda_rgk5()
Model_Susy.plot_lambda_rgk5()

#%% La courbe suivante représente l'évolution du pas en fonction de t = log(Q)
#pour la méthode de Runge-Kutta adaptative

Model_Standard.new_plot()
Model_Susy.new_plot()

Model_Standard.plot_pas_rgk5()
# L'évolution du pas est linéaire car il est soumit à certaines contraintes pour 
# ne pas augmenter ou diminuer trop vite
#%% La courbe suivante représente l'évolution de l'erreur calculé en utilisant une 
# norme Root Mean Square en fonction de t = log(Q) pour la méthode de Runge-Kutta adaptative

Model_Standard.new_plot()

Model_Standard.plot_erreur_rgk5()
