import numpy as np
#Variables
mz = 91.1876 # Z boson mass
mgut = 1.5*10**16 # GUT energy
mh = 125.38 # Higgs mass
mt = 172.76 # Top quark mass
v = 246.21 # quantum vacuum stability
lambda0 = mh**2/(2*v**2) # lambda Higgs parameter at mZ energy scale 
y0 = mt*np.sqrt(2)/v # Yukawa top quark term a mZ energy scale
alphaQED = 1/127.926 # mz QED coupling constant 
sin2thetaW = 0.23121 
alpha1 = (alphaQED/(1-sin2thetaW))*(5/3)
alpha2 = alphaQED/sin2thetaW
alpha3 = 0.1179
b_standardModel = np.array([41/10, -19/6, -7])
b_SUSYModel = np.array([33/5, 1, -3])
t0 = np.log10(mz)