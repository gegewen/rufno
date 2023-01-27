import numpy as np
import time
from scipy.interpolate import griddata
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def grid_perm_map(ungridded_perm_map):
    y_uni = np.linspace(0,200,96)
    x_uni = np.linspace(3.5938,27826*3.5938,27826)
    x_uni, y_uni = np.meshgrid(x_uni, y_uni)

    x_irr = np.cumsum(1.035012 ** np.arange(200) * 3.5938)
    y_irr = np.linspace(0,200,96) 
    x_irr, y_irr = np.meshgrid(x_irr, y_irr)
    shale = griddata((x_uni.flatten(), y_uni.flatten()), ungridded_perm_map.flatten(), (x_irr, y_irr), method='nearest')
    
    return shale

def generate_porosity_map(k_r, k_z):
    k_avg = (k_z + k_r)/2
    porosity = np.zeros(k_avg.shape)

    for j in range(200):
        root = fsolve(func, np.ones(k_avg[:96, j].shape), args=k_avg[:96, j])
        porosity[:96, j] = root
        
    porosity = porosity + np.random.normal(0,0.001,(96,200))
    
    return porosity

def nm2tomD(k_nm2):
    k_m2 = k_nm2 * 1e-18
    k_D = k_m2 / (0.986923 * 1e-12)
    k_md = 1000 * k_D
    return k_md

def func(phi, c):
    k =  31*phi + 7467*phi**2 + 191*(10*phi) ** 10
    return nm2tomD(k) - c