# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:56:17 2026

@author: akagupta
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert as ert
from sklearn.preprocessing import MinMaxScaler

# Uploading the file
data_file =r"D:\ERT_2_Pfaffingen\26_MG_Pygimli_file\2026_f3_MG"
data = ert.load(data_file)
print(data)        
 

# Quick geometry check
fig, ax = plt.subplots(figsize=(14,6),dpi=400)
ax.plot(pg.x(data), pg.z(data), "-r")
ax.set_aspect(4)
ax.set_xlabel("Distance(x) (m)", fontsize= 16)
ax.set_ylabel("Elevation(z) (m)",fontsize= 16)
ax.set_title("Elevation Profile_Wendelsheim_250625",fontsize= 16)
#ax.set_ylim(350,360)
plt.grid()
plt.show()
 
# Computing k
k_num= ert.createGeometricFactors(data, numerical=True) 
k_ana= ert.createGeometricFactors(data, numerical=False)

# Topography Effect or Ratio of Geometric Error
ert.show(data, vals=k_ana/k_num, label= 'Topographic Effect',
         cMap= "bwr", cMin= 0.8, cMax= 1.25, logScale=True);
fig = plt.gcf()         
fig.set_size_inches(10, 5)   # width=10in, height=5in
fig.set_dpi(900)   

# Error model in Pseudo-Section
data['k']= k_num
data['rhoa'] = data['r'] * data['k'] # creating for correlation length
data['err']= ert.estimateError(data,
                               absoluteUError=0.04, # 50uV,  Instrumental error in voltage
                               relativeError= 0.02) # 3%, Electrode contact resistance error or proportaional uncertainity.
ert.show(data, data['err']*100, label="error[%]");
fig = plt.gcf()         
fig.set_size_inches(10, 5)   # width=10in, height=5in
fig.set_dpi(400) 
print(data['err'])  

# Creating a Mesh
paraDepth= 60
area = 1000
mesh = mt.createParaMesh(data,
                         quality= 33.7,
                         paraDepth= paraDepth,
                         paraMaxCellSize=20,
                         #area = area
                         )
print(mesh)  # The inversion mesh

fig, ax = plt.subplots(figsize=(8, 4), dpi=400)
pg.show(mesh, ax = ax, markers= False)
ax.set_xlim(-1000, 1000) 
ax.set_ylim(-400, 400)

# INVERSION

# mgr= ert.ERTManager(data)
# mgr.setMesh(mesh)
# mgr.invert(verbose=True, lam=10, zWeight= 1)
# mgr.showResult(cMin=30, cMax= 100, xlabel= "x(m)", ylabel= "z(m)");
# fig = plt.gcf()         
# fig.set_size_inches(10, 5)   # width=10in, height=5in
# fig.set_dpi(300)

# Inversion with correlation model

mgr = ert.ERTManager(data, verbose=True)
mgr.setMesh(mesh)
mgr.inv.setDeltaPhiStop(0.5)
mgr.inv.stopAtChi1 = True
mgr.inv.setRegularization(cType=2,
                          correlationLengths=[35, 10], # Cx, Cz minimum=30, maximum Cx=150, minimum Cz = 2, maximum Cz= max inv. Depth / 3. Cx
                          limits=[1, 1000],
                          startModel=np.mean(data['rhoa']),
                          trans="log")
mgr.invert(lam=10, maxIter=15) 

# Extract inversion metrics
chi2 = mgr.inv.chi2()
rrms = mgr.inv.relrms()  # or mgr.inv.absrms() for absolute RMS

print(f"Chi-squared: {chi2:.3f}")
print(f"RRMS: {rrms:.2f}%")


# -------------------------------------------------------------------
# Vertical plotting together
# -------------------------------------------------------------------


# Set font sizes
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# Create figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), dpi=400)

# Top plot: Inversion result
pg.show(mgr.paraDomain, 
        mgr.model,
        ax=ax1,
        cMin=10, 
        cMax=150,
        cMap='Spectral_r',
        label=r'Resistivity ($\Omega$m)',
        orientation='horizontal',
        pad=0.65,
        logScale=True,
        coverage=mgr.coverage())
ax1.set_xlabel("Distance (m)", fontsize=14)
ax1.set_ylabel("Depth (m)", fontsize=14)
#ax1.set_xticks(np.arange(0, 360, 20))
#ax1.set_yticks(np.arange(340, 410, 10))
ax1.set_title(f'RRMS = {rrms:.2f}%,  $\chi^2$ = {chi2:.3f}', 
             fontsize=14, loc='left')

# Bottom plot: Residuals
data_sim = ert.simulate(mesh=mgr.paraDomain, res=mgr.model, scheme=data, verbose=False)
def residuals(data_meas, data_sim):
    meas = np.array(data_meas['rhoa'])
    sim = np.array(data_sim['rhoa'])
    res = (sim - meas) / meas * 100
    return res

res_percent = residuals(data, data_sim)

ert.show(
    data,
    vals=res_percent,    
    ax=ax2,
    label="Residuals (%)",
    cMap="coolwarm",
    cMin=-10,
    cMax=10,
    logScale=False,
    pad=0.65,
    plotType="scatter",
    marker='o',
    markersize=10
)
ax2.set_xlabel("Distance (m)", fontsize=14)
#ax2.set_ylabel("Configuration", fontsize=14)

plt.tight_layout()
plt.savefig('inversion_and_residuals_vertical.png', dpi=300, bbox_inches='tight')
plt.show()
 
