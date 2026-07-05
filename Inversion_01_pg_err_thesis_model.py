# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:21:29 2026

@author: akagupta
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import json
import pygimli.meshtools as mt
from pygimli.physics import ert as ert
from sklearn.preprocessing import MinMaxScaler
from matplotlib.gridspec import GridSpec

# Uploading the file
data_file = r"D:\ERT_2_Pfaffingen\26_MG_Pygimli_file\2026_f3_MG"
data = ert.load(data_file)
print(data)        

# Quick geometry check
fig, ax = plt.subplots(figsize=(14,6), dpi=400)
ax.plot(pg.x(data), pg.z(data), "-r")
ax.set_aspect(4)
ax.set_xlabel("Distance(x) (m)", fontsize=16)
ax.set_ylabel("Elevation(z) (m)", fontsize=16)
ax.set_title("Elevation Profile_Pfäffingen_13 and 14 April", fontsize=16)
ax.grid()
plt.show()
 
# Computing k
k_num = ert.createGeometricFactors(data, numerical=True) 
k_ana = ert.createGeometricFactors(data, numerical=False)

# Topography Effect or Ratio of Geometric Error
ert.show(data, vals=k_ana/k_num, label='Topographic Effect',
         cMap="bwr", cMin=0.8, cMax=1.25, logScale=True)
fig = plt.gcf()         
fig.set_size_inches(10, 5)
fig.set_dpi(300)   

# Error model in Pseudo-Section
data['k'] = k_num
data['rhoa'] = data['r'] * data['k']
print(np.min(data['rhoa']), np.max(data['rhoa']))
data['err'] = ert.estimateError(data,
                                absoluteUError=0.09,
                                relativeError=0.02)
ert.show(data, data['err']*100, label="error[%]")
fig = plt.gcf()         
fig.set_size_inches(10, 4.5)
fig.set_dpi(400) 
print(data['err'])  

# Creating a Mesh
paraDepth = 60
area = 1000
mesh = mt.createParaMesh(data,
                         quality=33.7,
                         paraDepth=paraDepth,
                         paraMaxCellSize=20)
print(mesh)

fig, ax = plt.subplots(figsize=(8, 4), dpi=400)
pg.show(mesh, ax=ax, markers=False)
ax.set_xlim(-1000, 1000) 
ax.set_ylim(-400, 400)
plt.show()

# INVERSION
mgr = ert.ERTManager(data)
mgr.setMesh(mesh)
mgr.invert(verbose=True, lam=10, zWeight=1)

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
ax1.set_xticks(np.arange(0, 360, 20))
ax1.set_yticks(np.arange(340, 410, 10))
ax1.set_title(f'RRMS = {rrms:.2f}%,  $\chi^2$ = {chi2:.3f}', 
             fontsize=16, loc='left')


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

plt.tight_layout()
plt.savefig('inversion_and_residuals_vertical.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------
# Vertical plotting separately wherever required
# -------------------------------------------------------------------

# Set font sizes
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# Calculate residuals 
data_sim = ert.simulate(mesh=mgr.paraDomain, res=mgr.model, scheme=data, verbose=False)
def residuals(data_meas, data_sim):
    meas = np.array(data_meas['rhoa'])
    sim = np.array(data_sim['rhoa'])
    res = (sim - meas) / meas * 100
    return res
res_percent = residuals(data, data_sim)

# ==================== FIGURE 1: Inversion Result ====================
fig1, ax1 = plt.subplots(figsize=(10, 4.5), dpi=400)

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
ax1.set_title(f'RRMS = {rrms:.2f}%,  $\chi^2$ = {chi2:.3f}', 
             fontsize=14, loc='left')

plt.tight_layout()
plt.savefig('inversion_model.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== FIGURE 2: Residuals ====================
fig2, ax2 = plt.subplots(figsize=(10, 4.5), dpi=400)

ert.show(
    data,
    vals=res_percent,    
    ax=ax2,
    label="Residuals (%)",
    cMap="coolwarm",
    cMin=-10,
    cMax=10,
    logScale=False,
    pad=0.65
)
ax2.set_xlabel("Distance (m)", fontsize=14)
#ax2.set_ylabel("Depth (m)", fontsize=14)

plt.tight_layout()
plt.savefig('residuals.png', dpi=300, bbox_inches='tight')
plt.show()


# -------------------------------------------------------------------
# Sensitivity coverage
# -------------------------------------------------------------------
# Set font sizes
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

coverage = mgr.coverage() 
mesh_base = mgr.paraDomain
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=400)
pg.show(
    mesh_base,
    coverage,
    ax=ax,
    label="Sensitivity",
    cMap="viridis",
    cMin=-2.5,
    cMax=2.2,
    logScale=False,
    pad=0.65
)
#ax.set_xticks(np.arange(0, 358))
ax.set_xlabel("Distance (m)", fontsize=14)
ax.set_ylabel("Depth (m)", fontsize=14)
plt.tight_layout()
plt.savefig('sensitivity.png', dpi=300, bbox_inches='tight')
print(min(coverage), max(coverage))
plt.show()


