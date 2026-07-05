# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 16:24:23 2025

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
import matplotlib.colors as mcolors

# ----------------------------------------------------------
# Uploading the DD file
# ----------------------------------------------------------
data_file = r"D:\ERT_2_Pfaffingen\26_DD_Pygimli_file\2026_f3_DD"
data = ert.load(data_file)
print(data)

# ----------------------------------------------------------
# Quick geometry check
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 9), dpi=900)
ax.plot(pg.x(data), pg.z(data), "-r")
ax.set_aspect(4)
ax.set_xlabel("Distance(x) (m)", fontsize=16)
ax.set_ylabel("Elevation(z) (m)", fontsize=16)
ax.set_title("Elevation Profile_Wendelsheim_250625", fontsize=16)
plt.grid()
plt.show()

# ----------------------------------------------------------
# Computing geometric factors
# ----------------------------------------------------------
k_num = ert.createGeometricFactors(data, numerical=True)
k_ana = ert.createGeometricFactors(data, numerical=False)

ert.show(
    data,
    vals=k_ana / k_num,
    label="Topographic Effect",
    cMap="bwr",
    cMin=0.8,
    cMax=1.25,
    logScale=True
)
fig = plt.gcf()
fig.set_size_inches(10, 5)
fig.set_dpi(900)

# ----------------------------------------------------------
# Error model
# ----------------------------------------------------------
data['k'] = k_num
data['rhoa'] = data['r'] * data['k']
data['err'] = ert.estimateError(
    data,
    absoluteUError=0.02,
    relativeError=0.03
)

ert.show(data, data['err'] * 100, label="error[%]")
fig = plt.gcf()
fig.set_size_inches(10, 5)
fig.set_dpi(900)

# ----------------------------------------------------------
# Creating inversion mesh
# ----------------------------------------------------------
paraDepth = 60

mesh = mt.createParaMesh(
    data,
    quality=33.7,
    paraDepth=paraDepth,
    paraMaxCellSize=20
)

print(mesh)

fig, ax = plt.subplots(figsize=(8, 4), dpi=900)
pg.show(mesh, ax=ax, markers=False)
ax.set_xlim(-1000, 1000)
ax.set_ylim(-400, 400)
plt.show()

# ----------------------------------------------------------
# DD INVERSION
# ----------------------------------------------------------
mgr = ert.ERTManager(data)
mgr.setMesh(mesh)

mgr.invert(
    verbose=True,
    lam=10,
    zWeight=1
)

mgr.showResult(
    cMin=10,
    cMax=150,
    xlabel="x(m)",
    ylabel="z(m)"
)
fig = plt.gcf()
fig.set_size_inches(10, 5)
fig.set_dpi(900)

mgr.showFit()
mgr.showResultAndFit()
fig = plt.gcf()
fig.set_size_inches(10, 5)
fig.set_dpi(900)

# ==========================================================
# DD → MG FORWARD MODELLING, has to clear something here..
# ==========================================================

# ----------------------------------------------------------
# Load MG field data (geometry)
# ----------------------------------------------------------
mg_file = r"D:\ERT_2_Pfaffingen\26_MG_Pygimli_file\2026_f3_MG"
data_mg = ert.load(mg_file)
print(data_mg)

# ----------------------------------------------------------
# Compute geometric factor and rhoa for MG
# ----------------------------------------------------------
k_mg = ert.createGeometricFactors(data_mg, numerical=True)

data_mg['k'] = k_mg
data_mg['rhoa'] = data_mg['r'] * data_mg['k']


# ----------------------------------------------------------
# Forward simulate MG response using DD model
# ----------------------------------------------------------
dd_model = np.array(mgr.model)
dd_mesh = mgr.paraDomain

data_mg_fwd = ert.simulate(
    mesh=dd_mesh,
    res=dd_model,
    scheme=data_mg,
    verbose=True
)

# ----------------------------------------------------------
# Plot MG forward pseudosection (DD model)
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=600)

# Set font sizes
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

pg.show(
    data_mg_fwd,
    vals='rhoa',          # use key, safer
    ax=ax,
    label="Apparent resistivity (Ωm)",
    cMap="coolwarm",
    logScale=True,
    cMin=10,              # ← minimum resistivity
    cMax=150,
    pad= 0.65,                      
    plotType="scatter"
)

ax.set_xlabel("Distance (m)", fontsize=14)
ax.set_title(f'MG Apparent resistivity: Forward response of DD inversion model', fontsize=16, loc='left')
plt.tight_layout()
plt.show()


# ----------------------------------------------------------
# Plot measured MG pseudosection
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=600)

# Set font sizes
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

ert.show(
    data_mg,
    vals=data_mg['rhoa'],
    ax=ax,
    label="Apparent resistivity (Ωm)",
    cMap="coolwarm",
    logScale=True,
    cMin=10,              # ← minimum resistivity
    cMax=150, 
    pad= 0.65, 
    plotType="scatter"
)

ax.set_xlabel("Distance (m)", fontsize=14)
ax.set_title(f'MG Apparent resistivity: Field measurement', fontsize=16, loc='left')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Difference pseudosection (%)
# ----------------------------------------------------------


# Set font sizes
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

rhoa_meas = np.array(data_mg['rhoa'])
rhoa_fwd = np.array(data_mg_fwd['rhoa'])

diff_percent = (rhoa_fwd - rhoa_meas) / rhoa_meas
np.sqrt(np.mean(diff_percent**2)) * 100

# RMS difference 
rms = np.sqrt(np.mean(diff_percent**2)) * 100
print(f"MG forward vs measured RMS difference = {rms:.2f} %")

data_mg['diff_percent'] = diff_percent

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=600)


ert.show(
    data_mg,
    vals=diff_percent,
    ax=ax,
    label="Relative Misfit (%)",
    cMap="bwr",
    cMin=-0.1,
    cMax=0.1,
    pad=0.65,
    logScale=False,
    plotType="scatter"
)

ax.set_xlabel("Distance (m)", fontsize=14)
ax.set_title(f'MG forward (DD model) – MG measured (%)= 0.12', fontsize=16,loc='left')
plt.tight_layout()
plt.show()










