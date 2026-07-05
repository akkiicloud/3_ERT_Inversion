# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 17:53:47 2026

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

# ==========================================================
# Load MG field data
# ==========================================================
data_file = r"D:\ERT_2_Pfaffingen\26_MG_Pygimli_file\2026_f3_MG"
data = ert.load(data_file)
print(data)

# ----------------------------------------------------------
# Geometry check (MG)
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 9), dpi=900)
ax.plot(pg.x(data), pg.z(data), "-r")
ax.set_aspect(4)
ax.set_xlabel("Distance (m)", fontsize=16)
ax.set_ylabel("Elevation (m)", fontsize=16)
ax.set_title("MG Elevation Profile – Wendelsheim", fontsize=16)
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

# ==========================================================
# MG INVERSION
# ==========================================================
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
# MG → DD FORWARD MODELLING
# ==========================================================

# ----------------------------------------------------------
# Load DD field data (geometry only)
# ----------------------------------------------------------
dd_file = r"D:\ERT_2_Pfaffingen\26_DD_Pygimli_file\2026_f3_DD"
data_dd = ert.load(dd_file)
print(data_dd)

# ----------------------------------------------------------
# Compute geometric factors and rhoa for DD
# ----------------------------------------------------------
k_dd = ert.createGeometricFactors(data_dd, numerical=True)
data_dd['k'] = k_dd
data_dd['rhoa'] = data_dd['r'] * data_dd['k']

# ----------------------------------------------------------
# Forward simulate DD response using MG model
# ----------------------------------------------------------
mg_model = np.array(mgr.model)
mg_mesh = mgr.paraDomain

data_dd_fwd = ert.simulate(
    mesh=mg_mesh,
    res=mg_model,
    scheme=data_dd,
    verbose=True
)

# ==========================================================
# Plot DD forward pseudosection (MG model)
# ==========================================================
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
    data_dd_fwd,
    vals='rhoa',
    ax=ax,
    label="Apparent resistivity (Ωm)",
    cMap="coolwarm",
    logScale=True,
    cMin=10,
    cMax=150,
    pad= 0.65,
    plotType="scatter"
)

ax.set_xlabel("Distance (m)", fontsize=14)
ax.set_title(f'DD Apparent resistivity: Forward response of MG inversion model', fontsize=16, loc='left')
plt.show()

# ==========================================================
# Plot measured DD pseudosection
# ==========================================================
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
    data_dd,
    vals='rhoa',
    ax=ax,
    label="Apparent resistivity (Ωm)",
    cMap="coolwarm",
    logScale=True,
    cMin=10,
    cMax=150,
    pad= 0.65,
    plotType="scatter"
)

ax.set_xlabel("Distance (m)", fontsize=14)
ax.set_title(f'DD Apparent resistivity: Field measurement', fontsize=16, loc='left')
plt.show()

# ==========================================================
# Difference pseudosection (%)
# ==========================================================

# Set font sizes
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})


rhoa_meas = np.array(data_dd['rhoa'])
rhoa_fwd = np.array(data_dd_fwd['rhoa'])

diff_percent = (rhoa_fwd - rhoa_meas) / rhoa_meas
np.sqrt(np.mean(diff_percent**2)) * 100

# RMS difference 
rms = np.sqrt(np.mean(diff_percent**2)) * 100
print(f"DD forward vs measured RMS difference = {rms:.2f} %")
data_dd['diff_percent'] = diff_percent




fig, ax = plt.subplots(figsize=(10, 4.5), dpi=600)


ert.show(
    data_dd,
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
ax.set_title(f'DD forward (MG model) – DD measured (%)= 0.22', fontsize=16,loc='left')
plt.tight_layout()
plt.show()




















