# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:36:49 2026

@author: akagupta
"""

import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import json

# ----------------------------------------------------------
# Global plot styling
# ----------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# ----------------------------------------------------------
# 1. Load ERT data
# ----------------------------------------------------------
data_file = r"D:\ERT_2_Pfaffingen\26_DD_Pygimli_file\2026_f3_DD"
data = ert.load(data_file)
print(data)

# ----------------------------------------------------------
# 2. Geometry check
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 9), dpi=400)
ax.plot(pg.x(data), pg.z(data), "-r")
ax.set_aspect(4)
ax.set_xlabel("Distance (m)", fontsize=16)
ax.set_ylabel("Elevation (m)", fontsize=16)
ax.set_title("Elevation profile – Wendelsheim", fontsize=16)
plt.grid()
plt.show()

# ----------------------------------------------------------
# 3. Geometric factor & error model
# ----------------------------------------------------------
k_num = ert.createGeometricFactors(data, numerical=True)
k_ana = ert.createGeometricFactors(data, numerical=False)

ert.show(data, vals=k_ana/k_num,
         label='Topographic Effect',
         cMap="bwr", cMin=0.8, cMax=1.25, logScale=True)

fig = plt.gcf()
fig.set_size_inches(10, 5)
fig.set_dpi(400)

data['k'] = k_num
data['rhoa'] = data['r'] * data['k']
data['err'] = ert.estimateError(data,
                                 absoluteUError=0.02,
                                 relativeError=0.03)

ert.show(data, data['err']*100, label="error[%]")

fig = plt.gcf()
fig.set_size_inches(10, 5)
fig.set_dpi(400)

print(data['err'])

# ----------------------------------------------------------
# 4. Create inversion mesh
# ----------------------------------------------------------
paraDepth = 60
area = 1000

mesh = mt.createParaMesh(
    data,
    quality=33.7,
    paraDepth=paraDepth,
    paraMaxCellSize=20
)

fig, ax = plt.subplots(figsize=(8, 4), dpi=400)
pg.show(mesh, ax=ax, markers=False)
ax.set_title("Inversion mesh")
plt.show()

# ----------------------------------------------------------
# 5. ERT Manager
# ----------------------------------------------------------
mgr = ert.ERTManager(data)
mgr.setMesh(mesh)

n_cells = mgr.paraDomain.cellCount()
10**np.mean(np.log10(data['rhoa'])) # this is the mean value which pygimli gives or homogeneous starting model


# ----------------------------------------------------------
# 6. START MODEL = 5 Ωm
# ----------------------------------------------------------
start_model_5 = np.ones(n_cells) * 5

mgr.invert(startModel=start_model_5,
           lam=10,
           zWeight=1,
           verbose=True)

model_5 = mgr.model.copy()
chi2_5 = mgr.inv.chi2()
rrms_5 = mgr.inv.relrms()

# ----------------------------------------------------------
# 7. START MODEL = 26.47 Ωm
# ----------------------------------------------------------
start_model_30 = np.ones(n_cells) * 30

mgr.invert(startModel=start_model_30,
           lam=10,
           zWeight=1,
           verbose=True)

model_30 = mgr.model.copy()
chi2_30 = mgr.inv.chi2()
rrms_30 = mgr.inv.relrms()

# ----------------------------------------------------------
# 8. START MODEL = 200 Ωm
# ----------------------------------------------------------
start_model_200 = np.ones(n_cells) * 200.0

mgr.invert(startModel=start_model_200,
           lam=10,
           zWeight=1,
           verbose=True)

model_200 = mgr.model.copy()
chi2_200 = mgr.inv.chi2()
rrms_200 = mgr.inv.relrms()

# ----------------------------------------------------------
# RMS summary
# ----------------------------------------------------------
print("===================================")
print("DD FORCED INVERSION RESULTS")
print(f"RMS (5 Ωm)   : {rrms_5:.2f}%")
print(f"RMS (30 Ωm)  : {rrms_30:.2f}%")
print(f"RMS (200 Ωm) : {rrms_200:.2f}%")
print("===================================")

# ==========================================================
# FIGURE A → (6 + 11)
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=400)

pg.show(mgr.paraDomain,
        model_5,
        ax=ax,
        cMin=10,
        cMax=150,
        cMap='Spectral_r',
        label=r'Resistivity ($\Omega$m)',
        orientation='horizontal',
        pad=0.65,
        logScale=True,
        coverage=mgr.coverage())

ax.set_xlabel("Distance (m)")
ax.set_ylabel("Depth (m)")
ax.set_title(f"RRMS = {rrms_5:.2f}%, χ² = {chi2_5:.3f} ", loc='left')



model_diff = ((model_30 - model_5) / model_30) * 100

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=400)
pg.show(
    mgr.paraDomain,
    model_diff,
    ax=ax,
    cMap="bwr",
    cMin=-50,
    cMax=50,
    orientation='horizontal',
    pad=0.65,
    label="Difference in Resistivity (Ωm)"
)
ax.set_title(f'Normalized difference model (30 Ωm – 5 Ωm)',fontsize=14, loc='left')
ax.set_xlabel("Distance (m)", fontsize=14)
ax.set_ylabel("Depth (m)", fontsize=14)
plt.show()


# ==========================================================
# FIGURE B → (7 + 10)
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=400)

pg.show(mgr.paraDomain,
        model_30,
        ax=ax,
        cMin=10,
        cMax=150,
        cMap='Spectral_r',
        label=r'Resistivity ($\Omega$m)',
        orientation='horizontal',
        pad=0.65,
        logScale=True,
        coverage=mgr.coverage())

ax.set_xlabel("Distance (m)")
ax.set_ylabel("Depth (m)")
ax.set_title(f"RRMS = {rrms_30:.2f}%, χ² = {chi2_30:.3f}", loc='left')

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=400)


model_diff = ((model_200 - model_5) / model_200) * 100

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=400)
pg.show(
    mgr.paraDomain,
    model_diff,
    ax=ax,
    cMap="bwr",
    cMin=-50,
    cMax=50,
    orientation='horizontal',
    pad=0.65,
    label="Difference in Resistivity (Ωm)"
)
ax.set_title(f'Normalized difference model (200 Ωm – 5 Ωm)',fontsize=14, loc='left')
ax.set_xlabel("Distance (m)", fontsize=14)
ax.set_ylabel("Depth (m)", fontsize=14)
plt.show()

# ==========================================================
# FIGURE C → (8 + 9)
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=400)

pg.show(mgr.paraDomain,
        model_200,
        ax=ax,
        cMin=10,
        cMax=150,
        cMap='Spectral_r',
        label=r'Resistivity ($\Omega$m)',
        orientation='horizontal',
        pad=0.65,
        logScale=True,
        coverage=mgr.coverage())

ax.set_xlabel("Distance (m)")
ax.set_ylabel("Depth (m)")
ax.set_title(f"RRMS = {rrms_200:.2f}%, χ² = {chi2_200:.3f}", loc='left')

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=400)

model_diff = ((model_200 - model_30) / model_200) * 100

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=400)
pg.show(
    mgr.paraDomain,
    model_diff,
    ax=ax,
    cMap="bwr",
    cMin=-50,
    cMax=50,
    orientation='horizontal',
    pad=0.65,
    label="Difference in Resistivity (Ωm)"
)
ax.set_title(f' Normalized difference model (200 Ωm – 30 Ωm)',fontsize=14, loc='left')
ax.set_xlabel("Distance (m)", fontsize=14)
ax.set_ylabel("Depth (m)", fontsize=14)
plt.show()
