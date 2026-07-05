# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:36:33 2026

@author: akagupta
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygimli as pg
import json
import pygimli.meshtools as mt
from pygimli.physics import ert as ert
from sklearn.preprocessing import MinMaxScaler
from matplotlib.gridspec import GridSpec

# Uploading the file
data_file = r"D:\ERT_1_Wendelsheim\Pygimli_MG_file\20250625_f1_MG"
data = ert.load(data_file)
print(data)        

# Quick geometry check
fig, ax = plt.subplots(figsize=(14,6), dpi=300)
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
                                absoluteUError=0.04,
                                relativeError=0.03)
ert.show(data, data['err']*100, label="error[%]")
fig = plt.gcf()         
fig.set_size_inches(10, 5)
fig.set_dpi(300) 
print(data['err'])  


# ---------------------------------------------------------------------------
# 1. UPLOADING THE FILE
# ---------------------------------------------------------------------------
elevation =r"D:\ERT_1_Wendelsheim\Profile_25_Wendelsheim\sensors.xlsx"
ele = pd.read_excel(elevation)
print(ele)

x_elec = ele.iloc[:, 0].to_numpy()
z_elec = ele.iloc[:, 2].to_numpy()

# Model extent
x_left  = -600
x_right = 600
z_top_left = 400
z_top_right = 400
z_bottom = -200
n_between = 1

# ---------------------------------------------------------------------------
# 2. FUNCTION TO CREATE MID-POINTS
# ---------------------------------------------------------------------------
def densify_profile(x, z, n_between=1):
    """
    Insert n_between points between each (x, z) pair.
    """
    x_new, z_new = [], []

    for i in range(len(x) - 1):
        x0, z0 = x[i], z[i]
        x1, z1 = x[i + 1], z[i + 1]

        x_new.append(x0)
        z_new.append(z0)

        for k in range(1, n_between + 1):
            t = k / (n_between + 1)
            x_new.append(x0 + t * (x1 - x0))
            z_new.append(z0 + t * (z1 - z0))

    x_new.append(x[-1])
    z_new.append(z[-1])

    return np.array(x_new), np.array(z_new)

# Apply densification
x_topo, z_topo = densify_profile(x_elec, z_elec, n_between=n_between)

# Flexibility moving
x_core_left  = -4
x_core_right = 362

z_core_left  = np.interp(x_core_left,  x_topo, z_topo)
z_core_right = np.interp(x_core_right, x_topo, z_topo)


print(x_topo, z_topo)

# ---------------------------------------------------------------------------
# 3. BUILD POLYGON POINTS
# ---------------------------------------------------------------------------
poly_points = []

# Top-left corner
poly_points.append([x_left, z_top_left])

# Core-left point
poly_points.append([x_core_left, z_core_left])


# Topography (from Wendelsheim file)
for x, z in zip(x_topo, z_topo):
    poly_points.append([x, z])
    
# Core-right point
poly_points.append([x_core_right, z_core_right])
    
# Top-right corner
poly_points.append([x_right, z_top_right])

# Bottom + closure
poly_points.extend([
    [x_right, z_bottom],
    [x_left, z_bottom],
    [x_left, z_top_left] # warning sign reason may be------------------------
])

poly_points = np.array(poly_points)

# ---------------------------------------------------------------------------
# 4. CREATE POLYGON
# ---------------------------------------------------------------------------
poly = mt.createPolygon(
    poly_points,
    isClosed=True,
    marker=1,
    boundaryMarker=1,
    area=300,
    markerPosition=(500,320)
    
)

# ---------------------------------------------------------------------------
# 5. VISUALIZATION OF THE POLYGON 
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
pg.show(poly, ax=ax)

#ax.plot(x_topo, z_topo, markersize=2) # this line makes it blue
ax.set_xlim(-600, 600)
ax.set_ylim(-200,410)
ax.legend()
ax.set_xlabel("x [m]")
ax.set_ylabel("Elevation [m]")
plt.show()


# ---------------------------------------------------------------------------
# 6. BUILDING THE CORE MODEL
# ---------------------------------------------------------------------------

# Define the extent
x_core_left = -4
x_core_right = 362
z_core_left = 394.25
z_core_right = 400.26
z_core_bottom = 340

poly_core_points = []

# Top-left corner
poly_core_points.append([x_core_left, z_core_left])


# Topography (from Wendelsheim file)
for x, z in zip(x_topo, z_topo):
    poly_core_points.append([x, z])
    
# Core-right point
poly_core_points.append([x_core_right, z_core_right])
    

# Bottom + closure (correct order)
poly_core_points.extend([
    [x_core_right, z_core_bottom],   # bottom right
    [x_core_left,  z_core_bottom],   # bottom left
    [x_core_left,  z_core_left]      # close polygon
])

poly_core_points = np.array(poly_core_points)


# ---------------------------------------------------------------------------
# 7. CREATE CORE-POLYGON
# ---------------------------------------------------------------------------
poly_core = mt.createPolygon(
    poly_core_points,
    isClosed=True,
    marker=2,
    boundaryMarker=2,
    area=30,
    markerPosition=(250,355)
    
)

# ---------------------------------------------------------------------------
# 8. VISUALIZATION OF THE CORE POLYGON
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
pg.show(poly_core, ax=ax)

#ax.plot(x_topo, z_topo, markersize=2) # this line makes it blue
ax.set_xlim(-4, 362)
ax.set_ylim(340,410)
ax.legend()
ax.set_xlabel("x [m]")
ax.set_ylabel("Elevation [m]")
plt.show()


# ---------------------------------------------------------------------------
# 9. MERGING CORE MODEL AND MODEL
# ---------------------------------------------------------------------------

Model = poly + poly_core

fig, ax = plt.subplots(figsize=(12, 5),dpi=200)
pg.show(Model, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(300, 410)
ax.set_xlabel("Distance [m]")
ax.set_ylabel("Elevation [m]")
ax.legend()
plt.show()


# ---------------------------------------------------------------------------
# 10.ADDING FAULT AND LAYERS IN CORE MODEL
# ---------------------------------------------------------------------------

# -----------------------------------
# Creating our fault
# -----------------------------------

fault_line= mt.createLine(start= [150, 400.122],
                          end =  [150, 340 ], nSegments=58)

# Accesing the fault nodes
fault_nodes = np.array([node.pos() for node in fault_line.nodes()])

print(fault_nodes)


Model_fault= Model+ fault_line
Model_fault.addRegionMarker([80,350], marker=3)

fig, ax = plt.subplots(figsize=(12, 5),dpi=200)
pg.show(Model_fault, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(300, 410)
ax.set_xlabel("Distance [m]")
ax.set_ylabel("Elevation [m]")
ax.legend()
plt.show()

# -----------------------------------
# Creating our layers with 10m offset
# -----------------------------------

layer_01_left= mt.createLine(start= [-4, 385.6097931],
                        end =  [150,385.6097931 ], nSegments=80)

Model_fault_layer_01_left= Model_fault+layer_01_left
Model_fault_layer_01_left.addRegionMarker([100,393], marker=4)




layer_01_right= mt.createLine(start= [150, 391.829 ],
                        end =  [362,391.829 ], nSegments=80)

Model_fault_layer_01_right= Model_fault_layer_01_left+layer_01_right
Model_fault_layer_01_right.addRegionMarker([200,397], marker=5)



layer_02_left_bottom= mt.createLine(start= [-4, 360.73172414 ],
                        end =  [150,360.73172414 ], nSegments=80)

Model_fault_layer_02_left_bottom= Model_fault_layer_01_right+layer_02_left_bottom
Model_fault_layer_02_left_bottom.addRegionMarker([120,375], marker=6)




layer_02_right_bottom= mt.createLine(start= [150, 370.061 ],
                        end =  [362,370.061 ], nSegments=80)

Model_fault_layer_02_right_bottom= Model_fault_layer_02_left_bottom+layer_02_right_bottom
Model_fault_layer_02_right_bottom.addRegionMarker([300,385], marker=7)


fig, ax = plt.subplots(figsize=(12, 5),dpi=200)
pg.show(Model_fault_layer_02_right_bottom, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(200, 410)
ax.set_xlabel("Distance [m]")
ax.set_ylabel("Elevation [m]")
ax.legend()
plt.show()

paraDepth = 60
area = 1000
mesh= mt.createMesh(Model_fault_layer_02_right_bottom,   # we can also use pg.viewer.showmesh
                    quality= 33.7,
                    paraDepth=paraDepth,
                    paraMaxCellSize=20
                    )  
print(mesh)

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
#ax1.set_xticks(np.arange(0, 360, 20))
#ax1.set_yticks(np.arange(340, 410, 10))
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













































