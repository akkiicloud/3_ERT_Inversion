# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:11:33 2026
This code is for fault zone present 145 to 155 (10 m), not inverting on the real data, whether invering on the 
synthetic model I created, resistivity values are 20, 40 and 100 for each layer.
starting model is vector of res, not the rhoa and the regularization is 20
@author: akagupta
"""

import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pygimli as pg
from pygimli.viewer import showMesh
import json
import pygimli.meshtools as mt
from pygimli.physics import ert as ert
from sklearn.preprocessing import MinMaxScaler


# Uploading the file
data_file = r"D:\ERT_1_Wendelsheim\Pygimli_file\_20250625_f3"
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
                                absoluteUError=0.02,
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
x_right = 958
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
    area=150,
    markerPosition=(500,320)
    
)

# ---------------------------------------------------------------------------
# 5. VISUALIZATION OF THE POLYGON 
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
pg.show(poly, ax=ax)

#ax.plot(x_topo, z_topo, markersize=2) # this line makes it blue
ax.set_xlim(-600, 958)
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
ax.set_xlim(-600, 958)
ax.set_ylim(300, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()

# ---------------------------------------------------------------------------
# 10.ADDING FAULT AND LAYERS IN CORE MODEL
# ---------------------------------------------------------------------------

# -----------------------------------
# Creating our fault zone
# -----------------------------------

fault_left= mt.createLine(start= [145, 399.738],
                          end =  [145, 340 ], nSegments=58)

# Accesing the fault nodes
fault_left_nodes = np.array([node.pos() for node in fault_left.nodes()])

print(fault_left_nodes)

Model_fault= Model+ fault_left+fault_left
Model_fault.addRegionMarker([120,350], marker=3)

fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()



fault_right = mt.createLine(start=[155, 400.502],
                            end=[155, 340], nSegments=58
   
)

# Accessing the fault nodes
fault_right_nodes = np.array([node.pos() for node in fault_right.nodes()])

print(fault_right_nodes)


Model_fault_main= Model_fault+ fault_right
Model_fault_main.addRegionMarker([150,350], marker=4)

fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_main, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()

# -----------------------------------
# Creating our layers with 10m offset
# -----------------------------------

# -------------------------------------------------------------------
# UPPER LAYER - LEFT SIDE
# -------------------------------------------------------------------

layer_01_left = mt.createLine(
    start=[-4, 385.318],
    end=[145, 385.318],
    nSegments=80
)

Model_fault_layer_01_left = (
    Model_fault_main + layer_01_left
)

Model_fault_layer_01_left.addRegionMarker(
    [100, 390],
    marker=5
)

# -------------------------------------------------------------------
# UPPER LAYER - RIGHT SIDE
# -------------------------------------------------------------------

layer_01_right = mt.createLine(
    start=[155, 391.114],
    end=[362, 391.114],
    nSegments=80
)

Model_fault_layer_01_right = (
    Model_fault_layer_01_left
    + layer_01_right
)

Model_fault_layer_01_right.addRegionMarker(
    [250, 397],
    marker=6
)

# -------------------------------------------------------------------
# LOWER LAYER - LEFT SIDE
# -------------------------------------------------------------------

layer_02_left_bottom = mt.createLine(
    start=[-4, 360.599],
    end=[145, 360.599],
    nSegments=80
)

Model_fault_layer_02_left_bottom = (
    Model_fault_layer_01_right
    + layer_02_left_bottom
)

Model_fault_layer_02_left_bottom.addRegionMarker(
    [120, 375],
    marker=7
)

# -------------------------------------------------------------------
# LOWER LAYER - RIGHT SIDE
# -------------------------------------------------------------------

layer_02_right_bottom = mt.createLine(
    start=[155, 366.078],
    end=[362, 366.078],
    nSegments=80
)

Model_fault_layer_02_right_bottom = (
    Model_fault_layer_02_left_bottom
    + layer_02_right_bottom
)

Model_fault_layer_02_right_bottom.addRegionMarker(
    [300, 385],
    marker=8
)


fig, ax = plt.subplots(figsize=(12, 5),dpi=200)
pg.show(Model_fault_layer_02_right_bottom, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(200, 410)
ax.set_xlabel("Distance [m]")
ax.set_ylabel("Elevation [m]")
ax.legend()
plt.show()


# -------------------------------------------------------------------
# VISUALIZATION
# -------------------------------------------------------------------

mesh= mt.createMesh(Model_fault_layer_02_right_bottom,   # we can also use pg.viewer.showmesh
                    quality= 33.7
                    #area=40
                    #paraMaxCellSize=4
                    )  
print(mesh)

fig, ax = plt.subplots(figsize=(12, 5),dpi=200)
pg.show(mesh, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 958)
ax.set_ylim(300, 410)
ax.set_xlabel("Distance [m]")
ax.set_ylabel("Elevation [m]")
ax.legend()
plt.show()


# ---------------------------------------------------------------------------
# 11.ADDING RESISTIVITY VALUE IN MODEL
# ---------------------------------------------------------------------------

res_map = [
    [1, 100.0],   # World background
    [2, 100.0],   
    [3, 100.0],   
    [4, 20.0],
    [5, 20.0],
    [6, 20.0],
    [7, 40],
    [8, 40]
]

fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
pg.show(mesh,
        data=res_map,
        ax=ax,
        showMesh=True,
        logScale=False,
        cMin=10,
        cMax=150,
        cMap="Spectral_r",
        colorBar=True)

ax.set_xlim(-600, 958)
ax.set_ylim(200, 410)
ax.set_xlabel("Distance [m]")
ax.set_ylabel("Elevation [m]")

plt.show()

# ---------------------------------------------------------------------------
# 12. TAKING SPATIAL CONTROL IN MODEL
# ---------------------------------------------------------------------------

res = np.zeros(mesh.cellCount())
len(res)  # this is the total cell in the mesh

for cell in mesh.cells():
    if cell.marker() == 1:
        res[cell.id()] = 100.0
    elif cell.marker() == 2:
        res[cell.id()] = 100.0
    elif cell.marker() == 3:
        res[cell.id()] = 100.0
    elif cell.marker() == 4:
        res[cell.id()] = 20.0
    elif cell.marker() == 5:
        res[cell.id()] = 20.0
    elif cell.marker() == 6:
        res[cell.id()] = 20.0
    elif cell.marker() == 7:
        res[cell.id()] = 40.0
    elif cell.marker() == 8:
        res[cell.id()] = 40.0  
        

for cell in mesh.cells():
    x = cell.center().x()
    z = cell.center().y()

    if -600 <= x <= -4 and 400 >= z >= 385.318:
        res[cell.id()] = 20.0
        
    if -600 <= x <= -4 and 385.318 >= z >= 360.599:
        res[cell.id()] = 40.0 
    
    if -600 <= x <= -4 and 360.599 >= z >= 340:
        res[cell.id()] = 100.0     
    
    if 362 <= x <= 958 and 400 >= z >= 391.114:
        res[cell.id()] = 20.0 
        
    if 362 <= x <= 958 and 391.114 >= z >= 366.078:
        res[cell.id()] = 40.0 
        
    if 362 <= x <= 958 and 366.078 >= z >= 340:
        res[cell.id()] = 100.0     
    
pg.show(mesh,
        data=res,
        ax=ax,
        cMap="Spectral_r",
        cMin=10,
        cMax=150,
        logScale=False,
        colorBar=True,
        label=r'Resistivity ($\Omega$m)',
        orientation='horizontal',
        pad=0.65)
ax.set_xlabel("Distance (m)", fontsize=14),
ax.set_ylabel("Depth (m)", fontsize=14),
ax.set_xlim(-600,600),
ax.set_ylim(200,410)
plt.show()

# ----------------------------------------------------------
# Forward response
# ----------------------------------------------------------
                        
data_fwd = ert.simulate(mesh, 
                           res=res_map, 
                           scheme=data,
                           verbose=True, 
                           #noiseLevel=0.1
                           )



rhoa_meas = np.array(data['rhoa'])
rhoa_fwd = np.array(data_fwd['rhoa'])

# RRMS difference
diff_percent = (rhoa_fwd - rhoa_meas) / rhoa_meas 
np.sqrt(np.mean(diff_percent**2)) * 100

# RMS difference in percent
rms = np.sqrt(np.mean(diff_percent**2))
print(f"forward vs measured RMS difference = {rms:.2f} %")

# Inversion with res and its mesh, not with the real data and its mesh
# mgr.paraDomain does not include the full mesh, it only includes the core polygon which I created.
# FULL MESH
# +-----------------------------+
# | outer world                |
# |  +---------------------+   |
# |  | inversion domain    |   |
# |  |   (core region)     |   |
# |  +---------------------+   |
# +-----------------------------+

mgr = ert.ERTManager(data, verbose=True)
mgr.setMesh(mesh)

# parameter domain cell count
nPara = mgr.paraDomain.cellCount()

print("Mesh cells:", mesh.cellCount())  # total mesh
print("Parameter cells:", nPara)        # parameter cells (cory polygon cells)

nPara = mgr.paraDomain.cellCount()
start_res = pg.Vector(nPara, 100.0)

for cell in mgr.paraDomain.cells():
    x = cell.center().x()
    z = cell.center().y()

    # keep center/fault untouched
    # left block
    if -600 <= x <= -4:
        if 385.318 <= z <= 400:
            start_res[cell.id()] = 20
        elif 360.599 <= z < 385.318:
            start_res[cell.id()] = 40
        elif 340 <= z < 360.599:
            start_res[cell.id()] = 100

    # right block
    elif 362 <= x <= 600:
        if 391.114 <= z <= 400:
            start_res[cell.id()] = 20
        elif 370.251 <= z < 391.114:
            start_res[cell.id()] = 40
        elif 340 <= z < 370.251:
            start_res[cell.id()] = 100

    # middle/fault zone → leave default / preserve

mgr.inv.setDeltaPhiStop(0.5)
mgr.inv.stopAtChi1 = True
mgr.inv.setRegularization(cType=1,
                          #correlationLengths=[50, 10], # Cx, Cz minimum=30, maximum Cx=150, minimum Cz = 2, maximum Cz= max inv. Depth / 3. Cx
                          limits=[1, 1000],
                          trans="log")
                          
mgr.invert(lam=40, 
           maxIter=15,
           startModel=pg.Vector(start_res), 
                      #np.mean(data['rhoa']),
           isReference=True) 

# -----------------------------
# FIT STATISTICS
# -----------------------------
chi2 = mgr.inv.chi2()
rrms = mgr.inv.relrms()
#rms  = mgr.inv.absrms()

print("Chi² =", chi2)
print("RRMS =", rrms, "%")

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
fig, ax1 = plt.subplots(1, 1, figsize=(10, 4.5), dpi=400)

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
        logScale=True)

ax1.set_xlabel("Distance (m)", fontsize=14)
ax1.set_ylabel("Depth (m)", fontsize=14)
plt.tight_layout()
ax1.set_title(f'RRMS = {rrms:.2f}%,  $\chi^2$ = {chi2:.3f}', 
             fontsize=14, loc='left')








































