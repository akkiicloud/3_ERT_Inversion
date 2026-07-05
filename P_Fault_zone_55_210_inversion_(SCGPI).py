# -*- coding: utf-8 -*-
"""
Created on Sun May 24 20:15:59 2026
This code is for creating two fault zone at (55-85m) and (210 to 235m), inverting the data on the mesh of res and 
also including the value of res, not on the pygimli defauult mesh.

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
data_file = r"D:\ERT_2_Pfaffingen\26_Pygimli_file\2026_f3"
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
elevation =r"D:\ERT_2_Pfaffingen\Profile_26_Pfaffingen\sensors_pfaffingen_combined.xlsx"
ele = pd.read_excel(elevation)
print(ele)

x_elec = ele.iloc[:, 0].to_numpy()
z_elec = ele.iloc[:, 2].to_numpy()


# Model extent
x_left  = -600
x_right = 958
z_top_left = 352.850
z_top_right = 355.776
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
ax.set_ylim(-200,380)
ax.set_yticks(np.arange(-200, 380, 50))
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
z_core_left = 352.850
z_core_right = 355.776
z_core_bottom = 300

poly_core_points = []

# Top-left corner
poly_core_points.append([x_core_left, z_core_left])


# Topography (from Pfaffingen file)
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
    markerPosition=(300,310)
    
)

# ---------------------------------------------------------------------------
# 8. VISUALIZATION OF THE CORE POLYGON
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
pg.show(poly_core, ax=ax)

#ax.plot(x_topo, z_topo, markersize=2) # this line makes it blue
ax.set_xlim(-4, 362)
ax.set_ylim(300,380)
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
ax.set_ylim(-200, 380)
ax.set_yticks(np.arange(-200, 380, 50))
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()


# ---------------------------------------------------------------------------
# 10.ADDING FAULT AND LAYERS IN CORE MODEL
# ---------------------------------------------------------------------------

# --------------------------------------------
# Creating our fault zone (210 to 235) m
# --------------------------------------------

fault_210= mt.createLine(start= [210, 345.517 ],     #353.247
                          end = [210, 300 ], nSegments=58)

# Accesing the fault nodes
fault_210_nodes = np.array([node.pos() for node in fault_210.nodes()])
print(fault_210_nodes)
Model_fault_210= Model+ fault_210
#Model_fault_210.addRegionMarker([120,320], marker=3)
fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_210, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()



fault_235 = mt.createLine(start=[235, 345.517],   #353.341 
                            end=[235, 300], nSegments=58)
# Accessing the fault nodes
fault_235_nodes = np.array([node.pos() for node in fault_235.nodes()])
print(fault_235_nodes)
Model_fault_235= Model_fault_210+ fault_235
#Model_fault_235.addRegionMarker([220,320], marker=4)
fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_235, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()


fault_close_second = mt.createLine(start=[210, 345.517],
                                   end= [235,345.517], nSegment= 10)

# Accessing the fault nodes
fault_close_second_nodes = np.array([node.pos() for node in fault_close_second.nodes()])
Model_fault_close_2= Model_fault_235+fault_close_second
Model_fault_close_2.addRegionMarker([220,320], marker=3)
fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_close_2, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()


# --------------------------------------------
# Creating our fault zone (55 to 85)m
# --------------------------------------------

fault_55= mt.createLine(start= [55, 345.517],
                          end =  [55, 300 ], nSegments=58)

# Accesing the fault nodes
fault_55_nodes = np.array([node.pos() for node in fault_55.nodes()])
print(fault_55_nodes)
Model_fault_55= Model_fault_close_2+ fault_55
#Model_fault_55.addRegionMarker([40,320], marker=5)
fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_55, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()



fault_85 = mt.createLine(start=[85, 345.517],
                            end=[85, 300], nSegments=58)

# Accessing the fault nodes
fault_85_nodes = np.array([node.pos() for node in fault_85.nodes()])
print(fault_85_nodes)
Model_fault_85= Model_fault_55+ fault_85
#Model_fault_85.addRegionMarker([65,320], marker=6)
fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_85, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()



fault_close_first = mt.createLine(start=[55, 345.517],
                                   end= [85,345.517], nSegment= 10)
fault_close_first_nodes = np.array([node.pos() for node in fault_close_first.nodes()])
print(fault_close_first_nodes)
Model_fault_close_first= Model_fault_85+ fault_close_first
Model_fault_close_first.addRegionMarker([65,320], marker=4)
fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_close_first, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()



# -----------------------------------------------------------------------------
# Creating our layers with 10m offset for 55 m fault
# -----------------------------------------------------------------------------

# -------------------------------------------------------------------
# UPPER LAYER - LEFT SIDE of 55
# -------------------------------------------------------------------

layer_01_left = mt.createLine(
    start=[-4, 345.517],
    end=[55, 345.517],
    nSegments=20
)

Model_fault_layer_01_left = (
    Model_fault_close_first + layer_01_left
)

Model_fault_layer_01_left.addRegionMarker(
    [45, 340],
    marker=5
)

fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_layer_01_left, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()


# -------------------------------------------------------------------
# LOWER LAYER - LEFT SIDE of 55
# -------------------------------------------------------------------

layer_02_left_bottom = mt.createLine(
    start=[-4, 335.586],
    end=[55, 335.586],
    nSegments=20
)

Model_fault_layer_02_left_bottom = (
    Model_fault_layer_01_left
    + layer_02_left_bottom
)

Model_fault_layer_02_left_bottom.addRegionMarker(
    [30, 328],
    marker=6)

fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_layer_02_left_bottom, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()


# -------------------------------------------------------------------
# BOTTOM LOWER LAYER - LEFT SIDE of 55
# -------------------------------------------------------------------

layer_03_left_bottom = mt.createLine(
    start=[-4, 320.69],
    end=[55, 320.69],
    nSegments=20
)

Model_fault_layer_03_left_bottom = (
    Model_fault_layer_02_left_bottom
    + layer_03_left_bottom
)

Model_fault_layer_03_left_bottom.addRegionMarker(
    [20, 310],
    marker=7)

fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_layer_03_left_bottom, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()



# -------------------------------------------------------------------
# UPPER LAYER - BETWEEN 85 and 210 m
# -------------------------------------------------------------------

between_layer_01 = mt.createLine(
    start=[85, 335.586],
    end=[210, 335.586],
    nSegments=80
)

Model_fault_between_layer_01 = (
    Model_fault_layer_03_left_bottom
    + between_layer_01
)

Model_fault_between_layer_01.addRegionMarker(
    [120, 330],
    marker=8
)

fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_between_layer_01, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()


# -------------------------------------------------------------------
# LOWER LAYER - BETWEEN 85 and 210 m
# -------------------------------------------------------------------

between_layer_02 = mt.createLine(
    start=[85, 325.655],
    end=[210, 325.655],
    nSegments=80
)

Model_fault_between_layer_02 = (
    Model_fault_between_layer_01
    + between_layer_02
)

Model_fault_between_layer_02.addRegionMarker(
    [150, 320],
    marker=9
)


fig, ax = plt.subplots(figsize=(12, 5),dpi=200)
pg.show(Model_fault_between_layer_02, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(200, 410)
ax.set_xlabel("Distance [m]")
ax.set_ylabel("Elevation [m]")
ax.legend()
plt.show()


# -------------------------------------------------------------------
# BOTTOM LOWER LAYER - BETWEEN 85 and 210 m
# -------------------------------------------------------------------

between_layer_03 = mt.createLine(
    start=[85, 310.759],
    end=[210, 310.759],
    nSegments=80
)

Model_fault_between_layer_03 = (
    Model_fault_between_layer_02
    + between_layer_03
)

Model_fault_between_layer_03.addRegionMarker(
    [190, 305],
    marker=10
)


fig, ax = plt.subplots(figsize=(12, 5),dpi=200)
pg.show(Model_fault_between_layer_03, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(200, 410)
ax.set_xlabel("Distance [m]")
ax.set_ylabel("Elevation [m]")
ax.legend()
plt.show()



# -------------------------------------------------------------------
# UPPER LAYER - RIGHT SIDE of 235
# -------------------------------------------------------------------

layer_01_right = mt.createLine(
    start=[235, 345.517],
    end=[362, 345.517],
    nSegments=60
)

Model_fault_layer_01_right = (
    Model_fault_between_layer_03 + layer_01_right
)

Model_fault_layer_01_right.addRegionMarker(
    [155, 350],
    marker=11
)

fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_layer_01_right, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()


# -------------------------------------------------------------------
# LOWER LAYER - RIGHT SIDE of 235
# -------------------------------------------------------------------

layer_02_right = mt.createLine(
    start=[235, 335.315],
    end=[362, 335.315],
    nSegments=60
)

Model_fault_layer_02_right = (
   Model_fault_layer_01_right + layer_02_right
)

Model_fault_layer_02_right.addRegionMarker(
    [255, 340],
    marker=12
)

fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_layer_02_right, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()


# -------------------------------------------------------------------
# BOTTOM LOWER LAYER - RIGHT SIDE of 235
# -------------------------------------------------------------------

layer_03_right = mt.createLine(
    start=[235, 320.404],
    end=[362, 320.404],
    nSegments=60
)

Model_fault_layer_03_right = (
   Model_fault_layer_02_right + layer_03_right
)

Model_fault_layer_03_right.addRegionMarker(
    [320, 330],
    marker=13
)

fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(Model_fault_layer_03_right, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 600)
ax.set_ylim(100, 410)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend()
plt.show()




# -------------------------------------------------------------------
# VISUALIZATION
# -------------------------------------------------------------------

mesh= mt.createMesh(Model_fault_layer_03_right,   # we can also use pg.viewer.showmesh
                    quality= 33.7
                    #area=40
                    #paraMaxCellSize=4
                    )  
print(mesh)

fig, ax = plt.subplots(figsize=(10, 4.5),dpi=400)
pg.show(mesh, ax=ax,
    showNodes=False)
ax.set_xlim(-600, 958)
ax.set_ylim(200, 410)
ax.set_xlabel("Distance (m)", fontsize=14)
ax.set_ylabel("Depth (m)", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 11.ADDING RESISTIVITY VALUE IN MODEL
# ---------------------------------------------------------------------------

res_map = [
    [1, 140.0],   # World background
    [2, 140.0],   
    [3, 120.0],   
    [4, 20.0],
    [5, 20.0],
    [6, 25.0],
    [7, 140],
    [8, 40],
    [9, 50],
    [10, 140],
    [11, 20],
    [12, 50],
    [13, 50], 
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
        res[cell.id()] = 140.0
    elif cell.marker() == 2:
        res[cell.id()] = 140.0
    elif cell.marker() == 3:
        res[cell.id()] = 120.0
    elif cell.marker() == 4:
        res[cell.id()] = 20.0
    elif cell.marker() == 5:
        res[cell.id()] = 20.0
    elif cell.marker() == 6:
        res[cell.id()] = 25.0
    elif cell.marker() == 7:
        res[cell.id()] = 140.0
    elif cell.marker() == 8:
        res[cell.id()] = 40.0
    elif cell.marker() == 9:
        res[cell.id()] = 50.0
    elif cell.marker() == 10:
        res[cell.id()] = 140.0
    elif cell.marker() == 11:
        res[cell.id()] = 20.0
    elif cell.marker() == 12:
        res[cell.id()] = 50.0
    elif cell.marker() == 13:
        res[cell.id()] = 50.0    
        
        

for cell in mesh.cells():
    x = cell.center().x()
    z = cell.center().y()

    if -600 <= x <= -4 and 352.850 >= z >= 345.517:
        res[cell.id()] = 20.0
        
    if -600 <= x <= -4 and 345.517 >= z >= 335.586:
        res[cell.id()] = 20.0 
    
    if -600 <= x <= -4 and 335.586 >= z >= 320.69:
        res[cell.id()] = 25.0 
    
    if -600 <= x <= -4 and 320.69 >= z >= 300:
        res[cell.id()] = 140.0 
    
    if 362 <= x <= 958 and 355.776 >= z >= 345.517:
        res[cell.id()] = 20.0 
        
    if 362 <= x <= 958 and 345.517 >= z >= 335.586:
        res[cell.id()] = 50.0 
        
    if 362 <= x <= 958 and 335.586 >= z >= 320.69:
        res[cell.id()] = 50.0
        
    if 362 <= x <= 958 and 320.69 >= z >= 300:
        res[cell.id()] = 140.0 
        
fig, ax = plt.subplots(figsize=(12, 5), dpi=400)    
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
ax.set_xlim(-600,958),
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

# RMS difference
diff_percent = (rhoa_fwd - rhoa_meas) / rhoa_meas 
diff= np.sqrt(np.mean(diff_percent**2)) * 100
print (diff)

# RMS difference in percent
rms = np.sqrt(np.mean(diff_percent**2))
print(f"MG forward vs measured RMS difference = {rms:.2f} %")

fig, ax = plt.subplots(figsize=(8,4),dpi=400)

pg.show(data_fwd, 
         vals=data_fwd['rhoa'], 
         ax=ax, 
         cMap='viridis',
         label='Apparent Resistivity (Ωm)')

ax.set_title("Forward Model Response")
plt.show()



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
        if 352.850 >= z >= 345.517:
            start_res[cell.id()] = 20
        elif 345.517 >= z >= 335.586:
            start_res[cell.id()] = 20
        elif 335.586 >= z >= 320.69:
            start_res[cell.id()] = 25
        elif 320.69 >= z >= 300:
            start_res[cell.id()] = 140

    # right block
    elif 362 <= x <= 600:
        if 355.776 >= z >= 345.517:
            start_res[cell.id()] = 20
        elif 345.517 >= z >= 335.586:
            start_res[cell.id()] = 50
        elif 335.586 >= z >= 320.69:
            start_res[cell.id()] = 50
        elif 320.69 >= z >= 300:    
            start_res[cell.id()] = 140
            
            
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



            































 
