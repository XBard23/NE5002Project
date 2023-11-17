#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:17:54 2023

@author: Xander Bard 
"""


"""
Libraries needed:
Numpy: 
    
Matplotlib:

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
Timing of methods using decorator functions
"""



"""
Input data 

Dictionary for each material
"""
def extractInput(fileInput):
    #Read input data (calls check here)
    df = pd.read_excel(fileInput)
    
    # Dictionary to store each material's properties
    materials = {}
    material_count = 1  # To keep track of material number
    
    # Loop over columns in the DataFrame to get material data
    for col in df.columns[1:]:
        # Check if there's valid data in the 'x-start' row of this column (assuming 'x-start' is always present for valid materials)
        if not pd.isna(df.at[1, col]):
            material_key = f"m{material_count}"  # Generate key like 'm1', 'm2', etc.
            materials[material_key] = {
                'x-end': df.at[1, col],
                'absorption': df.at[2, col],
                'scattering': df.at[3, col],
            }
            material_count += 1
            
    #Data optimization
    del col
    del df
    del material_count
    del material_key
    
    return materials


#Check input data

#Create mesh and show user
#Mesh size will be generated based on the neutron diffusion length
def mesh(materials):
    meshPoints = {}
    xStart = 0
    
    
    for key, mat in materials.items():
        L, D = diffusionLength(mat['absorption'], mat['scattering'])
        xEnd = mat['x-end']
        length = xEnd - xStart
        void = 2*D
        mat['L'] = L
        mat['D'] = D
        # Initialize dxy as half the diffusion length
        dxy = L / 2

        # Check if dxy is too large for the material's length
        if dxy > length:
            dxy = length

        # Adjust dxy to be the largest divisor of total_length less than or equal to L
        while length % dxy > 0.1 * L:
            dxy -= 0.1 * L  # Decrement dxy by 0.1*L until it's a suitable divisor

        mat['dxy'] = dxy

        # Extend the mesh for the void boundary only if it's the second material ('m2')
        meshPoints[key] = np.arange(xStart, xEnd, dxy)
        if key == 'm2':
            meshPoints[key] = np.append(meshPoints[key], xEnd + void)
        
        xStart = meshPoints[key][-1]
        

        #print("dxy = ", dxy)
        #print("L = ", L)

    return meshPoints



def plot_mesh_and_materials(meshPoints, materials):
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Loop through each material's mesh points
    previous_xEnd = 0
    for key, x_points in meshPoints.items():
        xEnd = materials[key]['x-end']
        
        # Plot the boundary for the material
        if key == 'm1':
            ax.plot([0, xEnd, xEnd, 0, 0], [0, 0, xEnd, xEnd, 0], color='black', linewidth=2)
        else:
            ax.plot([previous_xEnd, xEnd, xEnd, previous_xEnd], [0, 0, xEnd, xEnd], color='black', linewidth=2)
        
        # Overlay the mesh points as a lighter grid within the material
        for x in x_points:
            if x > previous_xEnd and x < xEnd:  # Ensure mesh is inside the material
                ax.axvline(x, color='lightblue', linestyle='--', linewidth=0.5)
                ax.axhline(x, color='lightblue', linestyle='--', linewidth=0.5)
        
        previous_xEnd = xEnd
    
    ax.set_xlim(0, xEnd)
    ax.set_ylim(0, xEnd)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Mesh and Materials Visualization')
    plt.show()

#A vectorized approach is used for maximal efficiency
def matrixCoord(meshPoints):
    # Combine and remove duplicates from the mesh points
    all_points = np.array([])
    for key in meshPoints:
        all_points = np.concatenate((all_points, meshPoints[key]))
    all_points = np.unique(all_points)

    # Use broadcasting to create a grid of coordinate pairs
    X, Y = np.meshgrid(all_points, all_points)
    coordinate_matrix = np.dstack([X, Y])   

    return coordinate_matrix




"""
Version data

"""
#Create output file

#Metadata information to output file (Call input data)

#Data visualizer

#Output


"""
Input echo
 
"""
#Print input to output


"""
Diffusion solver

"""

#Create matrix (then solver)
def createMatrix(materials, meshMatrix):
    meshPoints = mesh(materials)
    n = len(meshPoints['m1'])
    n += len(meshPoints['m2']) - 1
    #print(n)
    matInterface = materials['m1']['x-end']
    end = materials['m2']['x-end']
    
    matrixA = np.zeros((n, n, n, n))
    
    for i in range(n):
        matrixL = np.zeros((n, n))
        matrixC = np.zeros((n, n))
        matrixR = np.zeros((n, n))
        for j in range(n):
            #Get coordinates for this iteration
            x, y = meshMatrix[i, j, 0], meshMatrix[i, j, 1]
            #print("x:", x, "y:", y)
            
            
            #call equation components
            
        
    return

#Discretized equations
def reflectiveEdge(materials, x, y):
    matrixL = 0
    matrixC = np.zeros((1, 3))
    matrixR = 0
    m1End = materials['m1']['x-end']
    m2End = materials['m2']['x-end']
    D1 = materials['m1']['D']
    D2 = materials['m2']['D']
    a1 = materials['m1']['absorption']
    a2 = materials['m2']['absorption']
    m1dxy = materials['m1']['dxy']
    m2dxy = materials['m2']['dxy']
    aR = 0
    aL = 0
    aB = 0
    aT = 0
    aC = 0

    #Corner of reflective boundary in material 1
    if x == y: #can do this since we know one needs to be zero to call this function
        #calculate vairables
        aR = -D1/2
        aT = aR
        aC = a1 - (aR + aT)
     
    #matrix interface
    elif x < m1End and x + m1dxy > m1End or y < m1End and y + m1dxy > m1End:
        matrixL, matrixC, matrixL = reflectiveInterface(materials, x, D1, D2)
    
    #Void and reflective interface corner
    elif x < m2End and x + m2dxy > m1End or y < m2End and y + m2dxy > m2End:
        matrixL, matrixC, matrixR = dualBC(materials['m2']['dxy'], D2, a2, x)
        
    else:
        #Left boundary
        if x == 0:
            if y < m1End:
                D = D1
            else:
                D = D2
            
            aR = -D/2
            aL = aR
            aT = -D
            aB = 0
            aC = a1 - (aR + aL + aT + aB)
        
        #Bottom boundary
        else:
            if x < m1End:
                D = D1
            else:
                D = D2
            aR = -D
            aL = 0
            aT = -D/2
            aB = aT
            aC = a2 - (aR + aL + aT + aB)
            
    matrixL = aB
    matrixC = [aL, aC, aR]
    matrixR = aT
    
    return matrixL, matrixC, matrixR

def reflectiveInterface(materials, x, D1, D2):
    
    matrixL = 0
    matrixC = np.zeros((1, 3))
    matrixR = 0
    
    m1dxy = materials['m1']['dxy']
    m2dxy = materials['m2']['dxy']
    
    #Left Edge
    if x == 0:
        aR = (D1*m1dxy + D2*m2dxy) / ( 2*m1dxy)
        aB = -D1/2
        aT = -D2*m1dxy / m2dxy
        aL = 0
        aC = a2 - (aR + aL + aT + aB)
    
    #Botoom Edge
    else:
        pass
    
        
    matrixL = aB
    matrixC = [aL, aC, aR]
    matrixR = aT
    
    return matrixL, matrixC, matrixR
    

def voidEdge():
    
    pass


def dualBC(dxy, D2, a2, x):
    matrixL = 0
    matrixC = np.zeros((1, 3))
    matrixR = 0
    
    #Top left corner
    if x == 0:   
        aR = -D2
        aL = 0
        aT = -dxy/2
        aB = aT
        aC = a2 - (aR + aL + aT + aB)
    
    #Bottom right corner
    else:
        aR = -dxy/2
        aL = aR
        aT = -D2
        aB = 0
        aC = a2 - (aR + aL + aT + aB)
    
    matrixL = aB
    matrixC = [aL, aC, aR]
    matrixR = aT
    
    return matrixL, matrixC, matrixR
    pass

def materialEdge():
    
    pass

def freeSpace(D):
    
    pass


#Build Matrix

#Solver

"""
Addiitonal math calls for code readability
"""

def diffusionLength(ab, sc):
    
    transport = ab + sc
    D = 1 / (3 * transport)
    L = np.sqrt(D/ab)
    
    return L, D


"""
Main

"""

#Get input
materials = extractInput("Data/input.xlsx")
#Create matrix
meshPoints = mesh(materials)
plot_mesh_and_materials(meshPoints, materials)
meshMatrix = matrixCoord(meshPoints)
createMatrix(materials, meshMatrix)
#Get output
print(reflectiveEdge(materials, 0.5, 0))
