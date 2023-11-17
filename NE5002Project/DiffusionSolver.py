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
        
        # Initialize dxy as half the diffusion length
        dxy = L / 2

        # Check if dxy is too large for the material's length
        if dxy > length:
            dxy = length

        # Adjust dxy to be the largest divisor of total_length less than or equal to L
        while length % dxy > 0.1 * L:
            dxy -= 0.1 * L  # Decrement dxy by 0.1*L until it's a suitable divisor

        # Extend the mesh for the void boundary only if it's the second material ('m2')
        meshPoints[key] = np.arange(xStart, xEnd, dxy)
        if key == 'm2':
            meshPoints[key] = np.append(meshPoints[key], xEnd + void)
        
        xStart = meshPoints[key][-1]
        

        print("dxy = ", dxy)
        print("L = ", L)

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
def createMatrix(materials):
    meshPoints = mesh(materials)
    n = len(meshPoints['m1'])
    n += len(meshPoints['m2']) - 1
    print(n)
    
    meshMatrix = matrixCoord(meshPoints)
    
    matrixA = np.zeros((n, n, n, n))
    
    for i in range(n):

        for j in range(n):
            #Get coordinates for this iteration
            x, y = meshMatrix[i, j, 0], meshMatrix[i, j, 1]
            print("x:", x, "y:", y)
        
            
        
    return

#Discretized equation components

def reflectiveEdge():
    pass

def voidEdge():
    pass

def dualBC():
    pass

def materialEdge():
    pass

def freeSpace():
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
meshPoints = mesh(materials)
print(meshPoints)
plot_mesh_and_materials(meshPoints, materials)
createMatrix(materials)
#Create matrix

#Get output
