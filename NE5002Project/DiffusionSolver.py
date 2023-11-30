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
import math


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
    xStart = 0
    meshSize = 0
     
    for key, mat in materials.items():
        L, D = diffusionLength(mat['absorption'], mat['scattering'])
        xEnd = mat['x-end']
        length = xEnd - xStart
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
        
        if meshSize == 0 or meshSize > dxy:
            meshSize = dxy
        
        xStart = xEnd

    return meshSize


#Take out of final code
def plot_mesh_and_materials(meshPoints, materials):
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define variables for the total x extent
    total_xEnd = max(material['x-end'] for material in materials.values())

    # Plot the boundaries and mesh for each material
    previous_xEnd = 0
    for key, material in materials.items():
        xEnd = material['x-end']

        # Plot the boundary for the material
        ax.plot([previous_xEnd, xEnd, xEnd, previous_xEnd, previous_xEnd], [0, 0, xEnd, xEnd, 0], color='black', linewidth=2)
        
        previous_xEnd = xEnd

    # Overlay the mesh points as a lighter grid within the entire area
    x = meshPoints
    while x <= total_xEnd:
        ax.axvline(x, color='lightblue', linestyle='--', linewidth=0.5)
        ax.axhline(x, color='lightblue', linestyle='--', linewidth=0.5)
        x += meshPoints

    ax.set_xlim(0, total_xEnd)
    ax.set_ylim(0, total_xEnd)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Mesh and Materials Visualization')
    plt.show()




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
def createMatrix(materials, dxy):
    meshPoints = mesh(materials)
    
    #print(n)
    matInterface = materials['m1']['x-end']
    end = materials['m2']['x-end']
    

    


    m1End = materials['m1']['x-end']
    m2End = materials['m2']['x-end']
    D1 = materials['m1']['D']
    D2 = materials['m2']['D']
    a1 = materials['m1']['absorption']
    a2 = materials['m2']['absorption']
    
    
    nCalc = (end + 2*D2)/dxy
    n = math.floor(nCalc)

    matrixA = np.zeros((n*n, n*n))
    matrixAbsorption = np.zeros((1,n))
    
    #Create variables, refer to README.txt for clarification of numbering
    dEqs = {
        '0': descretizedEqs(0, 0, 0, D1, 0, dxy, 0, dxy),
        '1': descretizedEqs(0, D1, 0, D1, 0, dxy, dxy, dxy),
        '2': descretizedEqs(D1, D1, D1, D1, dxy, dxy, dxy, dxy),
        '3': descretizedEqs(D1, D1, D1, D1, dxy, dxy, dxy, dxy),
        '4': descretizedEqs(D1, D2, D1, D2, dxy, dxy, dxy, dxy),
        '5': descretizedEqs(D1, D2, D1, D2, dxy, dxy, dxy, dxy),
        '6': descretizedEqs(D1, D2, D1, D2, dxy, dxy, dxy, dxy),
        '7': descretizedEqs(D1, D2, D1, D2, dxy, dxy, dxy, dxy),
        '8': descretizedEqs(D1, D2, D1, D2, dxy, dxy, dxy, dxy),
        '9': descretizedEqs(D1, D2, D1, D2, dxy, dxy, dxy, dxy),
        }
    #adjustments: 
        #3(Diagnol needs to set Right and bottom to zero)
        #3( need 1/2 1 and 3 and v2 = 0)
    y = dxy
    
    #assign corner
    
    #calculate the rest
    while y < m2End + 2*D2:
        #assign x = 0:
            
        
        #assign x = y: 
            
        
        #assign free space: (if statement)
        
            
        y += dxy
    
    
    
    
    #for i, row in enumerate(meshMatrix):
        
        #for j, (x, y) in enumerate(row):
     
    return

#Descretized Equations, These are called when we know where we are and calculate the proper equations
def descretizedEqs(D00, D10, D01, D11, X0, X1, Y0, Y1):
    result = [0 if X0 == 0 else -(D00*Y0+D01*Y1)/(2*X0), #Left
              0 if X1 == 0 else -(D10*Y0+D11*Y1)/(2*X1), #Right
              0 if Y0 == 0 else -(D00*X0+D10*X1)/(2*Y0), #Bottom
              0 if Y1 == 0 else -(D01*X0+D11*X1)/(2*Y1), #Top
              ]
    result.extend([0.25 * x * y for x in [X0, X1] for y in [Y0, Y1]])
    
    return result

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
dxy = mesh(materials)
plot_mesh_and_materials(dxy, materials)
createMatrix(materials, dxy)
#Get output
#print(reflectiveEdge(materials, 0.5, 0))
