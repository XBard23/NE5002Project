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
    S1 = 0
    S2 = 0
    E1 = 0
    E2 = 0
    
    
    n = int((end + 2*D2)//dxy)
    x1 = int(m1End//dxy)
    x2 = n - x1
    print(n, x1, x2)

    matrixA = np.zeros((n*n//2, n*n//2))
    matrixAbsorption = np.zeros((1,n))
    
    #Create variables, refer to README.txt for clarification of numbering
    dEqs = {
        '0': descretizedEqs(0, 0, 0, D1, 0, dxy, 0, dxy, S1, S1, E1, E1), #Corner
        '1': descretizedEqs(0, D1, 0, D1, 0, dxy, dxy, dxy, S1, S2, E1, E2), #M1 LR
        '2': descretizedEqs(D1, D1, D1, D1, dxy, dxy, dxy, dxy, S1, S2, E1, E2), #M1 FS
        '3': descretizedEqs(D1, D1, D1, D1, dxy, dxy, dxy, dxy, S1, S2, E1, E2), # M1 RR
        '4': descretizedEqs(0, D1, 0, D2, dxy, dxy, dxy, dxy, S1, S2, E1, E2), # Int LR
        '5': descretizedEqs(D1, D1, D2, D2, dxy, dxy, dxy, dxy, S1, S2, E1, E2), # Int FS
        '6': descretizedEqs(D1, D1, D2, D2, dxy, dxy, dxy, dxy, S1, S2, E1, E2), # Int RR
        '7': descretizedEqs(0, D2, 0, D2, dxy, dxy, dxy, dxy, S1, S2, E1, E2), # M2 LR
        '8': descretizedEqs(D2, D2, D2, D2, dxy, dxy, dxy, dxy, S1, S2, E1, E2), # M2 Fs
        '9': descretizedEqs(D2, D2, D2, D2, dxy, dxy, dxy, dxy, S1, S2, E1, E2), # M2 RR
        }
    
    #adjustments: 
        #3, 6(Diagnol needs to set Right and bottom to zero)
        #3, 6( need 1/2 1 and 3 and v2 = 0)
        
    
    #Build m2End
    matrix2 = buildLine(dEqs['7'], dEqs['8'], dEqs['9'], n, n)
    matrixA = np.array(matrix2)
    matrixA = iterate(matrixA, matrix2, x2-2, n)
    #Build interface
    matrixI = buildLine(dEqs['4'], dEqs['5'], dEqs['6'], n, x1)


    matrixA = np.vstack([matrixI, matrixA])
    #build m1End
    matrix1 = buildLine(dEqs['1'], dEqs['2'], dEqs['3'], n, x1)
    matrixA = np.vstack([matrix1, matrixA])
    matrixA = iterate(matrixA, matrix1, x1-2, 0)

    
    #Assign first row
    matrix1b = buildLine(dEqs['1'], dEqs['2'], dEqs['3'], n, 2)
    
    #matrixA = np.vstack([matrix1b, matrixA])
    
    #assign corner
    matrixC = [0]*(n*n)
    matrixC[0] = dEqs['0'][4]; matrixC[n] = dEqs['0'][3]
    #matrixA = np.vstack([matrixC, matrixA])

    

    

     
    return matrixA

#Descretized Equations, These are called when we know where we are and calculate the proper equations
def descretizedEqs(D00, D10, D01, D11, X0, X1, Y0, Y1, S1, S2, E1, E2):
    result = [0 if X0 == 0 else -(D00*Y0+D01*Y1)/(2*X0), #Left
              0 if X1 == 0 else -(D10*Y0+D11*Y1)/(2*X1), #Right
              0 if Y0 == 0 else -(D00*X0+D10*X1)/(2*Y0), #Bottom
              0 if Y1 == 0 else -(D01*X0+D11*X1)/(2*Y1), #Top
              ]
    
    V = [0.25 * x * y for x in [X0, X1] for y in [Y0, Y1]]
    E = V[0]*E1 + V[1]*E2 + V[2]*E1 + V[3]*E2
    S = V[0]*S1 + V[1]*S2 + V[2]*S1 + V[3]*S2
    aC = E - sum(result)
    result.extend([aC, E, S])
    
    return result

#Build Matrix
def buildLine(t, m, b, n, x):
    
    y = 2*n+1
    n2 = n*(n+1)
    start = n2-y+1
   #Build individual matrices:
    top = [0]*(y)
    top[0] = t[2]; top[n] = t[4]; top[n+1] = t[1]; top[-1] = t[3]
    mid = [0]*(y)
    mid[0] = m[2]; mid[n-1] = m[0]; mid[n] = m[4]; mid[n+1] = m[1]; mid[-1] = t[3]
    bot = [0]*(y)
    bot[n-1] = b[0]; bot[n] = b[4]; bot[-1] = t[3]
    
   #Build full matrix:
    matrix = np.zeros((x, n2))
    matrix[0,start-x:start-x+y] = top
    matrix[-1, start-1:start+y-1] = bot
    
    midCenter = y // 2
    
    for i in range(1, x-1):
        begin = start + i-x
        end = begin + y
        matrix[i, max(begin, 0):min(end, n2)] = mid[max(-begin,0):y-max(end-n2,0)]
    
    return matrix


def iterate(matrixA, matrix, stop, n):
    
    matrix = np.delete(matrix, -2, axis=0)
    last = np.roll(matrix[-1], -1)
    matrix[-1] = last
    matrix = np.hstack([matrix[:, n:], np.zeros((matrix.shape[0], n))])
    
    matrixA = np.vstack([matrix, matrixA])
    
    if (stop >= 1):
        matrixA = iterate(matrixA, matrix, stop-1, n)
 
    return matrixA
        

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
matrix = createMatrix(materials, dxy)
#Get output
#print(reflectiveEdge(materials, 0.5, 0))
