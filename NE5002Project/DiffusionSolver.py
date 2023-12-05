#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:17:54 2023

@author: Xander Bard 
"""

"""
Libraries:
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
            dxy = length//2

        # Adjust dxy to be the largest divisor of total_length less than or equal to L
        while length % dxy > 0.4 * L:
            dxy -= 0.1 * L  # Decrement dxy by 0.1*L until it's a suitable divisor
        
        if meshSize == 0 or meshSize > dxy:
            meshSize = dxy
        
        xStart = xEnd

    return meshSize


def plot_mesh_and_materials(meshPoints, materials):
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define variables for the total x extent
    total_xEnd = max(material['x-end'] for material in materials.values())

    # Plot the boundaries and mesh for each material in the first quadrant
    previous_xEnd = 0
    for key, material in materials.items():
        xEnd = material['x-end']

        # Vertical line
        ax.plot([xEnd, xEnd], [0, xEnd], color='black', linewidth=2)  # Vertical line

        # Horizontal line for material 1
        if key == 'm1':
            ax.plot([previous_xEnd, xEnd], [xEnd, xEnd], color='black', linewidth=2)

        previous_xEnd = xEnd

    # Extend the material 1 boundary to form a larger square
    m1_xEnd = materials['m1']['x-end']
    ax.plot([-m1_xEnd, m1_xEnd], [m1_xEnd, m1_xEnd], color='black', linewidth=2)  # Top horizontal line
    ax.plot([-m1_xEnd, -m1_xEnd], [-m1_xEnd, m1_xEnd], color='black', linewidth=2)  # Left vertical line
    ax.plot([-m1_xEnd, m1_xEnd], [-m1_xEnd, -m1_xEnd], color='black', linewidth=2)  # Bottom horizontal line
    ax.plot([m1_xEnd, m1_xEnd], [-m1_xEnd, m1_xEnd], color='black', linewidth=2)  # Right vertical line

    # Overlay the mesh points as a lighter grid within the entire area
    for x in np.arange(-total_xEnd, total_xEnd + meshPoints, meshPoints):
        ax.axvline(x, color='lightblue', linestyle='--', linewidth=0.5)
        ax.axhline(x, color='lightblue', linestyle='--', linewidth=0.5)

    ax.set_xlim(-total_xEnd, total_xEnd)
    ax.set_ylim(-total_xEnd, total_xEnd)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Mesh and Materials Visualization')
    plt.show()




def plotResults(matrixA, flux, n, dxy, end):
    matrix = np.zeros((n, n))
    index = 0
    for i in range(n):
        for j in range(0,i+1):
            matrix[i, j] = flux[index]
            index += 1   
    for i in range(1, n):
        for j in range(i):
            matrix[j, i] = matrix[i, j]
    x, y = np.meshgrid(np.linspace(0, end, n), np.linspace(0, end, n))

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(x, y, matrix, cmap='viridis')
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.show()
    
    #X plot
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x[2, :], matrix[2, :], color='blue')
    ax2.set_title("Flux along x-axis at y = 0")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Flux")
    plt.show()
    
    #y=x diagonal
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    diagonal_values = np.array([matrix[i, i] for i in range(n)])
    ax3.plot(x[0, :], diagonal_values, color='green')
    ax3.set_title("Flux along y = x")
    ax3.set_xlabel("y = x")
    ax3.set_ylabel("Flux")
    plt.show()

    return

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
    m1End = materials['m1']['x-end']
    m2End = materials['m2']['x-end']
    D1 = materials['m1']['D']
    D2 = materials['m2']['D']
    a1 = materials['m1']['absorption']
    a2 = materials['m2']['absorption']
    S1 = 5
    S2 = 5
    
    n = int((m2End + 2*D2)//dxy)
    x1 = int(m1End//dxy)
    x2 = n - x1

    matrixA = np.zeros((n*n//2, n*n//2))
    sourceVector = []
    sv = []
    
    #Create variables, refer to README.txt for clarification of numbering
    dEqs = {
        '0': descretizedEqs(0, 0, 0, D1, 0, dxy, 0, dxy, 0, S1, 0, a1), #Corner
        '1': descretizedEqs(0, D1, 0, D1, 0, dxy, dxy, dxy, S1, S1, a1, a1), #M1 LR
        '2': descretizedEqs(D1, D1, D1, D1, dxy, dxy, dxy, dxy, S1, S1, a1, a1), #M1 FS
        '3': descretizedEqs(D1, 0, D1, D1, dxy, dxy, dxy, dxy, S1, S1, a1, a1), # M1 RR
        '4': descretizedEqs(0, D1, 0, D2, 0, dxy, dxy, dxy, S1, S2, a1, a2), # Int LR
        '5': descretizedEqs(D1, D1, D2, D2, dxy, dxy, dxy, dxy, S1, S2, a1, a2), # Int FS
        '6': descretizedEqs(D1, 0, D2, D2, dxy, dxy, dxy, dxy, S1, S2, a1, a2), # Int RR
        '7': descretizedEqs(0, D2, 0, D2, 0, dxy, dxy, dxy, S2, S2, a2, a2), # M2 LR
        '8': descretizedEqs(D2, D2, D2, D2, dxy, dxy, dxy, dxy, S2, S2, a2, a2), # M2 Fs
        '9': descretizedEqs(D2, 0, D2, D2, dxy, dxy, dxy, dxy, S2, S2, a2, a2), # M2 RR
        }
    
    #adjustments: 
        #3, 6, 9( need 1/2 1 and 3 and v2 = 0)
        
    #Build m2End
    matrix2, sv = buildLine2(dEqs['7'], dEqs['8'], dEqs['9'], n, n, 0)
    sourceVector = np.append(sourceVector, sv)
    matrixA = np.array(matrix2)
    matrixA, sourceVector = iterate(matrixA, matrix2, sourceVector, sv, x2-2, n)
    #Build interface
    matrixI, sv = buildLine2(dEqs['4'], dEqs['5'], dEqs['6'], n, x1+1, (x2-1)*n+x2-1)
    matrixA = np.vstack([matrixI, matrixA])
    sourceVector = np.append(sv, sourceVector)
    #build m1End
    if x1 > 1:
        matrix1, sv = buildLine2(dEqs['1'], dEqs['2'], dEqs['3'], n, x1, x2*(n+1))
        matrixA = np.vstack([matrix1, matrixA])
        matrixA, sourceVector = iterate(matrixA, matrix1, sourceVector, sv, x1-2, n)
        sourceVector = np.append(sv, sourceVector)
    
    #assign corner
    matrixC = [0]*(n*(n+1))
    matrixC[0] = dEqs['0'][4]; matrixC[n] = dEqs['0'][3]
    matrixA = np.vstack([matrixC, matrixA])
    sourceVector = np.append(dEqs['0'][6], sourceVector)

    #Condense matrix to square
    nonZero = ~np.all(matrixA == 0, axis = 0) 
    matrixA = matrixA[:, nonZero]
    matrixA = matrixA[:, :-n] 

    return matrixA, sourceVector, n

#Descretized Equations
def descretizedEqs(D00, D10, D01, D11, X0, X1, Y0, Y1, S1, S2, E1, E2):
    result = [0 if X0 == 0 else -(D00*Y0+D01*Y1)/(2*X0), #Left
              0 if X1 == 0 else -(D10*Y0+D11*Y1)/(2*X1), #Right
              0 if Y0 == 0 else -(D00*X0+D10*X1)/(2*Y0), #Bottom
              0 if Y1 == 0 else -(D01*X0+D11*X1)/(2*Y1), #Top
              ]
        
    V = [0.25 * x * y for x in [X0, X1] for y in [Y0, Y1]]
    aC = 0
    
    #Right Reflective adjustment
    if D10 == 0 and D00 > 0:
        V[2] = 0
        V[0] = 0.5 * V[0]
        V[3] = 0.5 * V[3] 
        result[1] = 0; result[2] = 0
    
    #Corner adjustment
    if D10 == 0 and D00 == 0:
        V[3] = 0.5*V[3]
        result[1] = 0
        
    #Calculate aC and source
    E = V[0]*E1 + V[1]*E2 + V[2]*E1 + V[3]*E2
    S = V[0]*S1 + V[1]*S2 + V[2]*S1 + V[3]*S2
    aC += E - sum(result)
    result.extend([aC, E, S])
    
    return result


#Build Matrix
def buildLine2(t, m, b, n, x, adj):
    
    y = 2*n+1
    n2 = n*(n+1)
    last = n2-adj
    first = last - y
   #Build individual matrices:
    top = [0]*(y)
    top[0] = t[2]; top[n] = t[4]; top[n+1] = t[1]; top[-1] = t[3]
    mid = [0]*(y)
    mid[0] = m[2]; mid[n-1] = m[0]; mid[n] = m[4]; mid[n+1] = m[1]; mid[-1] = m[3]
    bot = [0]*(y)
    bot[n-1] = b[0]; bot[n] = b[4]; bot[-1] = b[3]
    
   #Build source vector:
    sourceVector = []
    
   #Build full matrix:
    matrix = np.zeros((x, n2))
    matrix[0,first-x+1:last-x+1] = top
    matrix[-1, first:last] = bot
    
    sourceVector = np.append(t[6], sourceVector)
    
    for i in range(1, x-1):
        begin = first + i - x + 1
        end = begin + y
        matrix[i, max(begin, 0):min(end, n2)] = mid
        sourceVector = np.append(m[6], sourceVector)
        
    sourceVector = np.append(b[6], sourceVector)
    
    return matrix, sourceVector


def iterate(matrixA, matrix, sourceVector, sv, stop, n):
    while stop >= 1:
        # Main Matrix adjustments
        matrix = np.delete(matrix, -2, axis=0)
        last = np.roll(matrix[-1], -1)
        matrix[-1] = last
        matrix = np.hstack([matrix[:, n:], np.zeros((matrix.shape[0], n))])

        matrixA = np.vstack([matrix, matrixA])

        # Source Vector adjustments
        sv = np.delete(sv, 1)
        sourceVector = np.append(sv, sourceVector)

        stop -= 1
 
    return matrixA, sourceVector
        


"""
Solver
"""
def SOR(A, b, xk, w, tol):
    n = A.shape[0]
    xNew = np.zeros(n)
    
    while True:
        xOld = xNew.copy()
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], xNew[:i])
            sum2 = np.dot(A[i, i+1:], xOld[i+1:])
            xNew[i] = (1-w)*xOld[i] + w*(b[i] - sum1 - sum2) / A[i, i]
        
        xk.append(xNew.copy())

        # Calculate relative error
        relError = np.linalg.norm(xNew - xOld) / np.linalg.norm(xNew)
        print(relError)
        if relError < tol:
            break
    
    return xk[-1]


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
#plot_mesh_and_materials(dxy, materials)
matrixA, source, n = createMatrix(materials, dxy)

#Get output

flux = np.zeros(matrixA.shape[1])
guess = [flux]

flux = SOR(matrixA, source, guess, 5/4, 1e-4)

#Plot results
plotResults(matrixA, flux, n, dxy, materials['m2']['x-end'])

