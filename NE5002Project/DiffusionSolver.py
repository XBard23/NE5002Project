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
Input data 

Dictionary for each material
"""
#Read input data (calls check here)
df = pd.read_excel("Data/input.xlsx")

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
            'M': df.at[4, col],
            'density': df.at[5, col]
        }
        material_count += 1
        
#Data optimization
del col
del df
del material_count
del material_key

#Check input data



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

#Create matrix (Call BC here, then solver)

#Boundary conditions

#Solver



"""
Main

"""

#Get input

#Create matrix

#Get output
