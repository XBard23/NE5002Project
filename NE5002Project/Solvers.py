#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:45:52 2023

@author: Xander Bard
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def jacobi(A, b, xk, iterations):
    print("Method: Jacobi")
    n = A.shape[0]
    
    for k in range(iterations):
        xNew = np.zeros(n)
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], xk[-1][:i])
            sum2 = np.dot(A[i, i+1:], xk[-1][i+1:])
            xNew[i] = (b[i] - sum1 - sum2) / A[i, i]
            
        
        xk.append(xNew)
        print(f"Iteration {k+1}: {xNew}")
    print(f"Error x1: {(xk[-1][0] - 3)}")
    print(f"Error x2: {(xk[-1][1] - 4)}")
    print(f"Error x3: {(xk[-1][2] + 5)}")
    
    return xk

# Initialize variables
A = np.array([[4, 3, 0], [3, 4, -1], [0, -1, 4]])
b = np.array([24, 30, -24])
xz = []
xz.append(np.array([1, 1, 1]))

# Run Jacobi method
result = jacobi(A, b, xz, 4)

def gaussSiedel(A, b, xk, iterations):
    print("Method: Gauess Siedel")
    n = A.shape[0]
    
    for k in range(iterations):
        xNew = np.zeros(n)
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], xNew[:i])
            sum2 = np.dot(A[i, i+1:], xk[-1][i+1:])
            xNew[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        xk.append(xNew)
        print(f"Iteration {k+1}: {xNew}")
    print(f"Error x1: {xk[-1][0] -3 }")
    print(f"Error x2: {xk[-1][1] -4} ")
    print(f"Error x3: {xk[-1][2] +5 }")
    
    return xk

# Initialize variables
A = np.array([[4, 3, 0], [3, 4, -1], [0, -1, 4]])
b = np.array([24, 30, -24])
xz = []
xz.append(np.array([1, 1, 1]))

# Run GS method
result = gaussSiedel(A, b, xz, 4)

def SOR(A, b, xk, iterations, w):
    print("Method: SOR")
    n = A.shape[0]
    
    for k in range(iterations):
        xNew = np.zeros(n)
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], xNew[:i])
            sum2 = np.dot(A[i, i+1:], xk[-1][i+1:])
            xNew[i] = (1-w)*xk[-1][i] + w*(b[i] - sum1 - sum2) / A[i, i]
        
        xk.append(xNew)
        print(f"Iteration {k+1}: {xNew}")
    print(f"Error x1: {(xk[-1][0] - 3)}")
    print(f"Error x2: {(xk[-1][1] - 4)}")
    print(f"Error x3: {(xk[-1][2] + 5) }")
    
    return xk

# Initialize variables
A = np.array([[4, 3, 0], [3, 4, -1], [0, -1, 4]])
b = np.array([24, 30, -24])
xz = []
xz.append(np.array([1, 1, 1]))
w = 5/4

# Run SOR method
result = SOR(A, b, xz, 4, w)

#For fun: Heuristic approach: Particle Swarm Optimization (PSO)
def residual(x, A, b):
    return np.linalg.norm(np.dot(A, x) - b)

def PSO(A, b, n_particles=30, n_iterations=500, w=0.5, c1=1.5, c2=1.5):
    n_variables = A.shape[1]
    print("Method: Particle Swarm")
    
    # Initialize particle positions and velocities
    positions = np.random.uniform(-10, 10, (n_particles, n_variables))
    velocities = np.random.uniform(-1, 1, (n_particles, n_variables))
    personal_best_positions = np.copy(positions)
    global_best_position = np.copy(positions[0])
    
    # Evaluate initial personal and global bests
    personal_best_scores = np.apply_along_axis(residual, 1, personal_best_positions, A, b)
    global_best_score = residual(global_best_position, A, b)
    
    for i in range(n_iterations):
        for j in range(n_particles):
            # Update velocity
            inertia = w * velocities[j]
            cognitive = c1 * np.random.random() * (personal_best_positions[j] - positions[j])
            social = c2 * np.random.random() * (global_best_position - positions[j])
            velocities[j] = inertia + cognitive + social
            
            # Update position
            positions[j] += velocities[j]
            
            # Update personal best
            score = residual(positions[j], A, b)
            if score < personal_best_scores[j]:
                personal_best_scores[j] = score
                personal_best_positions[j] = positions[j]
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[j]
                    
        #print(f"Iteration {i+1}: Global best score = {global_best_score}, Global best position = {global_best_position}")
    
    return global_best_position, global_best_score

A = np.array([[4, 3, 0], [3, 4, -1], [0, -1, 4]])
b = np.array([24, 30, -24])

# Run PSO
optimal_position, optimal_score = PSO(A, b)
print(f"Optimal position: {optimal_position}, Optimal score: {optimal_score}")
