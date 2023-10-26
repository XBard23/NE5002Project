#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:42:05 2023

@author: xanderbard
"""

import tkinter as tk
from tkinter import ttk


def get_values():
    material_1_type = material1_type_entry.get()
    material_1_scattering = material1_scattering_entry.get()
    material_1_absorption = material1_absorption_entry.get()
    material_1_begin = material1_begin_entry.get()
    material_1_end = material1_end_entry.get()

    material_2_type = material2_type_entry.get()
    material_2_scattering = material2_scattering_entry.get()
    material_2_absorption = material2_absorption_entry.get()
    material_2_begin = material2_begin_entry.get()
    material_2_end = material2_end_entry.get()

    canvas.delete("all")
    
    size = max(float(material_1_end), float(material_2_end))
    cell_size = 400 / size

    for i in range(int(size)):
        for j in range(int(size)):
            canvas.create_rectangle(i * cell_size, j * cell_size, (i+1) * cell_size, (j+1) * cell_size)

    
    source_data = []
    for i, source in enumerate(sources):
        x = source[0].get()
        y = source[1].get()
        value = source[2].get()
        source_data.append((x, y, value))
        
    print("Material 1:", material_1_type, material_1_scattering, material_1_absorption, material_1_begin, material_1_end)
    print("Material 2:", material_2_type, material_2_scattering, material_2_absorption, material_2_begin, material_2_end)
    print("Sources:", source_data)
    

def add_source():
    x_entry = tk.Entry(root)
    y_entry = tk.Entry(root)
    value_entry = tk.Entry(root)
    
    x_entry.grid(row=len(sources) + 6, column=1)
    y_entry.grid(row=len(sources) + 6, column=3)
    value_entry.grid(row=len(sources) + 6, column=5)
    
    sources.append((x_entry, y_entry, value_entry))


# Initialize the Tkinter window
root = tk.Tk()
root.title("Material and Source Properties")

# Create grid
canvas = tk.Canvas(root, bg='white', width=400, height=400)
canvas.grid(row=6, column=10, rowspan=10, columnspan=10)


# Material 1
material1_type_entry = tk.Entry(root)
material1_type_entry.grid(row=1, column=1)
material1_scattering_entry = tk.Entry(root)
material1_scattering_entry.grid(row=1, column=3)
material1_absorption_entry = tk.Entry(root)
material1_absorption_entry.grid(row=1, column=5)
material1_begin_entry = tk.Entry(root)
material1_begin_entry.insert(0, "0")
material1_begin_entry.grid(row=1, column=7)
material1_end_entry = tk.Entry(root)
material1_end_entry.grid(row=1, column=9)

# Material 2
material2_type_entry = tk.Entry(root)
material2_type_entry.grid(row=3, column=1)
material2_scattering_entry = tk.Entry(root)
material2_scattering_entry.grid(row=3, column=3)
material2_absorption_entry = tk.Entry(root)
material2_absorption_entry.grid(row=3, column=5)
material2_begin_entry = tk.Entry(root)
material2_begin_entry.grid(row=3, column=7)
material2_end_entry = tk.Entry(root)
material2_end_entry.grid(row=3, column=9)

# Labels
tk.Label(root, text="Material Type").grid(row=0, column=1)
tk.Label(root, text="Scattering Cross-section").grid(row=0, column=3)
tk.Label(root, text="Absorption Cross-section").grid(row=0, column=5)
tk.Label(root, text="Begin Point").grid(row=0, column=7)
tk.Label(root, text="End Point").grid(row=0, column=9)

# Source data
tk.Button(root, text="Add Source", command=add_source).grid(row=5, column=0)
tk.Label(root, text="Source X").grid(row=5, column=1)
tk.Label(root, text="Source Y").grid(row=5, column=3)
tk.Label(root, text="Source Value").grid(row=5, column=5)

sources = []

#Update grid
material1_end_entry.bind("<KeyRelease>", lambda event: get_values())
material2_end_entry.bind("<KeyRelease>", lambda event: get_values())


# Submit button
tk.Button(root, text="Submit", command=get_values).grid(row=100, column=0)

# Run the Tkinter event loop
root.mainloop()
