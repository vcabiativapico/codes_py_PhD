#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:08:09 2024

@author: vcabiativapico
"""

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import cascaded_union, polygonize
import numpy as np
import csv
from scipy.spatial import Delaunay


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha shape.
        return MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points, if not in the list already.
        """
        if (i, j) in edges or (j, i) in edges:
            # Already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = np.array(points)
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = MultiPoint([coords[edge[0]] for edge in edges] + [coords[edge[1]] for edge in edges])
    return cascaded_union(list(polygonize(m.convex_hull)))

# Define the grid size
x = np.linspace(0, 10, 11)
y = np.linspace(0, 10, 11)
xx, yy = np.meshgrid(x, y)
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Select points within a circular area
center = np.array([5, 5])
radius = 3
area_points = grid_points[np.linalg.norm(grid_points - center, axis=1) < radius]

# Compute the alpha shape of the area points
alpha = 1.5  # You can adjust the alpha parameter to fit your needs
polygon = alpha_shape(area_points, alpha)

# Plot the grid points, area points, and the alpha shape
plt.scatter(grid_points[:, 0], grid_points[:, 1], c='blue', label='Grid Points')
plt.scatter(area_points[:, 0], area_points[:, 1], c='green', label='Area Points')

x, y = polygon.exterior.xy
plt.plot(x, y, 'r-', linewidth=2, label='Alpha Shape')

plt.title('Alpha Shape of Selected Area Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()


def read_pick(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
    return attr
