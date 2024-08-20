#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:25:48 2023

@author: spotkev
"""


import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from numba import jit
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import time
import csv
import sys
import gc
from spotfunk.res.input import segy_reader
from scipy.ndimage import gaussian_filter


from spotfunk.res import bspline


def load_weight_model(path_param,path_weights):
    """
    Function that allows to load the different files describing the model
    Parameters is an excel file containing data in 1 column.
    parameters are :
    * First INLINE of the cube
    * First CROSSLINE of the cube
    * First depth of the cube
    * INLINE step (= INL size of a cell)
    * CROSSLINE step (= XL size of a cell)
    * z step (positive number : z size of a cell
    * distance (m) between Inline 0 and Inline 1
    * distance (m) between Crossline 0 and Crossline 1
    * azimuth : angle between a major axis and XL or INL (case by case)
    * number of cells in x direction
    * number of cells in y direction
    * number of cells in z direction
    * start x (first x of the cube)
    * start y (first y of the cube)

    Weights are the B-splines Weights (for velocities and horizon)
    :param path: string, path of the different files
    :param number_layer: int, number of layers of the model
    :param project_name: str, name of the project. It matters in the name of the inputs
    :return: 3 lists, 1 with the parameter, 2 with the weights
    """

    Param_Input = []

    with open(path_param, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            Param_Input.append(float(row[0]))
            

    Weights = []

    with open(path_weights, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            Weights.append(float(row[0]))
       
    return Param_Input, Weights





def x_y_z_infos(INL, XL, Param_Input):
    """
    Converts INL XL into x,y coordinates
    :param INL: float, inline you want to convert
    :param XL: float, XL you want to convert
    :param Param_Input: list of Parameters of the model
    :return: floats, x and y given from INL and XL
    """

    INL_step = Param_Input[6]
    XL_step = Param_Input[7]
    azimuth = Param_Input[8]
    # azimuth = 0
    X_or = Param_Input[12]
    Y_or = Param_Input[13]

    x = np.cos(azimuth+np.pi/2) * INL_step * INL + np.sin(azimuth+np.pi/2) * XL_step * XL + X_or
    y = np.sin(azimuth+np.pi/2) * INL_step * INL - np.cos(azimuth+np.pi/2) * XL_step * XL + Y_or
    return x, y




def INL_XL_z_infos(x, y, Param_Input):
    """
    Converts x,y into INL,XL coordinates
    :param x: float, x you want to convert
    :param y: float, y you want to convert
    :param Param_Input: list of Parameters of the model
    :return: floats, INL and XL given from x and y
    """

    INL_step = Param_Input[6]
    XL_step = Param_Input[7]
    azimuth = Param_Input[8]
    X_or = Param_Input[12]
    Y_or = Param_Input[13]

    INL = np.cos(azimuth+np.pi/2) * (1 / INL_step) * (x - X_or) + np.sin(azimuth+np.pi/2) * (1 / INL_step) * (y - Y_or)
    XL = np.sin(azimuth+np.pi/2) * (1 / XL_step) * (x - X_or) - np.cos(azimuth+np.pi/2) * (1 / XL_step) * (y - Y_or)

    return INL, XL

# def B_spline1(x):
#     '''
#     Computes the value of the B-spline at a point x
#     :param x: x/y/z coordinate of a point (one of them)
#     :return: Value of the B-spline at a point x
#     '''

#     if x < 0:
#         return 0

#     elif x < 1:
#         B = 1 / 4 * x * x * x
#     elif x < 2:
#         B = ((-3 / 4 * x + 3) * x - 3) * x + 1
#     elif x < 3:
#         B = ((3 / 4 * x - 6) * x + 15) * x - 11
#     elif x < 4:
#         B = ((-1 / 4 * x + 3) * x - 12) * x + 16
#     else:
#         return 0

#     return B

def derivate_Bspline(x):
    '''
    Computes the value of the first derivative of the B-spline at a point x
    :param x: x/y/z coordinate of a point (one of them)
    :return: Value of the first derivative of the B-spline at a point x
    '''

    Mat_cal = [[-1 / 2, 4, -8],
               [3 / 2, -8, 10],
               [-3 / 2, 4, -2],
               [1 / 2, 0, 0]]

    if x < 0:
        return 0
    elif x < 1:
        B = 0
        for j in range(3):
            B += Mat_cal[3][j] * 3 / 2 * (x) ** (2 - j)
    elif x < 2:
        B = 0
        for j in range(3):
            B += Mat_cal[2][j] * 3 / 2 * (x) ** (2 - j)
    elif x < 3:
        B = 0
        for j in range(3):
            B += Mat_cal[1][j] * 3 / 2 * (x) ** (2 - j)
    elif x < 4:
        B = 0
        for j in range(3):
            B += Mat_cal[0][j] * 3 / 2 * (x) ** (2 - j)

    else:
        return 0

    return B


def second_derivative_bspline(x):
    '''
    Computes the value of the second derivative of the B-spline at a point x
    :param x: x/y/z coordinate of a point (one of them)
    :return: Value of the second derivative of the B-spline at a point x
    '''

    Mat_cal = [[-1, 4],
               [3, -8],
               [-3, 4],
               [1, 0]]

    if x < 0:
        return 0
    elif x < 1:
        B = 0
        for j in range(2):
            B += Mat_cal[3][j] * 3 / 2 * (x) ** (1 - j)
    elif x < 2:
        B = 0
        for j in range(2):
            B += Mat_cal[2][j] * 3 / 2 * (x) ** (1 - j)
    elif x < 3:
        B = 0
        for j in range(2):
            B += Mat_cal[1][j] * 3 / 2 * (x) ** (1 - j)
    elif x < 4:
        B = 0
        for j in range(2):
            B += Mat_cal[0][j] * 3 / 2 * (x) ** (1 - j)

    else:
        return 0

    return B


def Vitesse(x, y, z, Param_Input, Weights, gradient=True, Laplacien=True):
    '''
    Gives multiple data about velocity at a point : value, first and second derivative
    :param x: x coordinate of the point
    :param y: y coordinate of the point
    :param z: z coordinate of the point
    :param Param_Input: list of Parameters of the model
    :param Weights: list of Weights for the B-splines of the velocity
    :return:
    '''

    # if out_of_bounds(x, y, z, Param_Input) == False:
    #     print('Out of Bounds')

    start_x = Param_Input[0]
    start_y = Param_Input[1]
    start_z = -Param_Input[2]
    delta_x = Param_Input[3]
    delta_y = Param_Input[4]
    delta_z = -Param_Input[5]
    INL_step = Param_Input[6]
    XL_step = Param_Input[7]
    azimuth = Param_Input[8]
    I = Param_Input[9]
    J = Param_Input[10]
    K = Param_Input[11]

    INL, XL = INL_XL_z_infos(x, y, Param_Input)
    # print(INL,XL)

    end_x = start_x + (I - 1) * delta_x
    end_y = start_y + (J - 1) * delta_y
    end_z = start_z + (K - 1) * delta_z

    # print(end_z)

    delta_tx = delta_x
    delta_ty = delta_y
    delta_tz = delta_z

    tx = np.arange(start_x - 2 * delta_tx, end_x + delta_tx + 0.01, delta_tx)
    ty = np.arange(start_y - 2 * delta_ty, end_y + delta_ty + 0.01, delta_ty)
    tz = np.arange(start_z - 2 * delta_tz, end_z + delta_tz - 0.01, delta_tz)
    
    
    vitesse = bspline.spline_interp_value_3d(INL, XL, z, Weights, tx, ty, tz, delta_tx, delta_ty, delta_tz)
    
    grad_c = gradient_vitesse(INL, XL, z, Weights, tx, ty, tz, delta_tx, delta_ty, delta_tz, INL_step, XL_step,
                              azimuth)
    
    
    
    return vitesse,grad_c
    

def gradient_vitesse(INL, XL, z, weight, tx, ty, tz, delta_tx, delta_ty, delta_tz, INL_step, XL_step, azimuth, INL_constant=False, XL_constant=False):
    """
    Compute the gradient of the velocity at a point
    :param INL: INL coordinate of the point
    :param XL: XL coordinate of the point
    :param z: z coordinate of the point
    :param weight: list of Weights for the B-splines of the velocity
    :param tx:
    :param ty:
    :param tz:
    :param delta_tx:
    :param delta_ty:
    :param delta_tz:
    :param INL_step: Step between 2 inlines (if first cell is Inline 2 and second cell Inline 5, the step is 3)
    :param XL_step: Step between 2 crosslines (if first cell is Crossline 2 and second cell Crossline 5, the step is 3)
    :param azimuth: angle between a major axis and XL or INL (case by case)
    :return: list, gradient of the velocity at a point
    """

    N = len(ty)
    L = len(tz)

    cpa = 0
    while INL > tx[cpa]:
        cpa += 1
    ttx = [cpa - 2, cpa - 1, cpa, cpa + 1]

    cpb = 0
    while XL > ty[cpb]:
        cpb += 1
    tty = [cpb - 2, cpb - 1, cpb, cpb + 1]

    cpc = 0
    while z < tz[cpc]:
        cpc += 1
    ttz = [cpc - 2, cpc - 1, cpc, cpc + 1]

    derivalue_INL = 0
    derivalue_XL = 0
    derivalue_z = 0

    for dx in ttx:
        for dy in tty:
            for dz in ttz:
                indice_mnl = int(dz + dy * L + dx * N * L)

                derivalue_INL += derivate_Bspline((INL - tx[int(dx)] + 2 * delta_tx) / delta_tx) * B_spline1(
                    (XL - ty[int(dy)] + 2 * delta_ty) / delta_ty) * B_spline1(
                    (z - tz[int(dz)] + 2 * delta_tz) / delta_tz) * weight[indice_mnl]
                derivalue_XL += B_spline1((INL - tx[int(dx)] + 2 * delta_tx) / delta_tx) * derivate_Bspline(
                    (XL - ty[int(dy)] + 2 * delta_ty) / delta_ty) * B_spline1(
                    (z - tz[int(dz)] + 2 * delta_tz) / delta_tz) * weight[indice_mnl]
                derivalue_z += B_spline1((INL - tx[int(dx)] + 2 * delta_tx) / delta_tx) * B_spline1(
                    (XL - ty[int(dy)] + 2 * delta_ty) / delta_ty) * derivate_Bspline(
                    (z - tz[int(dz)] + 2 * delta_tz) / delta_tz) * weight[indice_mnl]

    if INL_constant:
        derivalue_INL = 0
    if XL_constant:
        derivalue_XL = 0
    # derivalue_x = (np.cos(azimuth) * derivalue_INL + np.sin(azimuth) * derivalue_XL) / (delta_tx * INL_step) #Is there a problem here ???
    # derivalue_y = (np.sin(azimuth) * derivalue_INL - np.cos(azimuth) * derivalue_XL) / (delta_ty * XL_step) #Is there a problem here ???
    
    derivalue_x = (np.cos(azimuth+np.pi/2) * derivalue_INL + np.sin(azimuth+np.pi/2) * derivalue_XL) / (delta_tx * INL_step)
    derivalue_y = (np.sin(azimuth+np.pi/2) * derivalue_INL - np.cos(azimuth+np.pi/2) * derivalue_XL) / (delta_ty * XL_step)  
    
    derivalue_z = derivalue_z / delta_tz

    return [derivalue_x, derivalue_y, -derivalue_z]

def read_txt_file(path, filename):
    file = open(path + filename, 'r')
    for i in range(3):
        file.readline() #skip the first 3 lines
    positions = []
    for lines in file:
        try:
            line = lines.split()
            positions.append((float(line[2]), float(line[3])))
        except: #if the last line is blank, we skip
            break
    file.close()
    return positions




class Param_Input_class:
    
    def __init__(self, Param):
        self.start_x_ = Param[0]
        self.start_y_ = Param[1]
        self.start_z_ = Param[2]
        self.delta_x_ = Param[3]
        self.delta_y_ = Param[4]
        self.delta_z_ = Param[5]
        self.INL_step_ = Param[6]
        self.XL_step_ = Param[7]
        self.azimuth_ = Param[8]
        self.I_ = Param[9]
        self.J_ = Param[10]
        self.K_ = Param[11]
        self.X_or_ = Param[12]
        self.Y_or_ = Param[13]

        self.end_x_ = self.start_x_ + (self.I_ - 1) * self.delta_x_
        self.end_y_ = self.start_y_ + (self.J_ - 1) * self.delta_y_
        self.end_z_ = self.start_z_ + (self.K_ - 1) * self.delta_z_

        # print(end_z)
        
        self.tx_ = np.arange(self.start_x_ - 2. * self.delta_x_, self.end_x_ + self.delta_x_ + 0.01, self.delta_x_)
        self.ty_ = np.arange(self.start_y_ - 2. * self.delta_y_, self.end_y_ + self.delta_y_ + 0.01, self.delta_y_)
        self.tz_ = np.arange(self.start_z_ - 2. * self.delta_z_, self.end_z_ + self.delta_z_ + 0.01, self.delta_z_)

@jit
def which_cell_for_zinterface(lowest_z, highest_z, z_step, nz, start_z):
    """
    Function that tells what cell contains the lowest z of the interface
    Goal : build layered B_splines
    We look for base of a cell
    Useful for start_z and K for the layered B-splines
    :param lowest_z: float, lowest z of the interface
    :param highest_z: float, highest z of the interface
    :param z_step : float, size of a cell
    :param nz : int, number of cells in z direction
    :param start_z : first value of z (center of first cell)
    :return:
    """
    high_cell, low_cell = -1, -1 #set high_cell & low_cell to -1 (negative because values are positive)
    for k in range(nz):
        if start_z + (k+1) * z_step >= highest_z and high_cell == -1:
            high_cell = k
        if start_z + (k + 1) * z_step >= lowest_z:  # First cell below interface
            low_cell = k
            break
    return high_cell, low_cell

def spline_interp_value_2d(x,y,weight,tx,ty,delta_tx,delta_ty):
    
    N = len(ty)
    
    cpa = 0
    while x > tx[cpa]:
        cpa += 1
    ttx = [cpa-2,cpa-1,cpa,cpa+1]
    
    cpb = 0
    while y > ty[cpb]:
        cpb += 1
    tty = [cpb-2,cpb-1,cpb,cpb+1]

    value = 0

    for i in range(4):
        a = int(ttx[i])
        ux = B_spline1((x-tx[a]+2*delta_tx)/delta_tx)
        for j in range(4):
            b = int(tty[j])
            indice_mnl = int(b + a*N)
            
            value += ux*B_spline1((y-ty[b]+2*delta_ty)/delta_ty)*weight[indice_mnl]
    return value

def Horizon(x, y, Param_Input, Weights_Horizon):
    '''
    Return z of the horizon given an x and y (and Weights and parameters of the model)
    It returns also the gradient of the horizon (= the normal) at a point
    :param x: float, x coordinate
    :param y: float, y coordinate
    :param Param_Input: object containing parameters of the model
    :param Weights_Horizon: list of Weights for the B-splines of the horizon
    :return: float, np.array ; z value of the interface and gradient at a point
    '''

    start_x = Param_Input.start_x_
    start_y = Param_Input.start_y_
    delta_x = Param_Input.delta_x_
    delta_y = Param_Input.delta_y_

    INL, XL = x,y

    # print(INL,XL)

    end_x = Param_Input.end_x_
    end_y = Param_Input.end_y_

    delta_tx = delta_x
    delta_ty = delta_y

    tx = Param_Input.tx_
    ty = Param_Input.ty_
    # print(tx)
    # print(ty)
    if start_x <= INL <= end_x and start_y <= XL <= end_y:
        vitesse = spline_interp_value_2d(INL, XL, Weights_Horizon, tx, ty, delta_tx, delta_ty)

    else:
        #1 coordinate
        if INL > end_x and start_y <= XL <= end_y:
            vitesse = spline_interp_value_2d(end_x, XL, Weights_Horizon, tx, ty, delta_tx, delta_ty)

        elif INL < start_x and start_y <= XL <= end_y:
            vitesse = spline_interp_value_2d(start_x, XL, Weights_Horizon, tx, ty, delta_tx, delta_ty)

        elif start_x <= INL <= end_x and XL > end_y:
            vitesse = spline_interp_value_2d(INL, end_y, Weights_Horizon, tx, ty, delta_tx, delta_ty)

        elif start_x <= INL <= end_x and XL < start_y:
            vitesse = spline_interp_value_2d(INL, start_y, Weights_Horizon, tx, ty, delta_tx, delta_ty)

        #2 coordinates
        elif INL > end_x and XL > end_y:
            vitesse = spline_interp_value_2d(end_x, end_y, Weights_Horizon, tx, ty, delta_tx, delta_ty)

        elif INL > end_x and XL < start_y:
            vitesse = spline_interp_value_2d(end_x, start_y, Weights_Horizon, tx, ty, delta_tx, delta_ty)

        elif INL < start_x and XL > end_y:
            vitesse = spline_interp_value_2d(start_x, end_y, Weights_Horizon, tx, ty, delta_tx, delta_ty)

        elif INL < start_x and XL < start_y:
            vitesse = spline_interp_value_2d(start_x, start_y, Weights_Horizon, tx, ty, delta_tx, delta_ty)
            

    return -vitesse #We need positive values and horizon values are negative

def where_interface(x0, Param_Input, Weights_Horizon):
    """
    Function that tells if we are on a side or another of a given interface
    :param x0: 3-list, position of the point to check (in x,y,z)
    :param Param_Input: object containing parameters of the model
    :param Weights_Horizon: list of Weights for the B-splines of the horizon
    :return: True if we are above the interface, False if below
    """

    Hori = Horizon(x0[0], x0[1], Param_Input, Weights_Horizon)

    Where = Hori - x0[2]

    # Where = Int_Para[0]*x0[0] + Int_Para[1]*x0[1] + Int_Para[2]*x0[2] - Int_Para[3] # -> equation du plan de l'interface
    # #print(Where)

    if Where > 0:
        return True
    else:
        return False

def build_velocity_grid(Param_Input, Param_Input_Horiz, Weights_Horizon_list, high_cell_list, low_cell_list, velocity_model):
    """
    Goal : fill new grids for layered B-splines
    It works for n layers
    The program returns a list of n lists, each one of these liste containing velocity models for each layer
    :param Param_Input: object containing parameters of the model
    :param Weights_Horizon_list: list of lists of weights for Horizon B-spline (1 list = 1 horizon)
    :param high_cell_list: list, z-ID of the top of the interface i
    :param low_cell_list: list, z-ID of the bottom of the interface i
    :param velocity_model: list, contains the velocities of the model
    :return: grid1, grid2 :
    """
    print('start')
    nx, ny, nz = int(Param_Input.I_), int(Param_Input.J_), int(Param_Input.K_)
    start_z = Param_Input.start_z_
    z_step = Param_Input.delta_z_
    start_INL = Param_Input.start_x_
    start_XL = Param_Input.start_y_
    delta_INL = Param_Input.delta_x_
    delta_XL = Param_Input.delta_y_

    list_grids = []
    for i in range(len(Weights_Horizon_list)+1):
        list_grids.append(velocity_model.copy())
    print('copy done')
    d = len(Weights_Horizon_list)

    for i in range(nx):
        start_time = time.time()
        for j in range(ny):
            position = [start_INL + i*delta_INL,start_XL + j*delta_XL]
            next_position = [start_INL + i*delta_INL,start_XL + j*delta_XL]

            for m in range(len(high_cell_list)):
                for k in range(high_cell_list[m] - 1, low_cell_list[m]+1): #We know the interface.s is.are between those cells
    
                    position.append(start_z + k * z_step)
                    next_position.append(start_z + (k + 1) * z_step)
    
                    l = 0
                    if where_interface(position, Param_Input_Horiz, Weights_Horizon_list[l]) != where_interface(next_position, Param_Input_Horiz, Weights_Horizon_list[l]):
                        #We crossed the interface l, which is between middle of cell k and middle of cell k+1. We have to fill the 2 grids
    
                        #Grid 1
                        c1 = velocity_model[k-1 + nz * j + i * ny * nz] #Taking the value of a cell just before
    
                        a = k
    
                        while a <= low_cell_list[l]+1:
                
                            #From the interface to low_cell, we apply c1 (velocity before interface) (and 1 more for safety)
                            list_grids[l][a + nz * j + i * ny * nz] = c1
                            a+=1
    
                        #Grid 2
                        c2 = velocity_model[k+1 + nz * j + i * ny * nz] #Taking the value of a cell just after
                        b = k
    
    
                        while b >= high_cell_list[l] - 1:
                            #From the interface to high_cell, we apply c2 (velocity after interface) (and 1 more for safety)
                            list_grids[l+1][b + nz * j + i * ny * nz] = c2
                            b-=1
                            
                        l += 1
                        if l == d:
                            break
                    
        print(str(i) + ' DONE in ' + str(time.time() - start_time))


    print('First part done. Now filling the new grids')
    final_grid_list = [[] for i in range(len(Weights_Horizon_list) +1)]
    #Here we filled the grid, we have to cut it now : create a new list and fill it?
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for l in range(len(Weights_Horizon_list)+1): #l is the number of the layer
                    if l == 0: #First layer
                        if k <= low_cell_list[l] + 1:
                            final_grid_list[l].append(list_grids[l][k + nz * j + i * ny * nz])
                    elif l == len(Weights_Horizon_list): #last layer
                        if k >= high_cell_list[l-1] - 1:
                            final_grid_list[l].append(list_grids[l][k + nz * j + i * ny * nz])
                    else: #layer different from the first and the last
                        if high_cell_list[l-1] - 1 <= k <= low_cell_list[l] + 1:
                            final_grid_list[l].append(list_grids[l][k + nz * j + i * ny * nz])

    print('Done creating the new grids')
    return final_grid_list

@jit
def B_spline1(x):
    '''
    Computes the value of the B-spline at a point x
    :param x: x/y/z coordinate of a point (one of them)
    :return: Value of the B-spline at a point x
    '''

    if x < 0:
        return 0

    elif x < 1:
        B = 1 / 4 * x * x * x
    elif x < 2:
        B = ((-3 / 4 * x + 3) * x - 3) * x + 1
    elif x < 3:
        B = ((3 / 4 * x - 6) * x + 15) * x - 11
    elif x < 4:
        B = ((-1 / 4 * x + 3) * x - 12) * x + 16
    else:
        return 0

    return B

@jit
def indice_mnl_g(m, n, l, N, L):
    index = l + n * L + m * N * L

    return index

@jit
def indice_ml_g(m, l, L):
    return l + m * L


def new_conditionnement3d(Mat,M,N,L,I,J,K,alpha):
    
    if M <= 10:
        m_range = np.arange(M)
    else:
        m_range = np.concatenate([np.arange(5),np.arange(M-5,M)])
        
    if N <= 10:
        n_range = np.arange(N)
    else:
        n_range = np.concatenate([np.arange(5),np.arange(N-5,N)])
        
    if L <= 10:
        l_range = np.arange(L)
    else:
        l_range = np.concatenate([np.arange(5),np.arange(L-5,L)])
        
        
    # print(m_range)
    # print(n_range)
    # print(l_range)
    mat_index_count = I*J*K
    count = 0
    
    for m in range(M):
        for n in range(N):
            for l in range(L):
                if (m in m_range) or (n in n_range) or (l in l_range):
                    
                    Mat[mat_index_count, indice_mnl_g(max(0,m - 1), n, l, N, L)] -= alpha * 1
                    Mat[mat_index_count, indice_mnl_g(min(M-1,m + 1), n, l, N, L)] += alpha * 1
                    
                    Mat[mat_index_count, indice_mnl_g(m, max(0,n - 1), l, N, L)] -= alpha * 1
                    Mat[mat_index_count, indice_mnl_g(m, min(N-1,n + 1), l, N, L)] += alpha * 1
                    
                    Mat[mat_index_count, indice_mnl_g(m, n, max(0,l - 1), N, L)] -= alpha * 1
                    Mat[mat_index_count, indice_mnl_g(m, n, min(L-1,l + 1), N, L)] += alpha * 1
                    
                     
                    # Mat[mat_index_count, indice_mnl_g(m, n, l, N, L)] += alpha * 6
                    
                    # Mat[mat_index_count, indice_mnl_g(max(0,m - 1), n, l, N, L)] += alpha * 1
                    # Mat[mat_index_count, indice_mnl_g(min(M-1,m + 1), n, l, N, L)] += alpha * 1
                    
                    # Mat[mat_index_count, indice_mnl_g(m, max(0,n - 1), l, N, L)] += alpha * 1
                    # Mat[mat_index_count, indice_mnl_g(m, min(M-1,n + 1), l, N, L)] += alpha * 1
                    
                    # Mat[mat_index_count, indice_mnl_g(m, n, max(0,l - 1), N, L)] += alpha * 1
                    # Mat[mat_index_count, indice_mnl_g(m, n, min(L-1,l + 1), N, L)] += alpha * 1
                # if m in m_range or n in n_range or l in l_range:
                    
                     
                #     # print(m,n,l,mat_index_count)
                #     # Mat[mat_index_count, indice_mnl_g(m, n, l, N, L)] -= alpha * 3
                #     Mat[mat_index_count, indice_mnl_g(m+1, n, l, N, L)] += alpha * 1
                    
                #     Mat[mat_index_count, indice_mnl_g(m, n+1, l, N, L)] += alpha * 1
                    
                #     Mat[mat_index_count, indice_mnl_g(m, n, l+1, N, L)] += alpha * 1
                    
                    mat_index_count += 1
                    count += 1
    # print(count)
    
    return Mat
                    
    

def conditionnement3d(Mat, M, N, L, I, J, K, alpha):
    # alpha = 0.3

    for m in range(M):
        for n in range(N):
            for l in range(L):
                indice_mnl_line = indice_mnl_g(m, n, l, N, L) + I * J * K

                if m == 0:
                    Mat[indice_mnl_line, indice_mnl_g(m, n, l, N, L)] -= alpha * 1
                    Mat[indice_mnl_line, indice_mnl_g(m + 1, n, l, N, L)] += alpha * 1

                elif m == M - 1:
                    Mat[indice_mnl_line, indice_mnl_g(m, n, l, N, L)] += alpha * 1
                    Mat[indice_mnl_line, indice_mnl_g(m - 1, n, l, N, L)] -= alpha * 1

                else:
                    Mat[indice_mnl_line, indice_mnl_g(m - 1, n, l, N, L)] -= alpha * 1
                    Mat[indice_mnl_line, indice_mnl_g(m + 1, n, l, N, L)] += alpha * 1

                if n == 0:
                    Mat[indice_mnl_line, indice_mnl_g(m, n, l, N, L)] -= alpha * 1
                    Mat[indice_mnl_line, indice_mnl_g(m, n + 1, l, N, L)] += alpha * 1

                elif n == N - 1:
                    Mat[indice_mnl_line, indice_mnl_g(m, n, l, N, L)] += alpha * 1
                    Mat[indice_mnl_line, indice_mnl_g(m, n - 1, l, N, L)] -= alpha * 1

                else:
                    Mat[indice_mnl_line, indice_mnl_g(m, n - 1, l, N, L)] -= alpha * 1
                    Mat[indice_mnl_line, indice_mnl_g(m, n + 1, l, N, L)] += alpha * 1

                if l == 0:
                    Mat[indice_mnl_line, indice_mnl_g(m, n, l, N, L)] -= alpha * 1
                    Mat[indice_mnl_line, indice_mnl_g(m, n, l + 1, N, L)] += alpha * 1

                elif l == L - 1:
                    Mat[indice_mnl_line, indice_mnl_g(m, n, l, N, L)] += alpha * 1
                    Mat[indice_mnl_line, indice_mnl_g(m, n, l - 1, N, L)] -= alpha * 1

                else:
                    Mat[indice_mnl_line, indice_mnl_g(m, n, l - 1, N, L)] -= alpha * 1
                    Mat[indice_mnl_line, indice_mnl_g(m, n, l + 1, N, L)] += alpha * 1

    return Mat

def interp3d(Dataset, Param_Input, limite=100):
    start_INL = Param_Input.start_x_
    start_XL = Param_Input.start_y_
    start_z = Param_Input.start_z_
    delta_INL = Param_Input.delta_x_
    delta_XL = Param_Input.delta_y_
    delta_z = Param_Input.delta_z_
    # INL_step = Param_Input[6]
    # XL_step = Param_Input[7]
    # azimuth = Param_Input[8]
    I = Param_Input.I_
    J = Param_Input.J_
    K = Param_Input.K_
    # X_or = Param_Input[12]
    # Y_or = Param_Input[13]

    M = I + 3
    N = J + 3
    L = K + 3

    end_INL = start_INL + (I - 1) * delta_INL
    end_XL = start_XL + (J - 1) * delta_XL
    end_z = start_z + (K - 1) * delta_z

    delta_tINL = delta_INL
    delta_tXL = delta_XL
    delta_tz = delta_z

    tINL = np.arange(start_INL - 2 * delta_tINL, end_INL + delta_tINL + 0.01, delta_tINL)
    tXL = np.arange(start_XL - 2 * delta_tXL, end_XL + delta_tXL + 0.01, delta_tXL)
    tz = np.arange(start_z - 2 * delta_tz, end_z + delta_tz + 0.01, delta_tz)

    INL = np.arange(start_INL, end_INL + 0.01, delta_INL)
    XL = np.arange(start_XL, end_XL + 0.01, delta_XL)
    z = np.arange(start_z, end_z + 0.01, delta_z)

    start_time = time.time()

    B_spline_INL = np.zeros((I, M))

    for i in range(I):
        m = np.arange(i, i + 3 + 0.001, 1)

        for mm in m:
            try:
                B_spline_INL[i][int(mm)] = B_spline1((INL[i] - tINL[int(mm)] + 2 * delta_tINL) / delta_tINL)
            except:
                pass

    B_spline_XL = np.zeros((J, N))

    for j in range(J):
        n = np.arange(j, j + 3 + 0.001, 1)

        for nn in n:
            try:
                B_spline_XL[j][int(nn)] = B_spline1((XL[j] - tXL[int(nn)] + 2 * delta_tXL) / delta_tXL)
            except:
                pass

    B_spline_z = np.zeros((K, L))

    for k in range(K):
        l = np.arange(k, k + 3 + 0.001, 1)

        for ll in l:
            try:
                B_spline_z[k][int(ll)] = B_spline1((z[k] - tz[int(ll)] + 2 * delta_tz) / delta_tz)
            except:
                pass

    Mat = lil_matrix((I * J * K + min(M,10)*N*L+min(N,10)*L*(M-min(M,10))+min(L,10)*(M-min(M,10))*(N-min(N,10)),
                      M * N * L))
    
    # print(I * J * K + min(M,10)*N*L + min(N,10)*M*L + min(L,10)*M*N)
    # Mat = lil_matrix((I * J * K, M * N * L))
    # Mat = lil_matrix((I * J * K +  M * N * L, M * N * L))

    # export_line = []
    # export_row = []
    # Mat_value = []

    for i in range(I):
        for j in range(J):
            for k in range(K):

                m = np.arange(i, i + 3 + 0.001, 1)
                n = np.arange(j, j + 3 + 0.001, 1)
                l = np.arange(k, k + 3 + 0.001, 1)

                for mm in m:
                    for nn in n:
                        for ll in l:
                            indice_mnl = int(ll + nn * L + mm * N * L)
                            indice_ijk = int(k + j * K + i * K * J)

                            Mat[indice_ijk, indice_mnl] = B_spline_INL[i][int(mm)] * B_spline_XL[j][int(nn)] * \
                                                          B_spline_z[k][int(ll)]

    alpha = 0.2

    Mat = new_conditionnement3d(Mat, M, N, L, I, J, K, alpha)
    # Mat = conditionnement3d(Mat, M, N, L, I, J, K, alpha)
    
    del B_spline_INL
    del B_spline_XL
    del B_spline_z
    gc.collect()

    print("Création de Mat : {}s".format(time.time() - start_time))

    b = np.zeros(I * J * K + min(M,10)*N*L+min(N,10)*L*(M-min(M,10))+min(L,10)*(M-min(M,10))*(N-min(N,10)))
    # b = np.zeros(I * J * K)
    # b = np.zeros(I * J * K+M*N*L)

    for trace_nb in range(I * J):
        i = int(trace_nb / J)
        j = trace_nb - i * J

        for k in range(K):
            indice_ijk = k + j * K + i * J * K

            b[indice_ijk] = Dataset[trace_nb][k]


    start_time = time.time()
    Weights = lsqr(Mat, b, show=True)

    print("Création de weights : {}s".format(time.time() - start_time))
    print("ARnorm : ", Weights[7])
    return Weights[0]

def load_weight_model_bis(path, number_layer, project_name):
    """
    Function that allows to load the different files describing the model
    Parameters is an excel file containing data in 1 column.
    parameters are :
    * First INLINE of the cube
    * First CROSSLINE of the cube
    * First depth of the cube
    * INLINE step (= INL size of a cell)
    * CROSSLINE step (= XL size of a cell)
    * z step (positive number : z size of a cell
    * Step between 2 inlines (if first cell is Inline 2 and second cell Inline 5, the step is 3
    * Step between 2 crosslines (if first cell is Crossline 2 and second cell Crossline 5, the step is 3
    * azimuth : angle between a major axis and XL or INL (case by case)
    * number of cells in x direction
    * number of cells in y direction
    * number of cells in z direction
    * start x (first x of the cube)
    * start y (first y of the cube)

    Weights are the B-splines Weights (for velocities and horizon)
    :param path: string, path of the different files
    :param number_layer: int, number of layers of the model
    :param project_name: str, name of the project. It matters in the name of the inputs
    :return: 3 lists, 1 with the parameter, 2 with the weights
    """

    Param_Input_Horiz = []

    file = open(path + 'Parametres_' + project_name + '_Horizon.txt', 'r')
    for lines in file:
        Param_Input_Horiz.append(float(lines))
        
    file.close()
            
    Param_Input = []

    file = open(path + 'Parametres_' + project_name + '.txt', 'r')
    for lines in file:
        Param_Input.append(float(lines))
        
    file.close()

    Weights_Horizon_list = []
    for i in range(1, number_layer):
        Weights_Horizon = []
        with open(path + 'Weights_' + project_name + '_Horizon' + str(i) + '.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                Weights_Horizon.append(float(row[0]))
        Weights_Horizon_list.append(Weights_Horizon)

    return Param_Input_Horiz, Param_Input, Weights_Horizon_list
  
def compute_and_export(project_name, final_grid, Param_Input, path, i, Param_Input_list):
    # Computing the weights for the velocities for the 2 layers
    Weights = interp3d(final_grid, Param_Input)
    print('Weights for layer ' + str(i+1) +' computed. ')
    # Saving the files (Weights + Parameters for the layer)
    np.savetxt(path + "Weights_" + project_name + "_full_layer" + str(i) + ".csv", Weights, fmt='%f', delimiter=',')
    np.savetxt(path + "Parametres_" + project_name + "_full_layer" + str(i) + ".csv", Param_Input_list, fmt='%f', delimiter=",")




def conditionnement2d(Mat,M,L,I,K,alpha):
    alpha = 0.3

    for m in range(M):

        for l in range(L):
            indice_ml_line = indice_ml_g(m,l,L)+I*K

            if m == 0:
                Mat[indice_ml_line,indice_ml_g(m,l,L)] -= alpha*2
                Mat[indice_ml_line,indice_ml_g(m+1,l,L)] += alpha*2

            elif m == M-1:
                Mat[indice_ml_line,indice_ml_g(m,l,L)] += alpha*2
                Mat[indice_ml_line,indice_ml_g(m-1,l,L)] -= alpha*2

            else:
                Mat[indice_ml_line,indice_ml_g(m-1,l,L)] -= alpha*1
                Mat[indice_ml_line,indice_ml_g(m+1,l,L)] += alpha*1


            if l == 0:
                Mat[indice_ml_line,indice_ml_g(m,l,L)] -= alpha*2
                Mat[indice_ml_line,indice_ml_g(m,l+1,L)] += alpha*2

            elif l == L-1:
                Mat[indice_ml_line,indice_ml_g(m,l,L)] += alpha*2
                Mat[indice_ml_line,indice_ml_g(m,l-1,L)] -= alpha*2

            else:
                Mat[indice_ml_line,indice_ml_g(m,l-1,L)] -= alpha*1
                Mat[indice_ml_line,indice_ml_g(m,l+1,L)] += alpha*1

    return Mat


def interp2d(Dataset,Param_Input,limite = 100):



    start_INL = Param_Input[0]
    start_z = Param_Input[1]
    delta_INL = Param_Input[2]
    delta_z = Param_Input[3]
    # INL_step = Param_Input[6]
    # XL_step = Param_Input[7]
    # azimuth = Param_Input[8]
    I = Param_Input[6]
    K = Param_Input[7]
    # X_or = Param_Input[12]
    # Y_or = Param_Input[13]

    M = I+3
    L = K+3



    end_INL = start_INL + (I-1)*delta_INL
    end_z = start_z + (K-1)*delta_z

    delta_tINL = delta_INL #Space between knots in INL direction
    delta_tz = delta_z #Space between knots in XL/z direction (for horizon, XL)

    tINL = np.arange(start_INL-2*delta_tINL,end_INL+delta_tINL+0.01,delta_tINL) #Knots list in INL
    tz = np.arange(start_z-2*delta_tz,end_z+delta_tz+0.01,delta_tz) #Knots list in XL/z

    INL = np.arange(start_INL,end_INL+0.01,delta_INL) #INL data
    z = np.arange(start_z,end_z+0.01,delta_z) #XL/z data

    start_time = time.time()

    B_spline_INL = np.zeros((I,M))

    for i in range(I):
        m = np.arange(i,i+3+0.001,1)

        for mm in m:
            try:
                B_spline_INL[i][int(mm)] = B_spline1((INL[i]-tINL[int(mm)]+2*delta_tINL)/delta_tINL)
            except:
                pass


    B_spline_z = np.zeros((K,L))

    for k in range(K):
        l = np.arange(k,k+3+0.001,1)

        for ll in l:
            try:
                B_spline_z[k][int(ll)] = B_spline1((z[k]-tz[int(ll)]+2*delta_tz)/delta_tz)
            except:
                pass



    Mat = lil_matrix((I*K+M*L,M*L))


    #export_line = []
    #export_row = []
    #Mat_value = []

    for i in range(I):

        for k in range(K):

            m = np.arange(i,i+3+0.001,1)
            l = np.arange(k,k+3+0.001,1)

            for mm in m:
                for ll in l:


                     indice_ml = int(ll + mm*L)
                     indice_ik = int(k  + i*K)

                     Mat[indice_ik,indice_ml] = B_spline_INL[i][int(mm)]*B_spline_z[k][int(ll)]


    alpha = 0.3
    Mat = conditionnement2d(Mat, M, L, I, K, alpha)

    print("Création de Mat : {}s".format(time.time()-start_time))

    b = np.zeros(I*K+M*L)

    for i in range(I):

        for k in range(K):

            indice_ijk = k + i*K

            b[indice_ijk] = Dataset[i,k]


    start_time = time.time()
    # Weights = lsqr(Mat,b,iter_lim = limite,show=True)
    Weights = lsqr(Mat,b,show=True)

    print("Création de weights : {}s".format(time.time()-start_time))


    return Weights[0]




def readbin(filename,nz,nx):
    with open(filename,'rb') as f:
        im = np.fromfile(f,dtype=np.float32)
    im = im.reshape(nz,nx,order='F')
    return im


def interp1d(Dataset,Param_Input,limite = 100):

    start_INL = Param_Input[0]
    # start_z = Param_Input[1]
    delta_INL = Param_Input[2]
    # delta_z = Param_Input[3]
    # INL_step = Param_Input[6]
    # XL_step = Param_Input[7]
    # azimuth = Param_Input[8]
    I = Param_Input[6]
    # K = Param_Input[7]
    # X_or = Param_Input[12]
    # Y_or = Param_Input[13]

    M = I+3
    # L = K+3



    end_INL = start_INL + (I-1)*delta_INL
    # end_z = start_z + (K-1)*delta_z

    delta_tINL = delta_INL #Space between knots in INL direction
    # delta_tz = delta_z #Space between knots in XL/z direction (for horizon, XL)

    tINL = np.arange(start_INL-2*delta_tINL,end_INL+delta_tINL+0.01,delta_tINL) #Knots list in INL
    # tz = np.arange(start_z-2*delta_tz,end_z+delta_tz+0.01,delta_tz) #Knots list in XL/z

    INL = np.arange(start_INL,end_INL+0.01,delta_INL) #INL data
    # z = np.arange(start_z,end_z+0.01,delta_z) #XL/z data

    start_time = time.time()

    B_spline_INL = np.zeros((I,M))

    for i in range(I):
        m = np.arange(i,i+3+0.001,1)

        for mm in m:
            try:
                B_spline_INL[i][int(mm)] = B_spline1((INL[i]-tINL[int(mm)]+2*delta_tINL)/delta_tINL)
            except:
                pass


    # B_spline_z = np.zeros((K,L))

    # for k in range(K):
    #     l = np.arange(k,k+3+0.001,1)

    #     for ll in l:
    #         try:
    #             B_spline_z[k][int(ll)] = B_spline1((z[k]-tz[int(ll)]+2*delta_tz)/delta_tz)
    #         except:
    #             pass



    Mat = lil_matrix((I,M))


    #export_line = []
    #export_row = []
    #Mat_value = []

    for i in range(I):

        # for k in range(K):

            m = np.arange(i,i+3+0.001,1)
            # l = np.arange(k,k+3+0.001,1)

            for mm in m:
                # for ll in l:


                     indice_ml = int(mm)
                     indice_ik = int(i)

                     Mat[indice_ik,indice_ml] = B_spline_INL[i][int(mm)]


    # alpha = 0.3
    # Mat = conditionnement2d(Mat, M, L, I, K, alpha)

    print("Création de Mat : {}s".format(time.time()-start_time))

    b = np.zeros(I)

    for i in range(I):

        # for k in range(K):

            indice_ijk =  i

            b[indice_ijk] = Dataset[i]


    start_time = time.time()
    # Weights = lsqr(Mat,b,iter_lim = limite,show=True)
    Weights = lsqr(Mat,b,show=True)

    print("Création de weights : {}s".format(time.time()-start_time))


    return Weights[0]




# %%

# path = 'C:/Users/Kevin/SpotLight/SpotLighters - Documents/SpotLight/R&D/DOSSIER_PERSO_SpotLighters_RD/SpotVictor/Demigration_SpotLight_Septembre2023/Demigration_Victor/'
file = '../input/45_marm_ano_v3/fwi_ano_114_percent.dat'
file = '../input/45_marm_ano_v3/fwi_betap_ano.dat'
file = '../input/45_marm_ano_v3/fwi_diff_sm.dat'
file = '../input/45_marm_ano_v3/fwi_diff_sm_ano.dat'
file = '../input/45_marm_ano_v3/fwi_diff_ano.dat'
file = '../input/46_flat_simple_taper/inp_vel_taper_all_ano.dat'

file = '../input/50_ts_model/sm_ano.dat'
# file = '../input/50_ts_model/sm_org.dat'

INL_step = 200 
XL_step = 12.00   
azimuth1 = 90
azimuth = azimuth1*2*np.pi/360
X_or = 0
Y_or = 0

I = 5
J = 601
K = 151

M = I+3
N = J+3
L = K+3

start_x = -2
start_y = 0
start_z = 0

delta_x = 1
delta_y = 1
delta_z = 12.00
            
Param_Input1 = [start_x,start_y,start_z,
              delta_x,delta_y,delta_z,
              INL_step,XL_step,azimuth,
              I,J,K,X_or,Y_or]


Vit_model1 = readbin(file,151,601).T*1000

Vit_model1 = Vit_model1

# Vit_model = np.vstack([Vit_model1,Vit_model1,Vit_model1,
#                       Vit_model1,Vit_model1])

x_disc = np.arange(601)*12.00
z_disc = np.arange(151)*12.00

d_max = 150*12.00
d_interp = np.arange(0,d_max-1,5)

x_spline = np.arange(601)*12.00
z_spline = np.arange(len(d_interp))*5
                    
plt.figure(figsize=(16,8))
plt.imshow(Vit_model1.T,vmin=1500,vmax=4500,aspect = 2, extent=(x_disc[0],x_disc[-1],z_disc[-1],z_disc[0]))
plt.colorbar()
# plt.gca().invert_yaxis()

# Vit_model1 = gaussian_filter(Vit_model1,5)
# plt.figure(figsize=(16,8))
# plt.imshow(Vit_model1.T,vmin=1500,vmax=3000,aspect = 2, extent=(x_disc[0],x_disc[-1],z_disc[-1],z_disc[0]))
# plt.colorbar()
# plt.gca().invert_yaxis()

Param_Input = Param_Input_class(Param_Input1)

Param_Input = [start_y,start_z,
              delta_y,delta_z,
              XL_step,1,
              J,K]


############################################################
# Interpolation Bspline 2D puis extension des poids en 3D.

Weights = interp2d(Vit_model1, Param_Input)


Weights_2D_mat = Weights.reshape((N,L))

Weights_3D_mat = Weights_2D_mat[np.newaxis,:,:]*np.ones(M)[:,np.newaxis,np.newaxis]*2/3

Weight_3D_inline = Weights_3D_mat.reshape(M*N*L)

############################################################


Param_Exit = [start_x,start_y,start_z,
              delta_x,delta_y,delta_z,
              INL_step,XL_step,azimuth,
              I,J,K,X_or,Y_or]

np.savetxt('../../../../Demigration_SpotLight_Septembre2023/063_TS_analytique/060_Param_marm_sm_ano.csv', Param_Exit, fmt='%f',delimiter=",")   

np.savetxt('../../../../Demigration_SpotLight_Septembre2023/063_TS_analytique/060_Weights_marm_sm_ano.csv',Weight_3D_inline,fmt='%f',delimiter=',') 

# np.savetxt('Parametres_vel_full_kevtest.csv', Param_Exit, fmt='%f',delimiter=",")   

# np.savetxt('Weights_vel_full_kevtest.csv',Weight_3D_inline,fmt='%f',delimiter=',') 


################################################



file_pick = '../input/40_marm_ano/binv_mig_pick_smooth.csv'
# file_pick = '../../../../Demigration_SpotLight_Septembre2023/input/041_marm2_slope/hz_inv.csv'


data_horizon = [] 
with open(file_pick, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        data_horizon.append(float(row[0]))
data_horizon = np.array(data_horizon)


Weights_hz = interp1d(data_horizon, Param_Input)

Weights_2D_mat_hz = Weights_hz[np.newaxis,:]*np.ones(M)[:,np.newaxis]*2/3

Weight_2D_inline = Weights_2D_mat_hz.reshape(M*N)

plt.figure()
plt.plot(Weight_2D_inline)

# np.savetxt('../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/test.csv', Param_Exit, fmt='%f',delimiter=",")   

# np.savetxt('../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/042_weights_hz_pick_inv_slope.csv',Weight_2D_inline,fmt='%f',delimiter=',') 


# %% Visualisations 


import tqdm

Param_File = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Param_marm_smooth.csv'
Weight_File = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Weights_marm_2p5D_smooth.csv'


# Param_File = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/015_Parametres_vel_marm_ext_bspline_90.csv'
# Weight_File = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/015_Weights_vel_marm_ext_bspline_90.csv'

# Param_File = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/046_Param_marm_smooth.csv'
# Weight_File = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/046_Weights_marm_2p5D_smooth.csv'



file = '../input/27_marm/marm2_sm15.dat'



Parameters,Weights = load_weight_model(Param_File, Weight_File)


Vit_model1 = readbin(file,151,601).T*1000

Vit_model = np.vstack([Vit_model1,Vit_model1,Vit_model1,
                      Vit_model1,Vit_model1])



d_max = (151)*12.00

d_interp = np.arange(0,d_max,12)

# d_max = 151*12.00
# d_interp = np.arange(-delta_z,d_max-1,5)


Bspline_2D = []

# for k in tqdm.tqdm(range(0,601,5)):
for k in tqdm.tqdm(range(0,601,1)):
    vitesse_list = []
    for j in range(0,len(d_interp)):
    # for j in range(0,len(d_interp),1):
        vitesse_list.append(Vitesse(0,12.00*k,-d_interp[j],Parameters,Weights)[0])
    
    Bspline_2D.append(vitesse_list)

Bspline_2D = np.array(Bspline_2D)

x_disc = np.arange(601)*12.00
z_disc = np.arange(151)*12.00

x_spline = np.arange(601)*12.00
z_spline = np.arange(0,len(d_interp))*12
                    
plt.figure(figsize=(16,8))
plt.imshow(Vit_model1.T,vmin=1500,vmax=3000,aspect = 2, extent=(x_disc[0],x_disc[-1],z_disc[-1],z_disc[0]))
plt.colorbar()
# plt.gca().invert_yaxis()

plt.figure(figsize=(16,8))
plt.imshow(Bspline_2D.T,vmin=1500,vmax=3000,aspect = 2, extent=(x_spline[0],x_spline[-1],z_spline[-1],z_spline[0]))
plt.colorbar()
# plt.gca().invert_yaxis()


colors = ['b','g','r','k','m']
plt.figure(figsize=(10,8))
for i in range(0,151,35):
    plt.plot(z_disc,Vit_model1.T[:,i],'-',color=colors[int(i/35)],label='bspline tr = '+str(i))
    plt.plot(z_spline,Bspline_2D.T[:,i],'.',color=colors[int(i/35)],alpha=0.7,label='model tr = '+str(i))
    
    plt.title('Model Bspline vs original model trace')
    plt.legend(fontsize = 'x-small')
    plt.tight_layout()
    plt.xlabel('Depth (km)')
    plt.ylabel('Velocity (m/s)')



#%%
''''''

y_line = np.arange(-390,390,1)


test_y = []
grad_x = []
grad_y = []
grad_z = []

for y in y_line:
    
    vitesse,grad = Vitesse(4000,y,-1000,Parameters,Weights)
    
    test_y.append(vitesse)
    grad_x.append(grad[0])
    grad_y.append(grad[1])
    grad_z.append(grad[2])
    
plt.figure(figsize=(16,12))
plt.plot(y_line,test_y)
plt.xlabel('Y (m)',fontsize=25) 
plt.ylabel('Vitesse (m/s)',fontsize=25) 
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.title('Test selon axe y',fontsize=25)    

# plt.figure(figsize=(16,12))
# plt.plot(y_line,test_y)
# plt.xlabel('Y (m)',fontsize=18) 
# plt.ylabel('Vitesse (m/s)',fontsize=18) 
# plt.xticks(fontsize=18)
# plt.xticks(fontsize=18)
# plt.ylim(2330,2340)
# plt.title('Test selon axe y (zoom)',fontsize=18)



plt.figure(figsize=(16,12))
plt.plot(y_line,grad_x)
plt.xlabel('Y (m)',fontsize=18) 
plt.ylabel('Gradient de vitesse',fontsize=18) 
plt.xticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title('Test selon axe y (gradient en x)',fontsize=18)

plt.figure(figsize=(16,12))
plt.plot(y_line,grad_y)
plt.xlabel('Y (m)',fontsize=18) 
plt.ylabel('Gradient de vitesse',fontsize=18) 
plt.xticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title('Test selon axe y (gradient en y)',fontsize=18)

plt.figure(figsize=(16,12))
plt.plot(y_line,grad_z)
plt.xlabel('Y (m)',fontsize=18) 
plt.ylabel('Gradient de vitesse',fontsize=18) 
plt.xticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title('Test selon axe y (gradient en z)',fontsize=18)




x_line = np.arange(10,7200,1)

test_y = []
grad_x = []
grad_y = []
grad_z = []

for x in x_line:
    
    vitesse,grad = Vitesse(x,0,-1000,Parameters,Weights)
    
    test_y.append(vitesse)
    grad_x.append(grad[0])
    grad_y.append(grad[1])
    grad_z.append(grad[2])
    
plt.figure(figsize=(16,12))
plt.plot(x_line,test_y)
plt.xlabel('X (m)',fontsize=18) 
plt.ylabel('Vitesse (m/s)',fontsize=18) 
plt.xticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title('Test selon axe x',fontsize=18)    

plt.figure(figsize=(16,12))
plt.plot(x_line,grad_x)
plt.xlabel('X (m)',fontsize=18) 
plt.ylabel('Gradient de vitesse',fontsize=18) 
plt.xticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title('Test selon axe x (gradient en x)',fontsize=18)

plt.figure(figsize=(16,12))
plt.plot(x_line,grad_y)
plt.xlabel('X (m)',fontsize=18) 
plt.ylabel('Gradient de vitesse',fontsize=18) 
plt.xticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title('Test selon axe x (gradient en y)',fontsize=18)

plt.figure(figsize=(16,12))
plt.plot(x_line,grad_z)
plt.xlabel('X (m)',fontsize=18) 
plt.ylabel('Gradient de vitesse',fontsize=18) 
plt.xticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title('Test selon axe x (gradient en z)',fontsize=18)




z_line = np.arange(10,1800,1)


test_y = []
grad_x = []
grad_y = []
grad_z = []

for z in z_line:
    
    vitesse,grad = Vitesse(4000,0,-z,Parameters,Weights)
    
    test_y.append(vitesse)
    grad_x.append(grad[0])
    grad_y.append(grad[1])
    grad_z.append(grad[2])
    

plt.figure(figsize=(16,12))
plt.plot(z_line,test_y)
plt.xlabel('Z (m)',fontsize=18) 
plt.ylabel('Vitesse (m/s)',fontsize=18) 
plt.xticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title('Test selon axe z',fontsize=18)    


plt.figure(figsize=(16,12))
plt.plot(z_line,grad_x)
plt.xlabel('Z (m)',fontsize=18) 
plt.ylabel('Gradient de vitesse',fontsize=18) 
plt.xticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title('Test selon axe z (gradient en x)',fontsize=18)


plt.figure(figsize=(16,12))
plt.plot(z_line,grad_y)
plt.xlabel('Z (m)',fontsize=18) 
plt.ylabel('Gradient de vitesse',fontsize=18) 
plt.xticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title('Test selon axe z (gradient en y)',fontsize=18)


plt.figure(figsize=(16,12))
plt.plot(z_line,grad_z)
plt.xlabel('Z (m)',fontsize=18) 
plt.ylabel('Gradient de vitesse',fontsize=18) 
plt.xticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title('Test selon axe z (gradient en z)',fontsize=18)




