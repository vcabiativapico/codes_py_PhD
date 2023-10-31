# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:55:57 2022

@author: KilyannRICHARD
"""


import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import time
import csv
# from spotfunk.res.input import segy_reader
from numba import jit

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
def indice_mnl_g(m,n,l,N,L):

    index = l+n*L+m*N*L

    return index

@jit
def indice_ml_g(m,l,L):

    return l+m*L



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


def interp2d(Dataset,Param_Input,limite = 100, facteur=1):



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




    end_INL = start_INL + (I-1)*delta_INL
    end_z = start_z + (K-1)*delta_z

    delta_tINL = facteur * delta_INL #Space between knots in INL direction
    delta_tz = facteur * delta_z #Space between knots in XL/z direction (for horizon, XL)
 
    #Knots list in INL
    tINL = [start_INL - 2* delta_tINL]
    last_value = start_INL - 2* delta_tINL
    while last_value < end_INL:
        last_value += delta_tINL
        tINL.append(last_value)
    tINL.append(last_value + delta_tINL)

    #Knots list in XL/z
    tz = [start_z - 2* delta_tz]
    last_value = start_z - 2* delta_tz
    while last_value < end_z:
        last_value += delta_tz
        tz.append(last_value)
    tz.append(last_value + delta_tz)

    M, L = len(tINL), len(tz)


    INL = np.arange(start_INL,end_INL+0.01,delta_INL)
    z = np.arange(start_z,end_z+0.01,delta_z)

    start_time = time.time()

    B_spline_INL = np.zeros((I,M))

    for i in range(I):
        start = np.ceil(i/facteur)
        m = np.arange(start, start + 3 + 0.001, 1)

        for mm in m:
            try:
                B_spline_INL[i][int(mm)] = B_spline1((INL[i]-tINL[int(mm)]+2*delta_tINL)/delta_tINL)
            except:
                pass


    B_spline_z = np.zeros((K,L))

    for k in range(K):
        start = np.ceil(k/facteur)
        l = np.arange(start, start + 3 + 0.001, 1)
        
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

            start = np.ceil(i/facteur)
            m = np.arange(start, start + 3 + 0.001, 1)
            start = np.ceil(k/facteur)
            l = np.arange(start, start + 3 + 0.001, 1)

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

            b[indice_ijk] = -Dataset[indice_ijk] #We want negative values


    start_time = time.time()
    # Weights = lsqr(Mat,b,iter_lim = limite,show=True)
    Weights = lsqr(Mat,b,show=True)

    print("Création de weights : {}s".format(time.time()-start_time))


    return Weights[0]





# %% Paramètres

#Put there the path of your csv files containing following data : INL, XL, z
#PLEASE sort it by INL AND THEN by XL
path = './Model_Vit_discret/'

#Path where the files will be written
out_path = './'

#Names of your horizon files
file1 = path + 'table_pick_testing.csv'

#list storing the files, used for a loop

list_files = [file1]


INL_step = 12 
XL_step = 200
azimuth1 = 0
azimuth = azimuth1*2*np.pi/360
X_or = 0
Y_or = 0

I = 601
J = 5
K = 151

M = I+3
N = J+3
L = K+3



start_x = 0
start_y = 0
start_z = 0

delta_x = 1
delta_y = 1
delta_z = 12 
            
Param_Input = [start_x,start_y,
              delta_x,delta_y,
              INL_step,XL_step,
              I,J]

fact = 8 #Smoothing factor. It is useful to space B-spline knots ; the space between knots will be delta_x * fact and delta_y * fact
#Recommended value : 8

project_name = "These_Victor" #Name used to label the outputs

#END OF FILLING PARAMETERS
##########################################

i = 0
for file in list_files:

    VDataset_temp = []
    with open(file, newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=';')
                count = 0
                for row in spamreader:
                    if count == 0:
                        count += 1
                        continue
                
                    VDataset_temp = row[0].split(',')
    VDataset = np.array([float(i) for i in VDataset_temp])
    
    VDataset1 = np.vstack([VDataset,VDataset,VDataset,VDataset,VDataset]).T.reshape(5*601)
    
    Weights = interp2d(VDataset1, Param_Input,limite=200, facteur=fact)
    i+=1
    
    np.savetxt(out_path + "Weights_" + project_name + "_Horizon_new_splines_smooth" + str(i) + ".csv",Weights,fmt='%f',delimiter=',')

# Weights_Horizon = []
# with open(out_path + 'Weights_' + project_name + '_Horizon_new_splines' + str(1) + '.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     for row in spamreader:
#         Weights_Horizon.append(0)

# np.savetxt(out_path + "Weights_"+ project_name +"_Horizon_new_splines" + str(0) + ".csv",Weights_Horizon,fmt='%f',delimiter=',')     
#Save the weights for horizon located at depth 0 (useful for multiples)