#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:22:03 2024

@author: vcabiativapico
"""



import numpy as np
import os
import glob
import matplotlib.pyplot  as plt
from tqdm import trange
from PIL import Image 

directory = "/home/vcabiativapico/Téléchargements/FWIOpenData/kimberlina_co2_official/kimberlina_co2_train_label/"
file_paths = glob.glob(os.path.join(directory, "*.npz"))
# Initialize an empty list to store loaded arrays
stacked_array = []
# Load each file and append to the list
for file_path in file_paths:
 array = np.load(file_path)['label']
 stacked_array.append(array.T) # Squeeze to remove the singleton dimension

# target images size
nz, nx = 151, 601

images = [Image.fromarray(arr) for arr in stacked_array]
resized_images = [img.resize((nx, nz), Image.LANCZOS) for img in images]
resized_array = np.array([np.array(img) for img in resized_images])
print(resized_array.shape)

n = np.random.randint(500)
n=0
print(n)
plt.figure()
plt.title(n)
vmin, vmax = resized_array.min(), resized_array.max()
plt.imshow(resized_array[n,:,:],aspect='auto',cmap='jet',vmin=vmin,vmax=vmax)
plt.colorbar()




# file_path = "kimberlina_co2.npy"
# np.save(file_path, resized_array)


# ## Read npy
# file_path = "kimberlina_co2.npy"
# cropped_images = np.load(file_path)
# vmin, vmax = cropped_images.min(), cropped_images.ma)
# print('vmin', vmin, 'vmax', vmax)

# n = np.random.randint(500)
# plt.figure()
# plt.imshow(cropped_images[n,:,:],cmap='jet',vmin=vmin,vmax=vmax)
# plt.colorbar()
