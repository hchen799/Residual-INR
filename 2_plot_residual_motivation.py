# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:34:55 2024

@author: hanqi
"""


import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def normalize_array(arr):
    """
    Normalize an array to the range [0, 1].
    
    Parameters:
    arr (numpy.ndarray): Input array to be normalized.
    
    Returns:
    numpy.ndarray: Normalized array with values in the range [0, 1].
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val) if max_val != min_val else arr - min_val
    return normalized_arr


original_obj_img1 = np.load(r"C:\Users\hanqi\Desktop\residual\cropped_object_original_boat0.npy")
original_obj_img2 = np.load(r"C:\Users\hanqi\Desktop\residual\cropped_object_original_boat3.npy")
original_obj_img3 = np.load(r"C:\Users\hanqi\Desktop\residual\cropped_object_original_person5.npy")

residual_obj_img1 = np.load(r"C:\Users\hanqi\Desktop\residual\cropped_object_boat.npy")
residual_obj_img2 = np.load(r"C:\Users\hanqi\Desktop\residual\cropped_object_boat3.npy")
residual_obj_img3 = np.load(r"C:\Users\hanqi\Desktop\residual\cropped_object_0_person.npy")

original_obj_img1_flat = original_obj_img1.flatten()
count_original_obj_img1_flat = Counter(original_obj_img1_flat)
elements_1 = np.array(list(count_original_obj_img1_flat.keys()))
normalized_elements_1 = normalize_array(elements_1)
counts_1 = np.array(list(count_original_obj_img1_flat.values()))


residual_obj_img1_flat = residual_obj_img1.flatten()
count_residual_obj_img1_flat = Counter(residual_obj_img1_flat)
elements_2 = np.array(list(count_residual_obj_img1_flat.keys()))
elements_2 = elements_2.astype(np.float64)
normalized_elements_2 = normalize_array(elements_2)
counts_2 = np.array(list(count_residual_obj_img1_flat.values()))


original_obj_img2_flat = original_obj_img2.flatten()
count_original_obj_img2_flat = Counter(original_obj_img2_flat)
elements_3 = np.array(list(count_original_obj_img2_flat.keys()))
normalized_elements_3 = normalize_array(elements_3)
counts_3 = np.array(list(count_original_obj_img2_flat.values()))

residual_obj_img2_flat = residual_obj_img2.flatten()
count_residual_obj_img2_flat = Counter(residual_obj_img2_flat)
elements_4 = np.array(list(count_residual_obj_img2_flat.keys()))
elements_4 = elements_4.astype(np.float64)
normalized_elements_4 = normalize_array(elements_4)
counts_4 = np.array(list(count_residual_obj_img2_flat.values()))


original_obj_img3_flat = original_obj_img3.flatten()
count_original_obj_img3_flat = Counter(original_obj_img3_flat)
elements_5 = np.array(list(count_original_obj_img3_flat.keys()))
normalized_elements_5 = normalize_array(elements_5)
counts_5 = np.array(list(count_original_obj_img3_flat.values()))

residual_obj_img3_flat = residual_obj_img3.flatten()
count_residual_obj_img3_flat = Counter(residual_obj_img3_flat)
elements_6 = np.array(list(count_residual_obj_img3_flat.keys()))
elements_6 = elements_6.astype(np.float64)
normalized_elements_6 = normalize_array(elements_6)
counts_6 = np.array(list(count_residual_obj_img3_flat.values()))


y_positions = [1, 2, 3, 4, 5, 6]  # Assigning each sample a different y-axis position
plt.figure(figsize=(10, 3))
#plt.scatter(normalized_elements_1, np.full_like(normalized_elements_1, y_positions[0]), alpha=0.6, label='Residual Encoding', color='blue', marker = 'x', s = 5.0)
plt.scatter(normalized_elements_3, np.full_like(normalized_elements_3, y_positions[1]), alpha=0.6, label='Residual Encoding', color='blue', marker = 'x', s = 5.0)
plt.scatter(normalized_elements_5, np.full_like(normalized_elements_5, y_positions[2]), alpha=0.6, label='Residual Encoding', color='blue', marker = 'x', s = 5.0)

#plt.scatter(normalized_elements_2, np.full_like(normalized_elements_2, y_positions[3]), alpha=0.6, label='Direct Encoding', color='green', marker = 'x', s = 5.0)
plt.scatter(normalized_elements_4, np.full_like(normalized_elements_4, y_positions[4]), alpha=0.6, label='Direct Encoding', color='green', marker = 'x', s = 5.0)
plt.scatter(normalized_elements_6, np.full_like(normalized_elements_6, y_positions[5]), alpha=0.6, label='Direct Encoding', color='green', marker = 'x', s = 5.0)
plt.yticks(y_positions, ['Img 1', 'Img 2', 'Img 3', 'Img 1', ' Img 2', 'Img 3'])  # Label y-axis positions
plt.ylim(0.5,6.5)
#plt.legend()
plt.show()

normalized_original_obj_img1 = normalize_array(original_obj_img1_flat)
normalized_original_obj_img2 = normalize_array(original_obj_img2_flat)
normalized_original_obj_img3 = normalize_array(original_obj_img3_flat)
np.save("normalized_original_obj_img1.npy",normalized_original_obj_img1)

normalized_residual_obj_img1 = normalize_array(residual_obj_img1_flat.astype(np.float64))
normalized_residual_obj_img2 = normalize_array(residual_obj_img2_flat.astype(np.float64))
normalized_residual_obj_img3 = normalize_array(residual_obj_img3_flat.astype(np.float64))


# Setting up the plot

fig, ax1 = plt.subplots(figsize=(5, 3))
n, bins, patches = plt.hist(normalized_residual_obj_img1, bins=50, color='#A4D233', edgecolor='black')
ax1.set_xlabel('Normalized RGB Value', fontsize=16)
ax1.tick_params(axis='x', labelcolor='black', labelsize=16)
ax1.ticklabel_format(style = 'sci', axis = 'y', scilimits=(0,0))
ax1.set_ylabel('Appearance Count', fontsize=16)
ax1.tick_params(axis='y', labelcolor='black', labelsize=16)
ax1.set_ylim(0,2500)
#plt.title('Distribution of Normalized Values', fontsize=16)
plt.show()


fig, ax1 = plt.subplots(figsize=(5, 3))
n, bins, patches = plt.hist(normalized_residual_obj_img2, bins=50, color='#A4D233', edgecolor='black')
ax1.set_xlabel('Normalized RGB Value', fontsize=16)
ax1.tick_params(axis='x', labelcolor='black', labelsize=16)
ax1.ticklabel_format(style = 'sci', axis = 'y', scilimits=(0,0))
ax1.set_ylabel('Appearance Count', fontsize=16)
ax1.tick_params(axis='y', labelcolor='black', labelsize=16)
ax1.set_ylim(0,500)
#plt.title('Distribution of Normalized Values', fontsize=16)
plt.show()

fig, ax1 = plt.subplots(figsize=(5, 3))
n, bins, patches = plt.hist(normalized_residual_obj_img3, bins=50, color='#A4D233', edgecolor='black')
ax1.set_xlabel('Normalized RGB Value', fontsize=16)
ax1.tick_params(axis='x', labelcolor='black', labelsize=16)
ax1.ticklabel_format(style = 'sci', axis = 'y', scilimits=(0,0))
ax1.set_ylabel('Appearance Count', fontsize=16)
ax1.tick_params(axis='y', labelcolor='black', labelsize=16)
#plt.title('Distribution of Normalized Values', fontsize=16)
ax1.set_ylim(0,350)
plt.show()


fig, ax1 = plt.subplots(figsize=(5, 3))
n, bins, patches = plt.hist(normalized_original_obj_img1, bins=50, color='#53ABD8', edgecolor='black')
ax1.set_xlabel('Normalized RGB Value', fontsize=16)
ax1.tick_params(axis='x', labelcolor='black', labelsize=16)
ax1.ticklabel_format(style = 'sci', axis = 'y', scilimits=(0,0))
ax1.set_ylabel('Appearance Count', fontsize=16)
ax1.tick_params(axis='y', labelcolor='black', labelsize=16)
ax1.set_ylim(0,2500)
#plt.title('Distribution of Normalized Values', fontsize=16)
plt.show()

fig, ax1 = plt.subplots(figsize=(5, 3))
n, bins, patches = plt.hist(normalized_original_obj_img2, bins=50, color='#53ABD8', edgecolor='black')
ax1.set_xlabel('Normalized RGB Value', fontsize=16)
ax1.tick_params(axis='x', labelcolor='black', labelsize=16)
ax1.ticklabel_format(style = 'sci', axis = 'y', scilimits=(0,0))
ax1.set_ylabel('Appearance Count', fontsize=16)
ax1.tick_params(axis='y', labelcolor='black', labelsize=16)
#plt.title('Distribution of Normalized Values', fontsize=16)
ax1.set_ylim(0,500)
plt.show()

fig, ax1 = plt.subplots(figsize=(5, 3))
n, bins, patches = plt.hist(normalized_original_obj_img3, bins=50, color='#53ABD8', edgecolor='black')
ax1.set_xlabel('Normalized RGB Value', fontsize=16)
ax1.tick_params(axis='x', labelcolor='black', labelsize=16)
ax1.ticklabel_format(style = 'sci', axis = 'y', scilimits=(0,0))
ax1.set_ylabel('Appearance Count', fontsize=16)
ax1.tick_params(axis='y', labelcolor='black', labelsize=16)
ax1.set_ylim(0,350)
#plt.title('Distribution of Normalized Values', fontsize=16)
plt.show()




# Adjusting the sample code to plot different samples at different y-axis positions
'''
import numpy as np
# Creating new y-axis positions for each sample
y_positions = [1, 1.1, 1.2, 1.3, 1.4, 1.5]  # Assigning each sample a different y-axis position

data1 = np.random.randn(100)
data2 = np.random.randn(100) + 2  # Shifted by +2 for differentiation
data3 = np.random.randn(100) - 2  # Shifted by -2 for differentiation
data4 = np.random.randn(100) - 1
data5 = np.random.randn(100) + 1  # Shifted by +2 for differentiation
data6 = np.random.randn(100) - 0.5  # Shifted by -2 for differentiation

# Creating the scatter plot with adjusted y-axis positions
plt.figure(figsize=(10, 4))
plt.scatter(data1, np.full_like(data1, y_positions[0]), alpha=0.6, label='Sample 1', color='blue')
plt.scatter(data2, np.full_like(data2, y_positions[1]), alpha=0.6, label='Sample 2', color='green')
plt.scatter(data3, np.full_like(data3, y_positions[2]), alpha=0.6, label='Sample 3', color='red')
plt.scatter(data4, np.full_like(data4, y_positions[3]), alpha=0.6, label='Sample 4', color='blue')
plt.scatter(data5, np.full_like(data5, y_positions[4]), alpha=0.6, label='Sample 5', color='green')
plt.scatter(data6, np.full_like(data6, y_positions[5]), alpha=0.6, label='Sample 6', color='red')

# Adding title and legend
plt.title('1-D Scatter Plot with Multiple Data Samples at Different Y Positions')
plt.yticks(y_positions, ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6'])  # Label y-axis positions
plt.ylim(0.9, 1.6)
plt.legend()

# Show plot
plt.show()
'''
