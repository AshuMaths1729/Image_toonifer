from time import time
start = time()
import numpy as np
import pandas as pd
import glob
import PIL
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.utils import shuffle 
from scipy.ndimage import median_filter, gaussian_filter
import sys

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

path_name = 'test5.jpg'
img = cv2.imread(path_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
w, h, d = original_shape = tuple(img.shape)
#print("Loading image ...")

# Median Filtering
img = median_filter(img, 1)

# Canny Edge Detection
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 1)
edges = cv2.Canny(blurred, 20, 200)
#plt.imshow(edges)

# Color Quantization using KMeans Clustering
imag = np.array(img, dtype=np.float64) / 255
img_array = np.reshape(imag, (w*h, d))
img_array_sample = shuffle(img_array, random_state=0)[:10000]
kmeans = KMeans(n_clusters=50, random_state=42).fit(img_array_sample)
labels = kmeans.predict(img_array)
new_image = recreate_image(kmeans.cluster_centers_, labels, w, h)
#plt.imshow(new_image)

## Applying dilation thrice
kernel = np.ones((3,3),np.uint8)
dl_img = cv2.dilate(edges, kernel, iterations=1)

kernel = np.ones((2,2),np.uint8)
dl_img = cv2.dilate(dl_img, kernel, iterations=1)

kernel = np.ones((2,1),np.uint8)
dl_img = cv2.dilate(dl_img, kernel, iterations=1)

# Bilateral Filtering
bil_filtrd = cv2.bilateralFilter(dl_img, 4, 85, 85)
#plt.imshow(bil_filtrd)

# Median Filtering
med_filtrd = median_filter(bil_filtrd, 7)
#plt.imshow(med_filtrd)

## Performing some image processing for edges to be sharper
edges_inv = cv2.bitwise_not(med_filtrd)
ret,thresh = cv2.threshold(edges_inv,127,255,0)
#plt.imshow(thresh)

## Find contours and draw them
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img_contours = cv2.drawContours(new_image, contours, -1, (0,0,0), 3)
plt.axis('off')
plt.imshow(img_contours)
#plt.savefig('toonified_'+path_name[:-4]+".pdf", format='pdf')
#print("Toonified the image in ",(time() - start),"secs.\n")