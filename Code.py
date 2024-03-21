rom skimage import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, util
from skimage.measure import label, regionprops
from skimage import morphology
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import math
imglist = []
imglist = io.imread_collection(r'C:\Users\ASUS\Desktop\專題研究\Calculate
the rotation speed pf the motor\figures of E.coli in 1MB
buffer(300Hz)\03\*.bmp')
m=40
n=500
All_centriods = [[0 for i in range(n)] for j in range(m)]
Final_All_centriods= [[0 for i in range(n)] for j in range(m)]
a = imglist[1][0:255,0:1280]
b = (a>5)
target = morphology.remove_small_objects(b,10,connectivity=1)
plt.imshow(target)
c = a*target
plt.imshow(c)
'a* is the picture picked from *imglist*'
'*b* is the picture adjusted from *a* by a specific threshold'
'*target* is the picture which has removed small objects from *b*'
'*c* is the picture which only saves brightness at where the beads located'
img = util.img_as_ubyte(target)
centroids = []*500
label_img = []
label_img = label(target, connectivity=2)
centroids = regionprops(label_img)
for t in range(500):
a = imglist[t][0:255,0:1280]

5

b = (a>5)
target = morphology.remove_small_objects(b,10,connectivity=1)
plt.imshow(target)
c = a*target
center = []
for i in range(30):
if centroids[i].centroid[0] > 245 or centroids[i].centroid[0] < 10 or
centroids[i].centroid[1] > 1270 or centroids[i].centroid[1] < 10 :
i=i+1
else :
mytuple =
(round(centroids[i].centroid[0]),round(centroids[i].centroid[1]))
center.append(mytuple)
'*center* is a list stored the geometric center of the beads, which has also
deleted the imformation of beads located around boundaries.'
R = len(center)
beads_centriod = []
for r in range(R):
sum_ix = 0
sum_iy = 0
sum_x_i = 0
sum_y_i = 0
centriod_x = 0
centriod_y = 0
for q in range(-10,10,1):
for p in range(-10,10,1):
sum_iy = sum_iy +
(int(center[r][0])+p)*c[int(center[r][0])+p][int(center[r][1])+q]
sum_ix = sum_ix +
(int(center[r][1])+p)*c[int(center[r][0])+q][int(center[r][1])+p]
sum_y_i = sum_y_i +
c[int(center[r][0])+p][int(center[r][1])+q]
sum_x_i = sum_x_i +
c[int(center[r][0])+q][int(center[r][1])+p]
centriod_x = sum_ix / sum_x_i
centriod_y = sum_iy / sum_y_i
if sum_ix == sum_iy == 0:

6
centriod_x = centriod_x
centriod_y = centriod_y
else:
centriod_x = sum_ix / sum_x_i
centriod_y = sum_iy / sum_y_i
newtuple = (centriod_y,centriod_x)
beads_centriod.append(newtuple)
S = len(beads_centriod)
w = 0
for w in range(S):
All_centriods[w][t] = beads_centriod[w]

w = 0
v = 0
angularspeed = []
for w in range(S):
x = [0]*500
y = [0]*500
complex_data = [0]*500
for t in range(500):
x[t] = All_centriods[w][t][0]
y[t] = All_centriods[w][t][1]
plt.plot(t,y[t],'or')
complex_data[t] = x[t] + y[t]*i
plt.title('y(t)',fontsize=7,color='#7A378B')
plt.show()
abs_fft = np.abs(np.fft.fft(complex_data-np.mean(complex_data)))
abs_fft[0] = abs_fft[1]
freq = np.fft.fftfreq(len(complex_data) , 1/300)
plt.plot(freq,abs_fft,'r')
plt.title('FFT of Mixed wave',fontsize=7,color='#7A378B')
plt.show()
max_intst = np.max(abs_fft)
'*beads_centriod* is a list stored the centroids of the beads'