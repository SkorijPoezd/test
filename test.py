import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

with open("eye_position/BioID_0001.eye") as file:
    file.readline()
    LX, LY, RX, RY = map(int,file.readline().split("\t"))

img = cv2.imread("eye_position/BioID_0001.pgm")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = img[LY-10:LY+10,LX-20:LX+20]
img_flat = img.flatten()
img_flat = img_flat.reshape(img_flat.shape + (1,))


km=KMeans(n_clusters=3)
km.fit(img_flat)

for i in range(len(km.labels_)):
    if km.labels_[i]==0:
        img_flat[i]=0

    if km.labels_[i]==1:
        img_flat[i]=127

    if km.labels_[i]==2:
        img_flat[i]=255
img = img_flat.reshape(img.shape)
#cv2.circle(img, (LX,LY), 0, (0,255,0))
#cv2.circle(img, (RX,RY), 0, (0,255,0))
#markers = cv2.watershed(img,markers)
plt.imshow(img,cmap=plt.cm.gray)
plt.show()