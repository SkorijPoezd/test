import cv2
import numpy as np
from imutils import rotate

def get_display_img(img):

    #img=cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr=cv2.threshold(imgray,20,255,cv2.THRESH_BINARY)[1]
    im2, counters, ierarchy = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    counters.sort(key = lambda cnt: cv2.contourArea(cnt),reverse=True)
    cnt=counters[1]

    x=sorted(cnt,key= lambda pt: pt[0][1])
    y=sorted(cnt,key= lambda pt: pt[0][0])

    x_min=x[0][0]
    x_max=x[-1][0]

    y_min=y[0][0]
    y_max=y[-1][0]

    pts = np.array([y_min, x_max, y_max, x_min,y_min])

    M = cv2.getRotationMatrix2D(tuple(y_max),np.arctan((y_max[1]-x_min[1])/(y_max[0]-x_min[0]))/np.pi*180,1.0)

    pts_rot = np.matmul(M,np.array([[i[0],i[1],1.0] for i in pts]).transpose()).transpose()

    img=cv2.warpAffine(img,M,tuple(img.shape[0:-1]))

    rects=((int(min([i[0] for i in pts_rot])), int(min([i[1] for i in pts_rot]))),
                                                                        (int(max([i[0] for i in pts_rot])), int(min([i[1] for i in pts_rot]))),
                                                                        (int(max([i[0] for i in pts_rot])), int(max([i[1] for i in pts_rot]))),
                                                                        (int(min([i[0] for i in pts_rot])), int(max([i[1] for i in pts_rot]))))

    rects=np.array(rects,np.float32)
    pts_rect = np.array((pts_rot[3], pts_rot[2], pts_rot[1],pts_rot[0]),np.float32)

    M_persp = cv2.getPerspectiveTransform(pts_rect,rects)
    img=cv2.warpPerspective(img,M_persp,(img.shape[0],img.shape[1]))

    rects=np.int32(rects)
    rect=img[rects[0][1]:rects[3][1],rects[0][0]:rects[1][0]]

    return rect

img=cv2.imread("elektr.jpg")
rect=get_display_img(img)
rect_cache = np.array(rect)
rect = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)

rect = cv2.erode(rect,np.ones((3,3)))

rect=cv2.morphologyEx(rect,cv2.MORPH_BLACKHAT, np.ones((200,200)))

rect = cv2.GaussianBlur(rect,(15,15),40)
#rect = cv2.adaptiveThreshold(rect,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,2)

rect=cv2.threshold(rect,25,255,cv2.THRESH_BINARY)[1]

cnts=cv2.findContours(rect,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
cv2.drawContours(rect_cache,cnts,-1,(0,0,255))

cv2.imshow("d",rect_cache)

cv2.waitKey(0)