import numpy as np
import dlib
from multiprocessing import Process, Queue
import cv2

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def face_detect(q,gray,detector):

    q.put(detector(gray,0))


predictor_path='shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)

colors=[(0,0,255),
        (0,255,0),
        (255,0,0),
        (0,255,255),
        (255,0,255),
        (255,255,0),
        (0,0,0),
        (255,255,255),
        (180,105,255)]

kk=0
n=1
import time as t

q = Queue()

t_0=t.time()

while True:

    t_begin = t.time()
    flag, frame = cap.read()

    if flag:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if kk == 0:

            q.put(detector(gray, 0))
            while q.empty():
                pass

        if not q.empty():
            rects=q.get_nowait()
            p=Process(target=face_detect,args=(q,gray,detector))
            p.start()

        for rect in rects:

            cv2.rectangle(frame,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(0,0,255))
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            ################### LEFT EYE ####################################################

            cache=np.array(shape[36:42])

            # for x,y in cache:
            #     cv2.circle(frame,(x,y),1,(0,0,255))

            x_min=min([i[0] for i in cache])
            y_min=min([i[1] for i in cache])
            x_max=max([i[0] for i in cache])
            y_max=max([i[1] for i in cache])

            weight=x_max-x_min
            height=y_max-y_min

            r=0.5


            eye_1=np.array(frame[int(y_min-r*height):int(y_max+r*height),int(x_min-r*weight):int(x_max+r*weight)])

            for i in range(len(cache)):
                cache[i]=(cache[i][0]-int(x_min-r*weight),cache[i][1]-int(y_min-r*height))

            try:
                eye_1_gray=cv2.cvtColor(eye_1,cv2.COLOR_BGR2GRAY)
            except:
                continue

            img = cv2.threshold(eye_1_gray, 100, 255, cv2.THRESH_BINARY)[1]

            mask = np.zeros((eye_1.shape[0],eye_1.shape[1],1),np.uint8)

            cv2.fillPoly(mask, [np.array(cache)], 1)
            img = cv2.bitwise_and(img, img, mask=mask,dst=np.array([[[127] for __ in range(eye_1.shape[1])] for _ in range(eye_1.shape[0])],np.uint8))

            cX_list=[]
            cY_list=[]

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j]==0:
                        cX_list.append(j)
                        cY_list.append(i)
            if cX_list!=[]:
                cX_1=int(np.mean(cX_list))
                cY_1=int(np.mean(cY_list))

            try:
                cX_1_glob=cX_1 + int(x_min-r*weight)
                cY_1_glob=cY_1 + int(y_min-r*height)
            except:
                continue

            ################### RIGHT EYE ####################################################

            cache = shape[42:48]

            x_min = min([i[0] for i in cache])
            y_min = min([i[1] for i in cache])
            x_max = max([i[0] for i in cache])
            y_max = max([i[1] for i in cache])

            weight = x_max - x_min
            height = y_max - y_min

            eye_1 = np.array(frame[int(y_min - r * height):int(y_max + r * height),
                             int(x_min - r * weight):int(x_max + r * weight)])

            for i in range(len(cache)):
                cache[i] = (cache[i][0] - int(x_min - r * weight), cache[i][1] - int(y_min - r * height))

            try:
                eye_1_gray = cv2.cvtColor(eye_1, cv2.COLOR_BGR2GRAY)
                eye_1_hsv = cv2.cvtColor(eye_1,cv2.COLOR_BGR2HSV)
                cv2.imshow("kf",eye_1_hsv)
                cv2.waitKey(0)
            except:
                continue

            img = cv2.threshold(eye_1_gray, 100, 255, cv2.THRESH_BINARY)[1]

            mask = np.zeros((eye_1.shape[0], eye_1.shape[1], 1), np.uint8)

            cv2.fillPoly(mask, [np.array(cache)], 1)
            img = cv2.bitwise_and(img, img, mask=mask, dst=np.array(
                [[[127] for __ in range(eye_1.shape[1])] for _ in range(eye_1.shape[0])], np.uint8))

            cX_list = []
            cY_list = []

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j] == 0:
                        cX_list.append(j)
                        cY_list.append(i)
            if cX_list !=[]:
                cX_2 = int(np.mean(cX_list))
                cY_2 = int(np.mean(cY_list))

            try:
                cX_2_glob = cX_2 + int(x_min - r * weight)
                cY_2_glob = cY_2 + int(y_min - r * height)
            except:
                continue

            ############# PLOT PUPIL ###########################

            cv2.circle(frame,(cX_1_glob,cY_1_glob),1,(0,255,0))
            cv2.circle(frame,(cX_2_glob,cY_2_glob),1,(0,255,0))

        kk+=1

        cv2.imshow("Frame", frame)
        delay=int(40-1000*(t.time()-t_begin))

        if delay<=0:
            key = cv2.waitKey(1) & 0xFF
        else:
            key = cv2.waitKey(delay) & 0xFF
        # #
        # # # if the `q` key was pressed, break from the loop


        if key == ord("q"):
          break
cap.release()
cv2.destroyAllWindows()