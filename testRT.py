#%%
from cv2 import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import numpy as np
from testTF import *
import time
frame_rate = 10
prev = 0

# def get_sample(src):
#     a = []
#     for i in range(50):
#         a.append(src)
#     return np.array(a)

cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FPS, 10)
# fps = int(cap.get(5))
# print("fps:", fps)
i = 0
a = []
while(True):
    time_elapsed = time.time() - prev
    res, frame = cap.read()

    if time_elapsed > 1./frame_rate:
        prev = time.time()
        # fps = cap.get(cv.CAP_PROP_FPS)
        # print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        frame = cv.resize(frame,(257,257))
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        template_kps = keypoint_detect(frame)
        frame_draw = draw_kps(frame,template_kps)
        # print(template_kps)
        cv.imshow('10frame',frame_draw)
        # if (template_kps[5:11,2] == np.ones(6)).all:
        try:
            if template_kps[5:11,2].all:
                a.append(template_kps)
                i = i+1
                # print("kuy")
            for j,k in enumerate(template_kps[5:11]):
                    if k[2] != 1:
                        template_kps[j] = a[-1][j]
            print("Found")
                # a.append(template_kps)
        except:
            # template_kps = keypoint_detect(frame)
            # frame_draw = draw_kps(frame,template_kps)
            print("Notfound Arm")
        # if (template_kps[5:11,2] != np.ones(6)).all:
            # a
        if i == 50:
            np.save("data.npy",a)
            a = a[-1]
            i = 0
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()



# %%
