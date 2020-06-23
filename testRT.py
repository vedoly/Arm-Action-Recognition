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
y = 0
model = tf.keras.models.load_model("./Conv1d_Result/Conv1d_handwave_angle.h5")
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
        
        # if (template_kps[5:11,2] == np.ones(6)).all:
        try:
            if template_kps[5:11,2].sum() == 6:
                a.append(template_kps)
                
                # print("kuy")
            for j,k in enumerate(template_kps[5:11]):
                    if k[2] != 1:
                        template_kps[j] = a[-1][j]
            i = i+1
            print("Found")
                # a.append(template_kps)
        except:
            # template_kps = keypoint_detect(frame)
            # frame_draw = draw_kps(frame,template_kps)
            print("Notfound Arm")
        # if (template_kps[5:11,2] != np.ones(6)).all:
            # a
        print(i)
        if i == 50:
            np.save("data.npy",a)
            j = []
            for h in a:
                m = []
                for l in h[5:11]:
                    m.append(l[:2])
                j.append(m)
            j = np.array(j)
            X_angle = []
            Y_angle = []
            AngleA = np.zeros((50,4))
            for i in range(len(j)):
                c = j[i]
                AngleA[i][0] = (findAngleR(c[3],c[4],c[5]))
                AngleA[i][1] = (findAngleR(c[4],c[3],c[0]))
                AngleA[i][2] = (findAngleL(c[1],c[0],c[3]))
                AngleA[i][3] = (findAngleL(c[0],c[1],c[2]))
            X_angle.append(AngleA)
            X_angle = np.array(X_angle)
            score = model.predict(X_angle)
            print(score)
            # print("kuy")
            a = a[-1]
            a = a.reshape(1,17,3)
            a = list(a)
            i = 0
    cv.imshow('10frame',frame_draw)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()



# %%
