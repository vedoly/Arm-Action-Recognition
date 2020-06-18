#%%
from cv2 import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import numpy as np
from testTF import *



cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv.resize(frame,(257,257))
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    template_kps = keypoint_detect(frame)
    frame_draw = draw_kps(frame,template_kps)
    # Display the resulting frame
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()



# %%
