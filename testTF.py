#%%
from cv2 import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import numpy as np

def parse_output(heatmap_data,offset_data, threshold):

  '''
  Input:
    heatmap_data - hetmaps for an image. Three dimension array
    offset_data - offset vectors for an image. Three dimension array
    threshold - probability threshold for the keypoints. Scalar value
  Output:
    array with coordinates of the keypoints and flags for those that have
    low probability
  '''

  joint_num = heatmap_data.shape[-1]
  pose_kps = np.zeros((joint_num,3), np.uint32)

  for i in range(heatmap_data.shape[-1]):

      joint_heatmap = heatmap_data[...,i]
      max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
      remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
      pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
      pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
      max_prob = np.max(joint_heatmap)

      if max_prob > threshold:
        if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
          pose_kps[i,2] = 1

  return pose_kps

def draw_kps(show_img,kps, ratio=None):
    for i in range(0,kps.shape[0]):
      if kps[i,2]:
        if isinstance(ratio, tuple):
          cv.circle(show_img,(int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0]))),2,(0,255,255),round(int(1*ratio[1])))
          continue
        cv.circle(show_img,(kps[i,1],kps[i,0]),2,(0,255,255),-1)
    return show_img

def keypoint_detect(img_ori):
  # img_ori = cv.imread('image_2.jpg')
  img_ori = cv.resize(img_ori, (257,257))
  img_ori = img_ori.reshape(1,257,257,3)
  # img = tf.reshape(tf.image.resize(img, [257,257]), [1,257,257,3])
  model = tf.lite.Interpreter('tracking_model.tflite')
  model.allocate_tensors()
  input_details = model.get_input_details()
  output_details = model.get_output_details()
  floating_model = input_details[0]['dtype'] == np.float32

  if floating_model:
      img = (np.float32(img_ori) - 127.5) / 127.5

  model.set_tensor(input_details[0]['index'], img)
  model.invoke()

  # output_data =  model.get_tensor(output_details[0]['index'])
  # offset_data = model.get_tensor(output_details[1]['index'])
  # print('output: {}'.format(output_data))
  
  # Extract output data from the interpreter
  template_output_data = model.get_tensor(output_details[0]['index'])
  template_offset_data = model.get_tensor(output_details[1]['index'])
  # Getting rid of the extra dimension
  template_heatmaps = np.squeeze(template_output_data)
  template_offsets = np.squeeze(template_offset_data)

  #template_kps[2] is bool for that node is exist
  template_kps = parse_output(template_heatmaps,template_offsets,0.3)

  # cv.imshow("x",draw_kps(img_ori[0,:,:,:].copy(),template_kps))
  # cv.waitKey(0)
  return template_kps

# %%
