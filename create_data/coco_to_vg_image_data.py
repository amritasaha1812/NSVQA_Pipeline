#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:23:13 2019

@author: amrita
"""
import os
import cv2
import json

version = '2017'
split = 'val'
vqa_images_dir = '/dccstor/cssblr/amrita/Mask_RCNN/data/coco/'+split+version
preprocessed_dir = '../preprocessed_data/coco/'+version+'/'+split
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)
vqa_images_data = []
for file in os.listdir(vqa_images_dir):
    image_id = file.split('.')[0]#int(file.split('.')[0].split('_')[-1])
    file = vqa_images_dir+'/'+file
    img = cv2.imread(file)
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    coco_id = image_id
    flickr_id = None
    url = None
    d= {'url':url, 'image_id':image_id, 'coco_id':coco_id, 'flickr_id':flickr_id, 'height':height, 'width':width}
    vqa_images_data.append(d)
    if len(vqa_images_data)%1000==0:
         print (len(vqa_images_data))
json.dump(vqa_images_data, open(preprocessed_dir+'/image_data.json', 'w'), indent=1)    
    
