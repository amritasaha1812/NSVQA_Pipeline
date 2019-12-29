#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:42:43 2019

@author: amrita
"""
import pickle as pkl
import json

preprocessed_dir = '../preprocessed_data/coco'
coco_vg_object_id_map = pkl.load(open(preprocessed_dir+'/coco_vg_object_id_map.pkl','rb'), encoding='latin1')
mrcnn_result_dir = '/dccstor/cssblr/amrita/Mask_RCNN/results/coco'
year = '2017'
split = 'train'
id = 1
image_annot_dict = {}
for file in os.listdir(mrcnn_result_dir):
  if 'coco_eval_'+year+'_'+split not in file:
    continue
  data_file = os.path.join(mrcnn_result_dir, file)
  data = pkl.load(open(data_file, 'rb'), encoding='latin1')
  prediction_data = data.cocoDt.anns
  for i in prediction_data:
    di = prediction_data[i]
    bbox = di['bbox']
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    coco_object_id = di['category_id']
    vg_object_name = list(coco_vg_object_id_map[coco_object_id])
    synsets = vg_object_name
    names = ['.'.join(xi.split('.')[:-2]) for xi in synsets]
    id+=1
    d = {'names':names, 'synsets':synsets, 'x':x, 'y':y, 'h':h, 'w':w, 'object_id':id, 'ids':[id]}
    image_id = di['image_id']
    if image_id not in image_annot_dict:
        image_annot_dict[image_id] = []
    image_annot_dict[image_id].append(d)

vg_objects_data = []
for k,v in image_annot_dict.items(): 
    d = {'image_id':k, 'objects':v}
    vg_objects_data.append(d)
pkl.dump(vg_objects_data, open(preprocessed_dir+'/'+year+'/'+split+'/mrcnn_output_vg_objects.pkl','wb'))    
    
    
