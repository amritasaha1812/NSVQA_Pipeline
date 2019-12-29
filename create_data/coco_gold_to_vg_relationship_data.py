#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:37:05 2019

@author: amrita
"""

import pickle as pkl
preprocessed_dir = '../preprocessed_data/coco/'
year = '2017'
split = 'train'
vg_object_data = pkl.load(open(preprocessed_dir+'/'+year+'/'+split+'/coco_gold_vg_objects.pkl', 'rb'), encoding='latin1')

def check_bbox(subject, object):
    sub_x1 = subject['x']
    sub_x2 = subject['x'] + subject['w']
    obj_x1 = object['x']
    obj_x2 = object['x'] + object['w']
    sub_y1 = subject['y']
    sub_y2 = subject['y'] + subject['h']
    obj_y1 = object['y']
    obj_y2 = object['y'] + object['h']
    x1_union = min(sub_x1, obj_x1)
    y1_union = min(sub_y1, obj_y1)
    x2_union = max(sub_x2, obj_x2)
    y2_union = max(sub_y2, obj_y2)
    w = x2_union - x1_union
    h = y2_union - y1_union
    if w==0 or h==0:
        return False
    else:
        return True

rel_data = []
rel_id = 0
for d in vg_object_data:
    image_id = d['image_id']
    objects = d['objects']
    relationships = []
    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            subject = objects[i]
            object = objects[j]
            if not check_bbox(subject, object):
                  print ('subject object boxes not right')
                  continue
            synsets = '__background__'
            relationship_id = rel_id
            predicate = 'BG'
            d = {'subject':subject, 'object':object, 'synsets':synsets, 'relationship_id':relationship_id, 'predicate':predicate}
            rel_id += 1
            relationships.append(d)
    rel_data.append({'image_id':image_id, 'relationships':relationships})
pkl.dump(rel_data, open(preprocessed_dir+'/'+year+'/'+split+'/coco_gold_vg_relationships.pkl','wb'))            
