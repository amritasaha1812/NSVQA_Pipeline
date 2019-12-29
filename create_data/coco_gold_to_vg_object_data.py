#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:40:59 2019

@author: amrita
"""
import json 
import pickle as pkl

coco_dir = '/dccstor/cssblr/amrita/coco/'
year = '2017'
split = 'train'
preprocessed_dir = '../preprocessed_data/coco/'+year+'/'+split
coco_vg_object_id_map = pkl.load(open('../preprocessed_data/coco/coco_vg_object_id_map.pkl','rb'), encoding='latin1')
coco_gold_data = json.load(open(coco_dir+'/annotations/instances_'+split+year+'.json'))
annotations = coco_gold_data['annotations']
image_annot_dict = {}
id = 1
for di in annotations:
    image_id = di['image_id']
    bbox = di['bbox']
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    coco_object_id = di['category_id']
    vg_object_name = list(coco_vg_object_id_map[coco_object_id])
    synsets = vg_object_name
    names = ['.'.join(xi.split('.')[:-2]) for xi in synsets]
    d = {'names':names, 'synsets':synsets, 'x':x, 'y':y, 'h':h, 'w':w, 'object_id':id, 'ids':[id]}
    id += 1
    if image_id not in image_annot_dict:
        image_annot_dict[image_id] = []
    image_annot_dict[image_id].append(d) 
    

vg_objects_data = []
for k,v in image_annot_dict.items(): 
    d = {'image_id':k, 'objects':v}
    vg_objects_data.append(d)
pkl.dump(vg_objects_data, open(preprocessed_dir+'/coco_gold_vg_objects.pkl','wb'))    
    
    
    
        
      
    
