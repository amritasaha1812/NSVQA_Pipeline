#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:18:27 2019

@author: amrita
"""
import os
import cv2
import json

class COCODataset():
    def __init__(self, opt):
        self.split = opt.coco_split
        self.year = opt.coco_year
        self.coco_dir = opt.coco_dir
        self.coco_images_dir = os.path.join(self.coco_dir, self.split+self.year)
        self.coco_preprocessed_dump_dir = os.path.join(opt.preprocessed_dump_dir, 'coco'+'/'+self.year+'/'+self.split)
        if not os.path.exists(self.coco_preprocessed_dump_dir):
            os.mkdir(self.coco_preprocessed_dump_dir)
        if not os.path.exists(os.path.join(self.coco_preprocessed_dump_dir, 'image_data.json')):
            self.coco_images_data = self.dump_image_data()
        else:
            self.coco_images_data = json.load(open(os.path.join(self.coco_preprocessed_dump_dir, 'image_data.json')))
            
    def dump_image_data(self):
        coco_images_data = []
        print (self.coco_images_dir)
        for file in os.listdir(self.coco_images_dir):
            image_id = file.split('.')[0]#int(file.split('.')[0].split('_')[-1])
            file = os.path.join(self.coco_images_dir, file)
            img = cv2.imread(file)
            height = img.shape[0]
            width = img.shape[1]
            channels = img.shape[2]
            coco_id = image_id
            flickr_id = None
            url = None
            d= {'url':url, 'image_id':image_id, 'coco_id':coco_id, 'flickr_id':flickr_id, 'height':height, 'width':width}
            coco_images_data.append(d)
            if len(coco_images_data)%1000==0:
                 print (len(coco_images_data))
        json.dump(coco_images_data, open(os.path.join(self.coco_preprocessed_dump_dir,'image_data.json'), 'w'), indent=1)
        return coco_images_data
