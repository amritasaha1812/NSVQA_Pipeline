#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:58:07 2019

@author: amrita
"""
import os
import numpy as np
import pickle as pkl
from utils.vqa_utils import VQAUtils
from scene_parsing.map_query_to_concept_vocab import MapQueryConceptsToVGConcepts
import utils.utils

class BboxData():
    
    def __init__(self, opt, split, year, bbox_detection_type, image_dir):
        self.opt = opt
        self.image_dir = image_dir
        self.bbox_detection_type = self.opt.bbox_detection_type
        self.mask_rcnn_dir = opt.mask_rcnn_dir
        self.split = split
        self.year = year
        self.vqa_utils = VQAUtils(self.image_dir, self.split, self.year)
        if self.opt.vqa_dataset=='vg_intersection_coco' and (self.bbox_detection_type=='gold' or self.opt.object_detection_type=='gold' or self.opt.attribute_detection_type=='gold'):
            self.preprocessed_dump_dir = opt.preprocessed_dump_dir
            self.vg_coco_preprocessed_dump_dir = os.path.join(self.preprocessed_dump_dir, 'vg_intersection_coco/'+self.year+'/'+self.split)
            self.gold_scene_parsed_data = pkl.load(open(os.path.join(self.vg_coco_preprocessed_dump_dir, 'scene_parsed_data_visualgenome_gold.pkl'), 'rb'), encoding='latin1')
            self.map_query_concepts_to_vg_concepts = MapQueryConceptsToVGConcepts(opt)
            self.load_gold_vg_bbox_data()
        else:
            self.load_bbox_data()
        
    def load_bbox_data(self):
        self.bbox_data = {}
        for file in os.listdir(os.path.join(self.mask_rcnn_dir, 'results/coco/')):
            if not 'coco_eval_'+self.year+'_'+self.split in file or not file.endswith('pkl'):
                continue
            file = os.path.join(os.path.join(self.mask_rcnn_dir, 'results/coco/'), file)
            bbox_data = pkl.load(open(file, 'rb'), encoding='latin1')
            if self.bbox_detection_type == 'mask_rcnn':
                bbox_data = bbox_data.cocoDt.anns.values()
            elif self.bbox_detection_type == 'gold':
                bbox_data = bbox_data.cocoGt.anns.values()
            for d in bbox_data:
                score = d['score']
                area = d['area']
                segmentation = d['segmentation']
                bbox = d['bbox']
                image_id = d['image_id']
                category_id = d['category_id']
                if image_id not in self.bbox_data:
                    self.bbox_data[image_id] = []
                self.bbox_data[image_id].append({'score':score, 'area':area, 'segmentation':segmentation, 'bbox':bbox, 'image_id':image_id, 'category_id':category_id}) 


    def load_gold_vg_bbox_data(self):
        self.bbox_data = {}
        for image_id in self.gold_scene_parsed_data:
            self.bbox_data[image_id] = self.load_gold_vg_bbox_data_for_image(image_id)

    def load_gold_vg_bbox_data_for_image(self, image_id):
        bboxes = []
        for bbox, value in self.gold_scene_parsed_data[image_id]['object_descriptors'].items():
            bbox = [int(x) for x in bbox.split(' ')]
            score = None
            segmentation = None
            area = bbox[-1]*bbox[-2]
            category_id = None
            if self.opt.object_detection_type=='gold':
                objects = []
            if self.opt.attribute_detection_type=='gold':
                attributes = []
            if self.opt.object_detection_type=='gold' or self.opt.attribute_detection_type=='gold':
                for obj_id, obj in value['synsets'].items():
                    if 'synsets' in obj:
                        object = obj['synsets']
                        objects.extend(object)
                    if 'attributes' in obj:
                        attribute = obj['attributes']
                        attributes.extend(attribute)
            print ('objects ', objects, 'attributes ', attributes)
            if self.opt.object_detection_type=='gold' or self.opt.attribute_detection_type=='gold':
                objects, attributes, _ = self.map_query_concepts_to_vg_concepts.map_concepts_to_vg_concepts(objects, attributes, None, one_hot=False)
                object_distribution = list(objects.values()) #np.sum(np.asarray(list(objects.values())), axis=0)
                attribute_distribution = list(attributes.values()) #np.sum(np.asarray(list(attributes.values())), axis=0)
            #if not overall_flag:
            #   print ('Bbox data ', value, '\n')
            d = {}
            d['score'] = score
            d['area'] = area
            d['segmentation'] = segmentation
            d['bbox'] = bbox
            d['image_id'] = image_id
            d['category_id'] = category_id
            if self.opt.object_detection_type=='gold':
                d['object_distribution'] = object_distribution
            if self.opt.attribute_detection_type=='gold':
                d['attribute_distribution'] = attribute_distribution
            bboxes.append(d)
        return bboxes
    
    def __getitem__(self, image_id):
        if image_id in self.bbox_data:
            bbox_data = self.bbox_data[image_id]
        else:
            print ('Cannot find ', image_id, ' in mask rcnn')
            bbox_data = None
        return bbox_data    
