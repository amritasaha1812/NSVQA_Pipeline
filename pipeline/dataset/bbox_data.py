#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:58:07 2019

@author: amrita
"""
import os
import numpy as np
import pickle as pkl
from utils import utils
from scene_parsing.map_query_to_concept_vocab import MapQueryConceptsToVGConcepts
import time

class BboxData():
    
    def __init__(self, opt, bbox_detection_type, image_dir, map_query_concepts_to_vg_concepts):
        self.opt = opt
        self.sceneparser_dataset_type = opt.sceneparser_dataset_type
        self.image_dir = image_dir
        self.bbox_detection_type = self.opt.bbox_detection_type
        self.mask_rcnn_dir = opt.mask_rcnn_dir
        if self.sceneparser_dataset_type == 'gqa':
            self.split = self.opt.gqa_split
            self.type = self.opt.gqa_type
        else:
            self.split = self.opt.coco_split
            self.year = self.opt.coco_year        
        self.preprocessed_dump_dir = opt.preprocessed_dump_dir
        if self.sceneparser_dataset_type=='visual_genome' and self.opt.vqa_dataset=='vg_intersection_coco' and (self.bbox_detection_type=='gold' or self.opt.object_detection_type=='gold' or self.opt.attribute_detection_type=='gold'):
            self.vg_coco_preprocessed_dump_dir = os.path.join(self.preprocessed_dump_dir, 'vg_intersection_coco/'+self.year+'/'+self.split)
            self.gold_scene_parsed_data = pkl.load(open(os.path.join(self.vg_coco_preprocessed_dump_dir, 'scene_parsed_data_visualgenome_gold.pkl'), 'rb'), encoding='latin1')
            self.map_query_concepts_to_vg_concepts = map_query_concepts_to_vg_concepts#MapQueryConceptsToVGConcepts(opt)
            self.load_gold_bbox_data()
        elif self.sceneparser_dataset_type=='gqa' and (self.bbox_detection_type=='gold' or self.opt.object_detection_type=='gold' or self.opt.attribute_detection_type=='gold'):
            self.gqa_preprocessed_dump_dir = os.path.join(self.preprocessed_dump_dir, 'gqa/'+self.type+'/'+self.split)
            self.gqa_preprocessed_file = os.path.join(self.gqa_preprocessed_dump_dir, 'scene_parsed_data_gqa_gold.pkl')
            self.gold_scene_parsed_data = pkl.load(open(self.gqa_preprocessed_file, 'rb'))
            self.map_query_concepts_to_vg_concepts = map_query_concepts_to_vg_concepts#MapQueryConceptsToVGConcepts(opt)
            self.load_gold_bbox_data()
        else:
            if self.sceneparser_dataset_type=='gqa':
                 self.load_gqa_bbox_data()
            else:
                 self.load_coco_bbox_data()
        
    def load_coco_bbox_data(self):
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


    def load_gold_bbox_data(self):
        self.bbox_data = {}
        for image_id in self.gold_scene_parsed_data:
            self.bbox_data[image_id] = self.load_gold_bbox_data_for_image(image_id)

    def load_gold_bbox_data_for_image(self, image_id):
        bboxes = []
        for bbox, value in self.gold_scene_parsed_data[image_id]['object_descriptors'].items():
            bbox = [int(x) for x in bbox.split(' ')]
            score = None
            segmentation = None
            area = bbox[-1]*bbox[-2]
            category_id = None
            if self.opt.object_detection_type=='gold':
                objects = set([])
            if self.opt.attribute_detection_type=='gold':
                attributes = set([])
            if self.opt.object_detection_type=='gold' or self.opt.attribute_detection_type=='gold':
                for obj_id, obj in value['synsets'].items():
                    if 'synsets' in obj:
                        object = obj['synsets']
                        objects.update(object)
                    if 'attributes' in obj:
                        attribute = obj['attributes']
                        attributes.update(attribute)
            if self.opt.object_detection_type=='gold' or self.opt.attribute_detection_type=='gold':
                objects = set([x.lower().strip() for x in objects])
                attributes = set([x.lower().strip() for x in attributes])
                object_distribution, attribute_distribution, objects, attributes = self.map_query_concepts_to_vg_concepts.map_concepts_to_vg_concepts(objects, attributes, None, one_hot=True, binarize_concept_map=True)
                object_distribution = utils.normalize(np.sum(np.concatenate(list(object_distribution.values()), axis=0), axis=0), 1)
                attribute_distribution = utils.normalize(np.sum(np.concatenate(list(attribute_distribution.values()), axis=0), axis=0), 1)
            d = {}
            d['score'] = score
            d['area'] = area
            d['segmentation'] = segmentation
            d['bbox'] = bbox
            d['image_id'] = image_id
            d['category_id'] = category_id
            if self.opt.object_detection_type=='gold':
                d['object_distribution'] = object_distribution
                d['objects'] = objects
            if self.opt.attribute_detection_type=='gold':
                d['attribute_distribution'] = attribute_distribution
                d['attributes'] = attributes
            bboxes.append(d)
        return bboxes
   
    def __getitem__(self, image_id):
        if image_id in self.bbox_data:
            bbox_data = self.bbox_data[image_id]
        else:
            print ('Cannot find ', image_id, ' in mask rcnn')
            bbox_data = None
        return bbox_data    
