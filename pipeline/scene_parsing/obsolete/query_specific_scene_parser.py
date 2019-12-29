#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:52:36 2019

@author: amrita
"""
import json 
import PIL.Image as Image
import cv2
import numpy as np
import math
from .query_cleaning import QueryCleaning
from .query_concept_extractor import QueryConceptExtractor
from .map_query_to_concept_vocab import MapQueryConceptsToVGConcepts
import pickle as pkl
from dataset.vqa_attribute_dataset import VQAAttributeDataset
from dataset.vqa_object_dataset import VQAObjectDataset
from catalog.vqa_attribute_catalog import VQAAttributeCatalog
from catalog.vqa_object_catalog import VQAObjectCatalog
from options import get_options
import os
from torch.utils.data import DataLoader
from dataset.vqa_dataset import VQADataset
from dataset.bbox_data import BboxData
from utils.vqa_utils import VQAUtils
from dataset.vg_coco_intersection_dataset import VGCOCOIntersectionDataset

class QuerySpecificSceneParser():
    
    def __init__(self, opt, split, year):
        self.opt = opt
        self.vqa_dir = opt.vqa_dir
        self.coco_dir = opt.coco_dir
        self.split = split
        self.year = year
        self.vqa_preprocessed_dump_dir = os.path.join(opt.preprocessed_dump_dir, 'vqa')
        if not os.path.exists(self.vqa_preprocessed_dump_dir):
            os.mkdir(self.vqa_preprocessed_dump_dir)
        self.image_dir = os.path.join(self.coco_dir, self.split+self.year)
        self.mask_rcnn_dir = opt.mask_rcnn_dir
        self.query_cleaning = QueryCleaning()
        self.map_query_concepts_to_vg_concepts = MapQueryConceptsToVGConcepts(opt)
        self.query_concept_extractor = QueryConceptExtractor(opt)
        self.vqa_utils = VQAUtils(self.image_dir, self.split, self.year)
        self.load()
            
    '''
    def get_bbox_data(self, image_id):
        print ('searching for ', image_id)
        if image_id in self.bbox_data:
            bbox_data = self.bbox_data[image_id]
        else:
            print ('1. Cannot find ', image_id, ' in mask rcnn')
            bbox_data = None
        return bbox_data
    '''

    def get_query_concepts_glove_embedding(self, query):
        query_parsed, query_toks, query_pos, query_lemma, query_tok_descendants = self.query_cleaning.execute(query)
        query_objects, query_attributes, query_relations = self.query_concept_extractor.execute(query_lemma, query_pos, query_tok_descendants)
        q_objects_emb, q_attrs_emb = self.map_query_concepts_to_vg_concepts.get_query_concepts_glove_emb(query_objects, query_attributes, query_relations)
        q_concepts_emb = {}
        q_concepts_emb.update(q_objects_emb)
        q_concepts_emb.update(q_attrs_emb)
        return list(q_concepts_emb.values())
    
    def get_query_specific_scene_parsing_data(self, vqa_data_instance):
        question_type = vqa_data_instance['question_type']
        question = vqa_data_instance['question']
        answers = vqa_data_instance['answers']
        image_id = vqa_data_instance['image_id']
        question_index = vqa_data_instance['question_index']
        bbox_data = self.bbox_data[image_id]#get_bbox_data(image_id)
        processed_data = []
        if not bbox_data:
            return processed_data
        for bbox_index,region in enumerate(bbox_data):
            score = region['score']
            bbox = region['bbox']
            image_region = self.vqa_utils.get_image_region(image_id, bbox)
            query_concepts_emb = np.asarray(self.get_query_concepts_glove_embedding(question))
            processed_data.append({'image_id':image_id, 'bbox':bbox, 'bbox_score':score, 'bbox_index':bbox_index, 'question_index':question_index, 'image_region':image_region, 'query_concepts_emb':query_concepts_emb})
        return processed_data
    
    def preprocess_vqa_data(self):
        self.processed_vqa_dataset = []
        for i,vqa_data_instance in enumerate(self.vqa_dataset):
            if not vqa_data_instance:
                break
            print ('processing ', i,'\'th data', type(vqa_data_instance), vqa_data_instance)
            processed_data = self.get_query_specific_scene_parsing_data(vqa_data_instance)
            self.processed_vqa_dataset.extend(processed_data)
            
        
    def load(self):
        if self.opt.vqa_dataset=='coco':
            self.vqa_dataset = VQADataset(self.opt, self.split, self.year)
        elif self.opt.vqa_dataset=='vg_intersection_coco':
            self.vqa_dataset = VGCOCOIntersectionDataset(self.opt, self.split, self.year)
        self.bbox_data = BboxData(self.opt, self.split, self.year, self.opt.bbox_detection_type, self.image_dir)
        if self.opt.vqa_dataset=="coco":
            self.preprocess_vqa_data()    
        if self.opt.object_detection_type=='catalog':
            self.vqa_object_catalog = VQAObjectCatalog(self.opt, self.processed_vqa_dataset, self.vqa_dataset)
        if self.opt.attribute_detection_type=='catalog':
            self.vqa_attribute_catalog = VQAAttributeCatalog(self.opt, self.processed_vqa_dataset, self.vqa_dataset)

    
    def execute(self): 
        #if self.opt.vqa_dataset=='coco' or self.opt.bbox_detection_type=='mask_rcnn':         
        self.add_bbox_in_vqa_data() 
        if self.opt.object_detection_type=='catalog':   
            self.vqa_object_catalog.execute()
        if self.opt.attribute_detection_type=='catalog':    
            self.vqa_attribute_catalog.execute()
        self.vqa_dataset.dump(os.path.join(self.vqa_preprocessed_dump_dir, 'v2_mscoco_'+self.split+self.year+'_annotations.pkl'))    
    
    def add_bbox_in_vqa_data(self):
        for i,vqa_data_instance in enumerate(self.vqa_dataset):
            if not vqa_data_instance:
                break
            print ('fetched image id ', vqa_data_instance['image_id'])
            vqa_data_instance['bbox'] = self.bbox_data[vqa_data_instance['image_id']]#self.get_bbox_data(vqa_data_instance['image_id'])
            self.vqa_dataset[i] = vqa_data_instance    
            
  
if __name__== '__main__':
    opt = get_options()
    split = 'train'
    year = '2014'
    query_specific_scene_parser = QuerySpecificSceneParser(opt, split, year)
    query_specific_scene_parser.execute()
    

