#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:52:36 2019

@author: amrita
"""
import argparse
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
from options import get_options, get_option_str, get_options_parser
import os
from torch.utils.data import DataLoader
from dataset.vqa_dataset import VQADataset
from dataset.gqa_dataset import GQADataset
from dataset.bbox_data import BboxData
from utils.vqa_utils import VQAUtils
from utils.gqa_utils import GQAUtils
from dataset.vg_coco_intersection_dataset import VGCOCOIntersectionDataset
import torch


class QuerySpecificSceneParser():
    
    def __init__(self, opt):
        self.opt = opt
        self.sceneparser_dataset_type = opt.sceneparser_dataset_type
        if self.sceneparser_dataset_type == 'gqa':
            self.gqa_dir = opt.gqa_dir
            self.split = opt.gqa_split
            self.type = opt.gqa_type
            self.gqa_preprocessed_dump_dir = os.path.join(os.path.join(opt.preprocessed_dump_dir, 'gqa'), get_option_str(opt))
            self.image_dir = os.path.join(self.gqa_dir, 'data/images')
            self.gqa_utils = GQAUtils(self.image_dir, self.split, self.type)
            if not os.path.exists(self.gqa_preprocessed_dump_dir):
               os.mkdir(self.gqa_preprocessed_dump_dir)
        else:
            self.vqa_dir = opt.vqa_dir
            self.coco_dir = opt.coco_dir
            self.split = opt.coco_split
            self.year = opt.coco_year
            self.vqa_preprocessed_dump_dir = os.path.join(os.path.join(opt.preprocessed_dump_dir, 'vqa'), get_option_str(opt))
            self.image_dir = os.path.join(self.coco_dir, self.split+self.year)
            self.vqa_utils = VQAUtils(self.image_dir, self.split, self.year)
            if not os.path.exists(self.vqa_preprocessed_dump_dir):
               os.mkdir(self.vqa_preprocessed_dump_dir)
        self.preprocessed_annotation_file = self.opt.preprocessed_annotation_file
        if opt.num_splits>1:
            self.preprocessed_annotation_file = self.preprocessed_annotation_file.replace('.pkl', '_'+str(opt.split_number)+'.pkl')
        self.mask_rcnn_dir = opt.mask_rcnn_dir
        self.vg_mapping_in_query = bool(opt.vg_mapping_in_query)
        self.query_cleaning = QueryCleaning()
        self.map_query_concepts_to_vg_concepts = MapQueryConceptsToVGConcepts(opt)
        self.query_concept_extractor = QueryConceptExtractor(opt)
        self.load()
            
   

    def get_query_concepts(self, query):
        query_parsed, query_toks, query_pos, query_lemma, query_tok_descendants = self.query_cleaning.execute(query)
        query_objects, query_attributes, query_relations = self.query_concept_extractor.execute(query_toks, query_lemma, query_pos, query_tok_descendants)
        q_objects_emb, q_attrs_emb, q_rels_emb = self.map_query_concepts_to_vg_concepts.get_query_concepts_glove_emb(query_objects, query_attributes, query_relations)
        if self.vg_mapping_in_query:
            q_concepts_to_objects_map, q_concepts_to_attrs_map, q_concepts_to_rels_map, q_objects, q_attrs, q_rels = self.map_query_concepts_to_vg_concepts.map_concepts_to_vg_concepts(query_objects, query_attributes, query_relations)
        else:
            q_objects, q_attrs, q_rels = self.map_query_concepts_to_vg_concepts.map_concepts_to_vg_concepts(query_objects, query_attributes, query_relations, get_map=False)
            q_concepts_to_objects_map = {x:i for i,x in enumerate(query_objects)}
            q_concepts_to_attrs_map = {x:i for i,x in enumerate(query_attributes)}
            q_concepts_to_rels_map = {x:i for i,x in enumerate(query_relations)}
        return q_objects_emb, q_attrs_emb, q_objects, q_attrs, q_rels, q_concepts_to_objects_map, q_concepts_to_attrs_map, q_concepts_to_rels_map
    
    def get_query_specific_scene_parsing_data(self, vqa_data_instance):
        question_type = vqa_data_instance['question_type']
        question = vqa_data_instance['question']
        answers = vqa_data_instance['answers']
        image_id = vqa_data_instance['image_id']
        question_index = vqa_data_instance['question_index']
        if question_index not in self.processed_vqa_queries:
            query_objects_emb, query_attrs_emb, query_objects, query_attrs, query_rels, query_to_vg_objects_map, query_to_vg_attrs_map, query_to_vg_rels_map = self.get_query_concepts(question)
            processed_query = {'question_index':question_index, 'query_objects_emb':query_objects_emb, 'query_attributes_emb':query_attrs_emb, 'query_objects':query_objects, 'query_attributes':query_attrs, 'query_relations':query_rels, 'query_to_vg_objects_map':query_to_vg_objects_map, 'query_to_vg_attrs_map':query_to_vg_attrs_map, 'query_to_vg_rels_map':query_to_vg_rels_map}
        else:
            processed_query = None
        bbox_data = self.bbox_data[image_id]#get_bbox_data(image_id)
        processed_bbox_data = []
        if not bbox_data:
            return processed_query, processed_bbox_data
        for bbox_index,region in enumerate(bbox_data):
            score = region['score']
            bbox = region['bbox']
            image_region = self.gqa_utils.get_image_region(image_id, bbox)
            processed_bbox_data.append({'image_id':image_id, 'bbox':bbox, 'bbox_score':score, 'bbox_index':bbox_index, 'image_region':image_region, 'question_index':question_index})
        return processed_query, processed_bbox_data
    
    def preprocess_vqa_data(self):
        self.processed_vqa_dataset = []
        self.processed_vqa_queries = {}
        i=0
        for vqa_data_instance in torch.utils.data.DataLoader(self.vqa_dataset, batch_size=None, batch_sampler=None):
            if not vqa_data_instance:
                break
            if i%100==0:
                print ('processing ', i,'\'th vqa data')
            processed_query, processed_bbox_data = self.get_query_specific_scene_parsing_data(vqa_data_instance)
            self.processed_vqa_dataset.extend(processed_bbox_data)
            if processed_query:
                self.processed_vqa_queries[str(vqa_data_instance['image_id'])+'_'+str(processed_query['question_index'])] = processed_query
            i+=1

    def load(self):
        if self.sceneparser_dataset_type=='gqa':
            self.vqa_dataset = GQADataset(self.opt)
        else:
            if self.opt.vqa_dataset=='coco':
                self.vqa_dataset = VQADataset(self.opt)
            elif self.opt.vqa_dataset=='vg_intersection_coco':
                self.vqa_dataset = VGCOCOIntersectionDataset(self.opt)
        self.bbox_data = BboxData(self.opt, self.opt.bbox_detection_type, self.image_dir, self.map_query_concepts_to_vg_concepts)
        self.preprocess_vqa_data()    
        if self.opt.object_detection_type=='catalog':
            self.vqa_object_catalog = VQAObjectCatalog(self.opt, self.processed_vqa_dataset, self.processed_vqa_queries, self.vqa_dataset)
        if self.opt.attribute_detection_type=='catalog':
            self.vqa_attribute_catalog = VQAAttributeCatalog(self.opt, self.processed_vqa_dataset, self.processed_vqa_queries, self.vqa_dataset)

    
    def execute(self): 
        #if self.opt.vqa_dataset=='coco' or self.opt.bbox_detection_type=='mask_rcnn':         
        self.add_bbox_in_vqa_data() 
        if self.opt.object_detection_type=='catalog':   
            self.vqa_object_catalog.execute()
        if self.opt.attribute_detection_type=='catalog':    
            self.vqa_attribute_catalog.execute()
        if self.sceneparser_dataset_type=='gqa':
           self.vqa_dataset.dump(os.path.join(self.gqa_preprocessed_dump_dir, self.preprocessed_annotation_file)) 
        else:
           self.vqa_dataset.dump(os.path.join(self.vqa_preprocessed_dump_dir, self.preprocessed_annotation_file))    
    
    def add_bbox_in_vqa_data(self):
        i=0
        for vqa_data_instance in torch.utils.data.DataLoader(self.vqa_dataset, batch_size=None, batch_sampler=None):
            if not vqa_data_instance:
                break
            #print ('fetched image id ', vqa_data_instance['image_id'])
            vqa_data_instance['bbox'] = self.bbox_data[vqa_data_instance['image_id']]#self.get_bbox_data(vqa_data_instance['image_id'])
            question_index = vqa_data_instance['question_index']
            vqa_data_instance.update(self.processed_vqa_queries[str(vqa_data_instance['image_id'])+'_'+str(question_index)])
            self.vqa_dataset[i] = vqa_data_instance    
            if i%1000==0:
               print ('added bounding box info for ', i, 'th data instance')
            
            i+=1
            
  
if __name__== '__main__':
    opt_parser = get_options_parser()
    opt_parser.add_argument('--split_number', default=0, type=int)
    opt_parser.add_argument('--num_splits', default=1, type=int)
    opt = opt_parser.parse_args()
    args = vars(opt)
    for k, v in args.items():
        print ('%s: %s' % (str(k), str(v)))
    query_specific_scene_parser = QuerySpecificSceneParser(opt)
    query_specific_scene_parser.execute()
    


