#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:11:11 2019

@author: amrita
"""
import json 
import os
import pickle as pkl
from .coco_dataset import COCODataset
from utils.vqa_utils import VQAUtils
from scene_parsing.map_query_to_concept_vocab import MapQueryConceptsToVGConcepts
import utils.utils
import math

class VGCOCOIntersectionDataset():
    
    def __init__(self, opt):
        self.opt = opt
        self.visual_genome_dir = opt.visual_genome_dir
        self.vqa_dir = opt.vqa_dir
        visual_genome_data_dir = os.path.join(self.visual_genome_dir, 'data/raw')
        self.image_ids_with_coco_id = {x['image_id']:x['coco_id'] for x in json.load(open(visual_genome_data_dir+'/image_data.json')) if x['coco_id']}
        self.preprocessed_dump_dir = opt.preprocessed_dump_dir
        self.split = self.opt.coco_split
        self.year = self.opt.coco_year
        self.coco_preprocessed_dump_dir = os.path.join(self.preprocessed_dump_dir, 'coco/'+self.year+'/'+self.split)
        self.image_dir = os.path.join(self.vqa_dir, 'data/'+self.split+self.year)
        self.coco_dataset = COCODataset(opt)
        self.coco_id_image_file_map = {int(x['image_id'].split('.')[0].split('_')[-1]):{'image_filename':x['image_id'], 'height':x['height'], 'width':x['width']} for x in json.load(open(os.path.join(self.coco_preprocessed_dump_dir, 'image_data.json')))}
        self.vg_objects_data_file = os.path.join(visual_genome_data_dir, 'objects.json')
        self.vg_attributes_data_file = os.path.join(visual_genome_data_dir, 'attributes.json')
        self.vg_relationships_data_file = os.path.join(visual_genome_data_dir, 'relationships.json')
        self.vg_coco_preprocessed_dump_dir = os.path.join(self.preprocessed_dump_dir, 'vg_intersection_coco/'+self.year+'/'+self.split)
        if not os.path.exists(self.vg_coco_preprocessed_dump_dir):
            os.makedirs(self.vg_coco_preprocessed_dump_dir) 
        if opt.bbox_detection_type=='gold' or opt.object_detection_type=='gold' or opt.attribute_detection_type=='gold':
            if not os.path.exists(os.path.join(self.vg_coco_preprocessed_dump_dir, 'scene_parsed_data_visualgenome_gold.pkl')):
                self.gold_scene_parsed_data = self.dump_gold_scene_parsed_data()
            else:
                self.gold_scene_parsed_data = pkl.load(open(os.path.join(self.vg_coco_preprocessed_dump_dir, 'scene_parsed_data_visualgenome_gold.pkl'), 'rb'), encoding='latin1')
        self.vqa_utils = VQAUtils(self.image_dir, self.split, self.year)
        self.map_query_concepts_to_vg_concepts = MapQueryConceptsToVGConcepts(opt)
        self.split_number = opt.split_number
        self.num_splits = opt.num_splits
        self.load_vqa_data()

    def dump_gold_scene_parsed_data(self):    
        gold_scene_parsed_data = {}
        for d in json.load(open(self.vg_objects_data_file)):
            image_id = d['image_id']
            if image_id not in self.image_ids_with_coco_id:
                continue
            coco_id = self.image_ids_with_coco_id[image_id]
            if coco_id not in self.coco_id_image_file_map:
                continue
            if coco_id not in gold_scene_parsed_data:
                gold_scene_parsed_data[coco_id] = {}
            gold_scene_parsed_data[coco_id]['image_details'] = self.coco_id_image_file_map[coco_id]
            gold_scene_parsed_data[coco_id]['image_details']['coco_id'] = coco_id
            objects = {}
            for object in d['objects']:
                bbox_str = ' '.join([str(x) for x in [object['x'], object['y'], object['w'], object['h']]])
                if bbox_str not in objects:
                    objects[bbox_str] = {'x':object['x'], 'y':object['y'], 'w':object['w'], 'h':object['h'], 'synsets':{}}
                objects[bbox_str]['synsets'][object['object_id']] = {'object_id':object['object_id'], 'synsets':object['synsets'], 'names':object['names']}
            gold_scene_parsed_data[coco_id]['object_descriptors'] = objects
        
        for d in json.load(open(self.vg_attributes_data_file)):
            image_id = d['image_id']
            if image_id not in self.image_ids_with_coco_id:
                continue
            coco_id = self.image_ids_with_coco_id[image_id]
            if coco_id not in self.coco_id_image_file_map:
                continue
            if coco_id not in gold_scene_parsed_data:
                    gold_scene_parsed_data[coco_id] = {}
                    gold_scene_parsed_data[coco_id]['image_details'] = self.coco_id_image_file_map[coco_id]
                    gold_scene_parsed_data[coco_id]['image_details']['coco_id'] = coco_id
                    gold_scene_parsed_data[coco_id]['object_descriptors'] = {}
            for attribute in d['attributes']:
                bbox_str = ' '.join([str(x) for x in [attribute['x'], attribute['y'], attribute['w'], attribute['h']]])
                if bbox_str not in gold_scene_parsed_data[coco_id]['object_descriptors']:
                    gold_scene_parsed_data[coco_id]['object_descriptors'][bbox_str]={'x':attribute['x'], 'y':attribute['y'], 'w':attribute['w'], 'h':attribute['h'], 'synsets':{}}
                    #gold_scene_parsed_data[coco_id]['object_descriptors'][bbox_str]['synsets'][attribute['object_id']] = {'object_id':attribute['object_id'], 'synsets':attribute['synsets'], 'names':attribute['names'], 'attributes':attribute['attributes']} 
                    gold_scene_parsed_data[coco_id]['object_descriptors'][bbox_str]['synsets'][attribute['object_id']] = attribute
                elif attribute['object_id'] not in gold_scene_parsed_data[coco_id]['object_descriptors'][bbox_str]['synsets']:
                    gold_scene_parsed_data[coco_id]['object_descriptors'][bbox_str]['synsets'][attribute['object_id']] = attribute
                elif 'attributes' in attribute:
                    gold_scene_parsed_data[coco_id]['object_descriptors'][bbox_str]['synsets'][attribute['object_id']]['attributes'] = attribute['attributes']
                
        for d in json.load(open(self.vg_relationships_data_file)):
            image_id = d['image_id']
            if image_id not in self.image_ids_with_coco_id:
                continue
            coco_id = self.image_ids_with_coco_id[image_id]
            if coco_id not in self.coco_id_image_file_map:
                continue
            if coco_id not in gold_scene_parsed_data:
                    gold_scene_parsed_data[coco_id] = {}
                    gold_scene_parsed_data[coco_id]['image_details'] = self.coco_id_image_file_map[coco_id]
                    gold_scene_parsed_data[coco_id]['image_details']['coco_id'] = coco_id
                    gold_scene_parsed_data[coco_id]['object_descriptors'] = {}
            for relationship in d['relationships']:
                sub = relationship['subject']
                obj = relationship['object']
                sub_bbox_str = ' '.join([str(x) for x in [sub['x'], sub['y'], sub['w'], sub['h']]])
                obj_bbox_str = ' '.join([str(x) for x in [obj['x'], obj['y'], obj['w'], obj['h']]])
                if sub_bbox_str not in gold_scene_parsed_data[coco_id]['object_descriptors']:
                    gold_scene_parsed_data[coco_id]['object_descriptors'][sub_bbox_str]={'x':sub['x'], 'y':sub['y'], 'w':sub['w'], 'h':sub['h'], 'synsets':{sub['object_id']:{}}}
                    gold_scene_parsed_data[coco_id]['object_descriptors'][sub_bbox_str]['synsets'][sub['object_id']] = {'object_id':sub['object_id'], 'synsets':sub['synsets']}
                    if 'name' in sub:
                          gold_scene_parsed_data[coco_id]['object_descriptors'][sub_bbox_str]['synsets'][sub['object_id']]['names'] = [sub['name']]
                if obj_bbox_str not in gold_scene_parsed_data[coco_id]['object_descriptors']:
                    gold_scene_parsed_data[coco_id]['object_descriptors'][obj_bbox_str]={'x':obj['x'], 'y':obj['y'], 'w':obj['w'], 'h':obj['h'], 'synsets':{obj['object_id']:{}}}
                    gold_scene_parsed_data[coco_id]['object_descriptors'][obj_bbox_str]['synsets'][obj['object_id']] = {'object_id':obj['object_id'], 'synsets':obj['synsets']}
                    if 'name' in obj:
                          gold_scene_parsed_data[coco_id]['object_descriptors'][obj_bbox_str]['synsets'][obj['object_id']]['names'] = [obj['name']]
                sub_id = sub['object_id']
                obj_id = obj['object_id']
                bbox_str = sub_bbox_str+'\t'+obj_bbox_str
                rel_id = str(sub_id)+'\t'+str(obj_id)
                if 'relationship_descriptors' not in gold_scene_parsed_data[coco_id]:
                     gold_scene_parsed_data[coco_id]['relationship_descriptors'] = {}
                if bbox_str not in gold_scene_parsed_data[coco_id]['relationship_descriptors']:
                     gold_scene_parsed_data[coco_id]['relationship_descriptors'][bbox_str] = {'synset_pairs' : {}}
                if rel_id not in gold_scene_parsed_data[coco_id]['relationship_descriptors'][bbox_str]['synset_pairs']:
                     gold_scene_parsed_data[coco_id]['relationship_descriptors'][bbox_str]['synset_pairs'][rel_id] = []
                gold_scene_parsed_data[coco_id]['relationship_descriptors'][bbox_str]['synset_pairs'][rel_id].append(relationship)        
        pkl.dump(gold_scene_parsed_data, open(os.path.join(self.vg_coco_preprocessed_dump_dir, 'scene_parsed_data_visualgenome_gold.pkl'), 'wb'))
        return gold_scene_parsed_data        
    
    def load_vqa_data(self):
        vqa_annotation = json.load(open(self.vqa_dir+'/data/v2_mscoco_'+self.split+self.year+'_annotations.json'))['annotations']
        vqa_questions = {x['question_id']:x for x in json.load(open(self.vqa_dir+'/data/v2_OpenEnded_mscoco_'+self.split+self.year+'_questions.json'))['questions']}
        self.vqa_data = {}
        self.data_index = {}
        self.split_size = int(math.ceil(len(vqa_annotation)/self.num_splits))
        self.start = self.split_number*self.split_size
        self.end = min(len(vqa_annotation), self.start+self.split_size)
        i = 0
        for d in vqa_annotation[self.start:self.end]:
            question_type = d['question_type']
            answers = d['answers']
            answer_type = d['answer_type']
            image_id = d['image_id']
            if image_id not in self.gold_scene_parsed_data:
                continue
            question_id = d['question_id']
            question = vqa_questions[question_id]['question']
            if image_id not in self.vqa_data:
                self.vqa_data[image_id] = []
            question_index =len(self.vqa_data[image_id])    
            data = {'question_type':question_type, 'question':question, 'question_index':question_index, 'answers':answers, 'answer_type':answer_type, 'image_id':image_id, 'question_id':question_id} 
            self.vqa_data[image_id].append(data)    
            self.data_index[i] = (image_id, question_index)
            i+=1 
            #if len(self.data_index)%3000==0 and len(self.data_index)>0:
            #    break
            
            
    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, i):
        if isinstance(i, tuple):
            image_id = i[0]
            question_index = i[1]
        elif isinstance(i, int):    
            if i >= len(self.data_index):
                return None
            image_id = self.data_index[i][0]
            question_index = self.data_index[i][1]
        return self.vqa_data[image_id][question_index]   

    def __setitem__(self, index, value):
        image_id = None
        question_index = None
        bbox_index = None
        if isinstance(index, tuple):
            image_id = index[0]
            question_index = index[1]
            if len(index)==3:
                bbox_index = index[2]
        else:
            image_id = self.data_index[index][0]
            question_index = self.data_index[index][1]
        if bbox_index is None:
            assert value['image_id'] == image_id
            assert value['question_index'] == question_index
            self.vqa_data[image_id][question_index] = value
        else:
            self.vqa_data[image_id][question_index]['bbox'][bbox_index].update(value)
    
    def dump(self, filename):
        pkl.dump(self.vqa_data, open(filename, 'wb'))
        
