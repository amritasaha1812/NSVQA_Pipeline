#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:10:01 2019

@author: amrita
"""

import json
import math
import torch
from options import get_options
from torch.utils.data import Dataset
import pickle as pkl
from gqa_gold_program_conversion.convert_gqa_programs import ConvertGQAProgramToCustom
import os

class GQADataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.gqa_dir = opt.gqa_dir
        self.split = opt.gqa_split
        self.type = opt.gqa_type
        self.gqa_scenegraph = json.load(open(self.gqa_dir+'/data/raw/scenegraphs/'+self.split+'_sceneGraphs.json', 'rb'))
        self.preprocessed_dump_dir = opt.preprocessed_dump_dir
        self.gqa_preprocessed_dump_dir = os.path.join(self.preprocessed_dump_dir, 'gqa/'+self.type+'/'+self.split)
        if not os.path.exists(self.gqa_preprocessed_dump_dir):
            os.makedirs(self.gqa_preprocessed_dump_dir)
        self.gqa_preprocessed_file = os.path.join(self.gqa_preprocessed_dump_dir, 'scene_parsed_data_gqa_gold.pkl')
        if not os.path.exists(self.gqa_preprocessed_file):
            self.dump_gold_scene_parsed_data()
            pkl.dump(self.gold_scene_parsed_data, open(self.gqa_preprocessed_file, 'wb'))
        else:
            self.gold_scene_parsed_data = pkl.load(open(self.gqa_preprocessed_file, 'rb'))
        self.convert_gqa_program = ConvertGQAProgramToCustom(opt)
        self.split_number = opt.split_number
        self.num_splits = opt.num_splits
        self.sort_data_by_image = bool(opt.sort_data_by_image)
        self.load_gqa_data()
        
    def load_gqa_data(self):
        self.gqa_questions = json.load(open(self.gqa_dir+'/data/raw/questions/'+self.split+'_'+self.type+'_questions.json'))
        self.gqa_question_ids = sorted(list(self.gqa_questions.keys()))
        self.split_size = int(math.ceil(len(self.gqa_question_ids)/self.num_splits))
        self.start = self.split_number*self.split_size
        self.end = min(len(self.gqa_question_ids), self.start+self.split_size)
        if not self.sort_data_by_image:
             self.gqa_question_ids = self.gqa_question_ids[self.start:self.end]
             return

        self.gqa_data = {}
        self.data_index = {}
        self.split_size = int(math.ceil(len(self.gqa_question_ids)/self.num_splits))
        self.start = self.split_number*self.split_size
        self.end = min(len(self.gqa_question_ids), self.start+self.split_size)
        print ('Split size ', self.split_size, ' Start: ', self.start, ' End: ', self.end)
        i = 0
        for question_id in self.gqa_question_ids[self.start:self.end]:  
             d = self.gqa_questions[question_id]
             question_type = d['types']['detailed']
             answer = d['answer']
             image_id = d['imageId']
             if image_id not in self.gold_scene_parsed_data:
                 continue
             question = d['question']
             if image_id not in self.gqa_data:
                 self.gqa_data[image_id] = []
             question_index = len(self.gqa_data[image_id])
             program = self.gqa_questions[question_id]['semantic']
             #converted_program = self.convert_gqa_program.convert_program(program)
             data = {'question_type':question_type, 'question':question, 'question_index':question_index, 'answers':[answer], 'image_id':image_id, 'question_id':question_id, 'program':program}#converted_program':converted_program}
             self.gqa_data[image_id].append(data)
             self.data_index[i] = (image_id, question_index)
             i+=1
             
    def __len__(self):
        if not self.sort_data_by_image:
            return len(self.gqa_question_ids)
        else:
            return len(self.data_index)
   
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
            self.gqa_data[image_id][question_index] = value
        else:
            self.gqa_data[image_id][question_index]['bbox'][bbox_index].update(value)
 
    def __getitem__(self, i):
        if self.sort_data_by_image:
            if isinstance(i, tuple):
                image_id = i[0]
                question_index = i[1]
            elif isinstance(i, int):
                if i >= len(self.data_index):
                   return None
                image_id = self.data_index[i][0]
                question_index = self.data_index[i][1]
            return self.gqa_data[image_id][question_index]
        else:
            question_id = self.gqa_question_ids[i]
            gqa_question = self.gqa_questions[question_id]['question']
            gqa_question_type = self.gqa_question[question_id]['types']['detailed']
            answer = self.gqa_questions[question_id]['answer']
            program = self.gqa_question[question_id]['semantic']
            converted_program = self.convert_gqa_program.convert_program(program)
            image_id = self.gqa_question[question_id]['imageId']
            image_scenegraph = self.gold_scene_parsed_data[image_id]
            d = {'question':gqa_question, 'question_type':gqa_question_type, 'question_id':question_id, 'answers':[answer], 'image_id':image_id, 'question_id':question_id, 'converted_program':converted_program}
            return d
    
    def dump(self, filename):
        pkl.dump(self.gqa_data, open(filename, 'wb'))
             
    def dump_gold_scene_parsed_data(self):
        self.gold_scene_parsed_data = {}
        for image_id in self.gqa_scenegraph:
            d = self.gqa_scenegraph[image_id]
            self.gold_scene_parsed_data[image_id] = {'object_descriptors':{}, 'relationship_descriptors':{}}
            sg_objects = d['objects']
            object_bbox_dict = {}
            for object_id, object_val in sg_objects.items():
                name = object_val['name']
                attributes = object_val['attributes']
                bbox_pos = [object_val['x'], object_val['y'], object_val['w'], object_val['h']]
                bbox_str  = ' '.join([str(x) for x in bbox_pos]) 
                object_bbox_dict[object_id] = bbox_str
                if bbox_str not in self.gold_scene_parsed_data[image_id]['object_descriptors']:
                    self.gold_scene_parsed_data[image_id]['object_descriptors'][bbox_str] = {'x':object_val['x'], 'y':object_val['y'], 'h':object_val['h'], 'w':object_val['w'], 'synsets':{object_id:{}}}
                    self.gold_scene_parsed_data[image_id]['object_descriptors'][bbox_str]['synsets'][object_id]['names'] = [name]
                    self.gold_scene_parsed_data[image_id]['object_descriptors'][bbox_str]['synsets'][object_id]['attributes'] = attributes
            for object_id1, object_val1 in sg_objects.items():
                object_bbox1 = object_bbox_dict[object_id1]
                for rel in object_val1['relations']:
                    object_id2 = rel['object']
                    rel_name = rel['name']
                    object_bbox2 = object_bbox_dict[object_id2]
                    bbox_str = object_bbox1+'\t'+object_bbox2
                    self.gold_scene_parsed_data[image_id]['relationship_descriptors'][bbox_str]={'synset_pairs': {}}
                    rel_id = str(object_id1)+'\t'+str(object_id2)
                    self.gold_scene_parsed_data[image_id]['relationship_descriptors'][bbox_str]['synset_pairs'][rel_id] = {'predicate':rel_name, 'subject':object_id1, 'object':object_id2}


                    
if __name__=="__main__":
    opt = get_options()
    gqa_dataset = GQADataset(opt)
          
