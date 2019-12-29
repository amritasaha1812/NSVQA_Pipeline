#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:43:46 2019

@author: amrita
"""
import json
import torch
import pickle as pkl

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, opt, split, year):
        self.vqa_dir = opt.vqa_dir
        self.split = split
        self.year = year
        self.load_vqa_data()
        
    def load_vqa_data(self):
        vqa_annotation = json.load(open(self.vqa_dir+'/data/v2_mscoco_'+self.split+self.year+'_annotations.json'))['annotations']
        vqa_questions = {x['question_id']:x for x in json.load(open(self.vqa_dir+'/data/v2_OpenEnded_mscoco_'+self.split+self.year+'_questions.json'))['questions']}
        self.vqa_data = {}
        self.data_index = {}
        for i,d in enumerate(vqa_annotation):
            question_type = d['question_type']
            answers = d['answers']
            answer_type = d['answer_type']
            image_id = d['image_id']
            question_id = d['question_id']
            question = vqa_questions[question_id]['question']
            if image_id not in self.vqa_data:
                self.vqa_data[image_id] = []
            question_index =len(self.vqa_data[image_id])    
            self.vqa_data[image_id].append({'question_type':question_type, 'question':question, 'question_index':question_index, 'answers':answers, 'answer_type':answer_type, 'image_id':image_id, 'question_id':question_id})    
            self.data_index[i] = (image_id, question_index)
            if i%100==0 and i>0:
                break 
    
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
