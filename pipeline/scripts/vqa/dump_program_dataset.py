#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:58:48 2019

@author: amrita
"""
from rule_based_program_generation.rule_based_program_generator import RuleBasedProgramGeneration
#from rule_based_program_generation.rule_based_program_generator import RuleBasedProgramGeneration
import os
import pickle as pkl
from options import get_options, get_option_str
from utils.vqa_utils import VQAUtils
from dataset.vqa_dataset import VQADataset
from dataset.vg_coco_intersection_dataset import VGCOCOIntersectionDataset

class DumpProgramDataset():
    
    def __init__(self, opt):
        self.opt = opt
        self.split = opt.coco_split
        self.year = opt.coco_year
        self.coco_dir = opt.coco_dir
        self.image_dir = os.path.join(self.coco_dir, self.split+self.year)
        self.vqa_utils = VQAUtils(self.image_dir, self.split, self.year)
        if self.opt.vqa_dataset == 'coco':
            self.dataset = VQADataset(self.opt)
        elif self.opt.vqa_dataset == 'vg_intersection_coco':
            self.dataset = VGCOCOIntersectionDataset(self.opt)
        self.rule_based_program_generator = RuleBasedProgramGeneration(self.opt)
        self.preprocessed_annotation_dir = os.path.join(os.path.join(opt.preprocessed_dump_dir, 'vqa'), get_option_str(opt))
        self.program_annotation_file = os.path.join(self.preprocessed_annotation_dir, opt.program_annotation_file)
        print ('self.preprocessed_annotation_dir ',self.preprocessed_annotation_dir)
        print ('self.program_annotation_file ', self.program_annotation_file)        

    def execute(self):
        data = []
        for entry in self.dataset:
            if entry is None:
                break
            question_id = entry['question_id']
            question_type = entry['question_type']
            question = entry['question']
            question_index = entry['question_id']
            answers = entry['answers']
            answer_type = entry['answer_type']
            image_id = entry['image_id']
            programs = self.rule_based_program_generator.rule_based_program(question)
            d = {}
            d['image_index'] = image_id
            d['program'] = programs
            d['question_index'] = question_id
            d['image_filename'] = self.vqa_utils.coco_image_id_to_filename(image_id)
            d['question'] = question
            d['question_family_index'] = question_type
            d['split'] = self.split
            d['answer'] = answers
            data.append(d)
        self.program_data = {'info':{'split':self.split, 'year':self.year}, 'questions':data}
        
    def dump(self):
        pkl.dump(self.program_data, open(self.program_annotation_file, 'wb'))
       
    
if __name__=="__main__":
    opt = get_options()
    dump_program_dataset = DumpProgramDataset(opt)
    dump_program_dataset.execute()
    dump_program_dataset.dump()
         
