#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 08:49:10 2019

@author: amrita
"""
from .answer_cleaning import AnswerCleaning
from .answer_concept_extractor import AnswerConceptExtractor
from .answer_type_extractor import AnswerTypeExtractor
from options import get_options, get_option_str
from dataset.vqa_dataset import VQADataset
import os
import json

class GoldAnswerParser():
    
    def __init__(self, opt):
        self.opt = opt
        self.vqa_dir = opt.vqa_dir
        self.coco_dir = opt.coco_dir
        self.split = opt.coco_split
        self.year = opt.coco_year
        self.vqa_preprocessed_dump_dir = os.path.join(os.path.join(opt.preprocessed_dump_dir, 'vqa'), get_option_str(opt))
        if not os.path.exists(self.vqa_preprocessed_dump_dir):
            os.mkdir(self.vqa_preprocessed_dump_dir)
        self.answer_cleaning = AnswerCleaning()
        self.answer_concept_extractor = AnswerConceptExtractor(opt)
        self.answer_type_extractor = AnswerTypeExtractor()
        self.load()
        
    def get_answer_concepts(self, answer):
        answer_parsed, answer_toks, answer_pos, answer_lemma, answer_tok_descendants = self.answer_cleaning.execute(answer)
        #print ([ai+'('+aj+')' for ai,aj in zip(answer_toks, answer_pos)])
        answer_concepts, answer_substring = self.answer_concept_extractor.execute(answer_toks, answer_lemma, answer_pos, answer_tok_descendants)
        #print (answer, '--->', answer_concepts, '(', answer_substring, ')')
        return answer_substring
 
    def create_answer_format(self):    
        answer_data = []
        #for i, vqa_data_instance in enumerate(self.vqa_dataset):
        for i in range(len(self.vqa_dataset.data_index)):
            index = self.vqa_dataset.data_index[i]
            image_id = index[0]
            question_index = index[1]
            vqa_data_instance = self.vqa_dataset.vqa_data[image_id][question_index] 
            if vqa_data_instance is None:
                break
            question_id = vqa_data_instance['question_id']
            answers = list(set([x['answer'] for x in vqa_data_instance['answers']]))
            answer_d = []
            for answer in answers:
                 #answer_substring = self.get_answer_concepts(answer)       
                 #d = {'answer':answer_substring, 'question_id':question_id}
                 answer_type = list(self.answer_type_extractor.get_answer_type(answer))
                 d = {'answer':answer, 'answer_type':answer_type} 
                 answer_d.append(d)
            answer_data.append(answer_d)
            if i%1000==0:
                print ('finished ', i, 'out of ', len(self.vqa_dataset))
            if i%5000==0:
                json.dump(answer_data, open(self.vqa_preprocessed_dump_dir + '/gold_answer_type_'+self.split+'_'+self.year+'.json', 'w'))
        json.dump(answer_data, open(self.vqa_preprocessed_dump_dir + '/gold_answer_type_'+self.split+'_'+self.year+'.json', 'w')) 

    def load(self):
        if self.opt.vqa_dataset=='coco':
            self.vqa_dataset = VQADataset(self.opt)
            
    def execute(self):
        self.create_answer_format()        


if __name__=="__main__":
    opt = get_options()
    gold_answer_parser = GoldAnswerParser(opt)
    gold_answer_parser.execute()
