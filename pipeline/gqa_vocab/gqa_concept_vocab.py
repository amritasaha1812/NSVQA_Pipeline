#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:31:50 2019

@author: amrita
"""
import json
import os
import pickle as pkl

class GQAConceptVocabulary():
    
    def __init__(self, opt):
        self.gqa_dir = opt.gqa_dir
        self.gqa_sg_dir = opt.gqa_dir+'/data/raw/scenegraphs/'
        self.gqa_dump_dir = self.gqa_dir+'/data/preprocessed/'
        self.gqa_objects = {}
        self.gqa_attributes = {}
        self.gqa_relations = {}
        if not os.path.exists(self.gqa_dump_dir+'/gqa_objects.pkl') or not os.path.exists(self.gqa_dump_dir+'/gqa_attributes.pkl') or not os.path.exists(self.gqa_dump_dir+'/gqa_relations.pkl'):
            self.dump_gqa_concepts()
            
    def dump_gqa_concepts(self):
        gqa_o_set = set([])
        gqa_a_set = set([])
        gqa_r_set = set([])
        for gqa_file in os.listdir(self.gqa_sg_dir):
            gqa_data = json.load(open(self.gqa_sg_dir+'/'+gqa_file))
            for k in gqa_data:
                for o in gqa_data[k]['objects']:
                    gqa_o_set.add(gqa_data[k]['objects'][o]['name'])
                    gqa_r_set.update(set([gqa_data[k]['objects'][o]['relations'][i]['name'] for i in range(len(gqa_data[k]['objects'][o]['relations']))]))
                    gqa_a_set.update(gqa_data[k]['objects'][o]['attributes'])
        count=0
        for o in gqa_o_set:
            self.gqa_objects[o] = count
            count += 1 
        count=0
        for a in gqa_a_set:
             self.gqa_attributes[a] = count
             count += 1
        count=0
        for r in gqa_r_set:
            self.gqa_relations[r] = count
            count += 1
        self.gqa_relations['looking_for'] = len(self.gqa_relations)
        to_add_to_rels = ["shaking hands","brushing teeth","taking bath","taking a photo","taking a picture","making a face","sticking out of", "looking_for", "taking pictures", "looking", "taking a photo"]
        to_add_to_attrs = ["the same direction","across","along","beyond","backwards","ahead","away", "at camera","front","upward","upwards","downwards","downward", "forward","outside"]
        to_add_to_objs = ["hand towel","cast on shadow","eraser","the camera","frying pan","dish towel", "thing"]
        for k in to_add_to_rels:
            self.gqa_relations[k] = len(self.gqa_relations)
        for k in to_add_to_attrs:
            self.gqa_attributes[k] = len(self.gqa_attributes)
        for k in to_add_to_objs:
            self.gqa_objects[k] = len(self.gqa_objects)
        pkl.dump(self.gqa_objects, open(self.gqa_dump_dir+'/gqa_objects.pkl', 'wb'))
        pkl.dump(self.gqa_attributes, open(self.gqa_dump_dir+'/gqa_attributes.pkl', 'wb'))
        pkl.dump(self.gqa_relations, open(self.gqa_dump_dir+'/gqa_relations.pkl', 'wb'))
    
    def load_gqa_concepts(self, concept_type):
        return pkl.load(open(self.gqa_dump_dir+'/gqa_'+concept_type+'.pkl', 'rb'))
    
    
