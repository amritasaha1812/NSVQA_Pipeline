#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:57:55 2019

@author: amrita
"""

import argparse
import os


class OptionsExecutor():
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--max_bboxes_per_image', default=10, type=int, help='maximum number of bounding boxes per image')
        self.parser.add_argument('--max_attributes_per_bbox', default=5, type=int, help='maximum number of attributes per bounding box')
        self.parser.add_argument('--max_objects_per_bbox', default=5, type=int, help='maximum number of objects per bounding box')
        self.parser.add_argument('--object_sampler_threshold', default=0.0, type=float, help='threshold of object sampling score')
        self.parser.add_argument('--attribute_sampler_threshold', default=0.0, type=float, help='threshold of attribute sampling score')
        self.parser.add_argument('--max_objects_per_query', default=5, type=int, help='maximum number of objects per query')
        self.parser.add_argument('--max_attributes_per_query', default=5, type=int, help='maximum number of attributes per query')
        self.parser.add_argument('--max_relations_per_query', default=5, type=int, help='maximum number of relations per query')
    def parse(self):
        if not self.initialized:
            self.initialize()
            
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        for k, v in args.items():
            print ('%s: %s' % (str(k), str(v)))
        return self.opt    
    
def get_options_exec():
    return OptionsExecutor().parse()    
