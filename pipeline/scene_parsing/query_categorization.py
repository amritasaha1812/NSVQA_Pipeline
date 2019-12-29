#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:29:20 2019

@author: amrita
"""
import json
import spacy


class QueryCategorization():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    def execute(self, query):
        query = query.lower()
        query_parsed = self.nlp(query)
        query_pos = [tok.pos_ for tok in query_parsed]
        if query_pos[0]=='VERB':
           query_type = 'boolean'
        elif query_pos[0]=='ADV' and query_pos[1]=='ADJ':
           query_type = 'quantitative'
        else:
           query_type = 'logical'
        #print (query, ':::: ', query_type, ':::', query_pos)
        return query_type    
        
        
if __name__=="__main__":
    query_categorization = QueryCategorization()
    d=json.load(open('/dccstor/cssblr/amrita/VQA/data/v2_OpenEnded_mscoco_train2014_questions.json'))
    query_categories_data = {}
    for question in d['questions']:
        question = question['question']
        query_type = query_categorization.execute(question)        
        if query_type not in query_categories_data:
             query_categories_data[query_type] = []
        query_categories_data[query_type].append(question)
    json.dump(query_categories_data, open('q_cat.json', 'w'), indent=1)
