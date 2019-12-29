#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:29:20 2019

@author: amrita
"""
import json
import spacy
from .query_cleaning import QueryCleaning
from .query_concept_extractor import QueryConceptExtractor
from options import get_options

class QueryToTemplate():
    def __init__(self, opt):
        nlp = spacy.load("en")
        self.stopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.nonstopwords = set(['five', 'here', 'above', 'take', 'show', 'various', 'go', 'behind', 'third', 'onto', 'eleven', 'full', 'give', 'next', 'make', 'nine', 'both', 'less', 'front', 'used', 'beside', 'whither','into','twenty', 'one', 'bottom', 'across', 'same','more', 'together', 'fifteen', 'few', 'before', 'after',  'two', 'twelve','sixty', 'further','under', 'still','three','along','four','down', 'top','another', 'eight','alone', 'enough', 'only', 'six', 'empty', 'made', 'whole', 'least', 'ten', 'back', 'up', 'side', 'any', 'part', 'not', 'neither', 'over', 'upon', 'first', 'after', 'last', 'within', 'between', 'hundred', 'below', 'several', 'forty', 'fifty', 'back','doing', 'on', 'in', 'towards'])
        self.stopwords = self.stopwords - self.nonstopwords
        self.query_cleaning = QueryCleaning()
        self.query_concept_extractor = QueryConceptExtractor(opt)
        
    def execute(self, query):
        query_parsed, query_toks, query_pos, query_lemma, query_tok_descendants = self.query_cleaning.execute(query)
        objects, attributes, relations = self.query_concept_extractor.execute(query_toks, query_lemma, query_pos, query_tok_descendants)
        template = []
        query_concepts = set([])
        for word in query_toks:
            if word in self.stopwords:
                template.append(word)
            elif word in objects:
                template.append('<OBJECT>')
                query_concepts.add(word)
            elif word in attributes:
                template.append('<ATTRIBUTE>')
                query_concepts.add(word)
            elif word in relations:
                template.append('<RELATION>')
                query_concepts.add(word)
            else:
                template.append(word)
        return ' '.join(template), query_concepts      
        
        
if __name__=="__main__":
    opt = get_options()
    query_to_template = QueryToTemplate(opt)
    d=json.load(open('/dccstor/cssblr/amrita/VQA/data/v2_OpenEnded_mscoco_train2014_questions.json'))
    for question in d['questions']:
        question = question['question']
        print (question)
        print (query_to_template.execute(question)+'\n')     
