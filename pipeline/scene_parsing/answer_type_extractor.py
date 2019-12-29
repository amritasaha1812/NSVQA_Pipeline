#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:54:07 2019

@author: amrita
"""

from .answer_cleaning import AnswerCleaning
import spacy
from pattern.en import singularize
from dateutil.parser import parse

class AnswerTypeExtractor():
    
    def __init__(self, ):
        self.answer_cleaning = AnswerCleaning()
        self.spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.boolean_words = ['yes', 'no', 'true', 'false', 'nothing']
        self.answer_types = {'none':0, 'int':1, 'bool':2, 'datetime':3, 'string':4}
        self.answer_types_inv = {v:k for k,v in self.answer_types.items()}
 
    def is_date(self, w):
        try:
           w_words = w.split(' ')
           if any([x in w_words for x in ['year', 'yr', 'years', 'hr', 'hrs', 'hour', 'hours', 'minute', 'min', 'second', 'sec', "o'clock", "o' clock", "o clock"]]):
             return True
           if parse(w, True):
             return True
        except:
           return False 
         
    def get_answer_type(self, answer):
        answer_types = self.get_answer_type_dict(answer)
        return set(answer_types.values())

    def get_answer_type_dict(self, answer):
        answer_parsed, answer_toks, answer_pos, answer_lemma, _ = self.answer_cleaning.execute(answer)
        answer_types = {}
        if self.is_date(answer):
            answer_types[answer] = self.answer_types['datetime']
            return answer_types
        for w, w_lemma, w_pos in zip(answer_toks, answer_lemma, answer_pos):
            w_sing = singularize(w)
            if w_pos=='NUM' or w.isdigit():
                answer_type = 'int'
            elif w in self.boolean_words or w_lemma in self.boolean_words or w_sing in self.boolean_words:
                answer_type = 'bool'
            elif self.is_date(w):
                answer_type = 'datetime'
            elif w in self.spacy_stopwords or w_lemma in self.spacy_stopwords or w_sing in self.spacy_stopwords:    
                continue
            else:
                answer_type = 'string'
            answer_types[w] = self.answer_types[answer_type]
        return answer_types
            
            
