#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:08:56 2019

@author: amrita
"""

import spacy

class AnswerCleaning():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def execute(self, answer):
        answer_parsed = self.nlp(answer)
        answer_toks = [tok.text for tok in answer_parsed]
        answer_pos = [tok.pos_ for tok in answer_parsed]
        answer_lemma = [tok.lemma_ for tok in answer_parsed]
        answer_tok_descendants = []
        for i, token in enumerate(answer_parsed):
            answer_tok_descendants.append([d.lemma_ for d in token.subtree if d!=token])
        return answer_parsed, answer_toks, answer_pos, answer_lemma, answer_tok_descendants    
        
