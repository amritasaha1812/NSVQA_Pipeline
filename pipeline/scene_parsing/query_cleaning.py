#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:03:59 2019

@author: amrita
code: query cleaning
"""
import spacy

class QueryCleaning():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.simplifying_phrase_file = '../resources/simplifying_phrases.txt'
        self.simplifying_phrases = {x.split('\t')[0].strip():x.split('\t')[1].strip() if '\t' in x else '' for x in open(self.simplifying_phrase_file)}
        self.middle_verb = set(['is', 'was', 'are', 'were', 'would', 'should', 'shall', 'do', 'did', 'does', 'has', 'have', 'kind', 'can', 'be', 'will', 'what', 'why', 'how']) 

    def lemmatize_word(self, word):
        return self.nlp(word)[0].lemma_       
 
    def execute(self, query):
        query = query.lower()
        for phrase, replacement in self.simplifying_phrases.items():
            query = query.replace(phrase, replacement)
        query_parsed = self.nlp(query)
        query_toks = [tok.text for tok in query_parsed]
        query_pos = [tok.pos_ for tok in query_parsed]
        query_lemma = [tok.lemma_ for tok in query_parsed]
        query_tok_descendants = []
        for i, token in enumerate(query_parsed):
            query_tok_descendants.append([d.lemma_ for d in token.subtree if d!=token])
        return query_parsed, query_toks, query_pos, query_lemma, query_tok_descendants
    
    
            
