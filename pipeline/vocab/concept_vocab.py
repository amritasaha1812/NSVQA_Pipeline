#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:13:47 2019

@author: amrita
"""

import json
import os
import spacy

class ConceptVocabulary():
    
    def __init__(self, opt, concept):
       self.nlp = spacy.load("en_core_web_sm")
       self.visual_genome_dir = opt.visual_genome_dir
       self.vocab_file = self.visual_genome_dir+'/data/raw/'+concept+'_types.json'
       self.processed_vocab_file = self.visual_genome_dir+'/data/preprocessed/'+concept+'_types.json'
       self.vocab = self.process_vocab({}, self.vocab_file)
       self.alias_file = self.visual_genome_dir+'/data/raw/'+concept+'_alias.txt'
       self.processed_alias_file = self.visual_genome_dir+'/data/preprocessed/'+concept+'_alias.txt'

    def get_expanded_vocab(self, pos_tag):
        self.expanded_vocab_file = self.processed_vocab_file.replace('.json', '_expanded.json')
        if os.path.exists(self.expanded_vocab_file):
            self.vocab = json.load(open(self.expanded_vocab_file))
        elif os.path.exists(self.alias_file):
            self.vocab.update(self.process_alias_vocab(self.vocab, self.alias_file, pos_tag))
            json.dump(self.vocab, open(self.expanded_vocab_file, 'w'), indent=1)
        print ('got expanded vocab')
        return self.vocab
    
    def get_vocab(self):
        return self.vocab
    
    def process_vocab(self, d, f):
        if os.path.exists(self.processed_vocab_file):
            d = json.load(open(self.processed_vocab_file))
        else: 
            for x in json.load(open(f)):
                pos_tag = x.split('.')[-2]
                x_text = '.'.join(x.split('.')[:-2]).lower().replace('_',' ')
                x_lemma = ' '.join([xi.text if xi.lemma_.startswith('-') else xi.lemma_ for xi in self.nlp(x_text.replace('-',' '))])
                d[x_text+'.'+pos_tag] = [x]
                d[x_lemma+'.'+pos_tag] = [x]
                #for xi in x_text.split(' '):
                #   d[xi+'.'+pos_tag] = [x]
                #for xi in x_lemma.split(' '):
                #   d[xi+'.'+pos_tag] = [x]
            json.dump(d, open(self.processed_vocab_file, 'w'), indent=1)
        return d   
    
    def process_alias_vocab(self, d, f, pos_tags):
        if os.path.exists(self.processed_alias_file):
            d = json.load(open(self.processed_alias_file))
        else: 
            original_lexicon = {'.'.join(x[0].split('.')[:-2]):x[0] for x in d.values()}
            for x in open(f).readlines():
                xs = [xi.strip() for xi in x.split(',')]
                xs_words_in_lexicon = set([])
                for x in xs:
                        xs_words_in_lexicon.update([original_lexicon[w] for w in x.split(' ') if w in original_lexicon])
                xs_words_in_lexicon = list(xs_words_in_lexicon)
                for xi in xs:
                        d.update({xi+'.'+pos_i:xs_words_in_lexicon for pos_i in pos_tags})
            json.dump(d, open(self.processed_alias_file, 'w'), indent=1)
        return d
