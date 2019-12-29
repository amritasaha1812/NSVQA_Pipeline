#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:06:39 2019

@author: amrita
"""
from vocab.concept_vocab import ConceptVocabulary
from .query_cleaning import QueryCleaning
import spacy

class QueryConceptExtractor():
    
    def __init__(self, opt):
        self.visual_genome_dir = opt.visual_genome_dir
        self.vg_attr_types = ConceptVocabulary(opt, opt.vg_attr_types_file)
        self.vg_object_types = ConceptVocabulary(opt, opt.vg_object_types_file)
        self.vg_rel_types = ConceptVocabulary(opt, opt.vg_rel_types_file)
        
        self.vg_attr_vocab = self.vg_attr_types.get_vocab()
        self.vg_object_vocab = self.vg_object_types.get_expanded_vocab(['n'])
        self.vg_rel_vocab = self.vg_rel_types.get_expanded_vocab(['v', 'r'])
        nlp = spacy.load("en")
        self.stopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.nonstopwords = set(['five', 'here', 'above', 'take', 'show', 'various', 'go', 'behind', 'third', 'onto', 'eleven', 'full', 'give', 'next', 'make', 'nine', 'both', 'less', 'front', 'used', 'beside', 'whither','into','twenty', 'one', 'bottom', 'across', 'same','more', 'together', 'fifteen', 'few', 'before', 'after',  'two', 'twelve','sixty', 'further','under', 'still','three','along','four','down', 'top','another', 'eight','alone', 'enough', 'only', 'six', 'empty', 'made', 'whole', 'least', 'ten', 'back', 'up', 'side', 'any', 'part', 'not', 'neither', 'over', 'upon', 'first', 'after', 'last', 'within', 'between', 'hundred', 'below', 'several', 'forty', 'fifty', 'back','doing', 'on', 'in', 'towards'])
        self.stopwords = self.stopwords - self.nonstopwords        
        
    def execute(self, query, query_lemma, query_pos, query_tok_descendants):
        objects, attributes, relations = self.extract_surface_level(query, query_lemma, query_pos, query_tok_descendants)
        return objects, attributes, relations
        
            
    def extract_surface_level(self, segment, segment_lemma, segment_pos, segment_descandants):
        objects_in_segment = []
        attributes_in_segment = []
        relations_in_segment = []
        segment_desc = {w:w_desc for w, w_desc in zip(segment, segment_descandants)}
        for w, w_lemma, w_pos in zip(segment, segment_lemma, segment_pos):
                if w in self.stopwords or w_lemma in self.stopwords:
                        continue
                if (w+'.n' in self.vg_object_vocab or w_lemma+'.n' in self.vg_object_vocab) and w_pos=='NOUN':
                        objects_in_segment.append(w)
                elif (w+'.a' in self.vg_attr_vocab or w_lemma+'.a' in self.vg_attr_vocab) and (w_pos=='ADJ' or w_pos=='ADP'):
                        attributes_in_segment.append(w)
                elif (w+'.n' in self.vg_attr_vocab or w_lemma+'.n' in self.vg_attr_vocab) and w_pos=='NOUN':
                        attributes_in_segment.append(w)
                elif (w+'.r' in self.vg_attr_vocab or w_lemma+'.r' in self.vg_attr_vocab) and (w_pos=='ADP' or w_pos=='ADJ'):
                        attributes_in_segment.append(w)
                elif (w+'.v' in self.vg_rel_vocab or w_lemma+'.v' in self.vg_rel_vocab) and w_pos=='VERB':
                        relations_in_segment.append(w)
                elif (w+'.r' in self.vg_rel_vocab or w_lemma+'.r' in self.vg_rel_vocab) and (w_pos=='ADP' or w_pos=='ADJ'):
                        relations_in_segment.append(w)
                elif (w+'.s' in self.vg_attr_vocab or w_lemma+'.s' in self.vg_attr_vocab):
                        attributes_in_segment.append(w)
                elif (w+'.n' in self.vg_object_vocab or w_lemma+'.n' in self.vg_object_vocab):
                        objects_in_segment.append(w)
                elif (w+'.a' in self.vg_attr_vocab or w_lemma+'.a' in self.vg_attr_vocab):
                        attributes_in_segment.append(w)
                elif (w+'.n' in self.vg_attr_vocab or w_lemma+'.n' in self.vg_attr_vocab):
                        attributes_in_segment.append(w)
                elif (w+'.r' in self.vg_attr_vocab or w_lemma+'.r' in self.vg_attr_vocab):
                        attributes_in_segment.append(w)
                elif (w+'.v' in self.vg_rel_vocab or w_lemma+'.v' in self.vg_rel_vocab):
                        relations_in_segment.append(w)
                elif (w+'.r' in self.vg_rel_vocab or w_lemma+'.r' in self.vg_rel_vocab):
                        relations_in_segment.append(w)
                #print ('w ', w)
        '''
        intersecting_objects_attributes = set(objects_in_segment).intersection(set(attributes_in_segment))
        for e in intersecting_objects_attributes:
                if e in segment_desc and any([d in intersecting_objects_attributes for d in segment_desc[e]]):
                        attributes_in_segment.remove(e)
        intersecting_objects_relations =  set(objects_in_segment).intersection(set(relations_in_segment))
        for e in intersecting_objects_relations:
                if e in segment_desc and any([d in intersecting_objects_relations for d in segment_desc[e]]):
                        relations_in_segment.remove(e)
                        #print ('deleting ', e , 'from object')
        for o in list(objects_in_segment.keys()):
                if o in attributes_in_segment:
                        attributes_in_segment.remove(o)
                        #print ('deleting ', e, 'from object')
        '''
        return objects_in_segment, attributes_in_segment, relations_in_segment
    
    
