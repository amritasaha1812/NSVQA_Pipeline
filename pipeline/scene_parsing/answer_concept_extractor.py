#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:51:49 2019

@author: amrita
"""
from vocab.concept_vocab import ConceptVocabulary
from .answer_cleaning import AnswerCleaning 
import spacy

class AnswerConceptExtractor():
    
    def __init__(self, opt):
        self.visual_genome_dir = opt.visual_genome_dir
        self.vg_attr_types = ConceptVocabulary(opt, opt.vg_attr_types_file)
        self.vg_object_types = ConceptVocabulary(opt, opt.vg_object_types_file)
        self.vg_rel_types = ConceptVocabulary(opt, opt.vg_rel_types_file)
        
        self.vg_attr_vocab = self.vg_attr_types.get_vocab()
        self.vg_object_vocab = self.vg_object_types.get_expanded_vocab(['n'])
        self.vg_rel_vocab = self.vg_rel_types.get_expanded_vocab(['v'])
        self.boolean_words = ['yes', 'no', 'true', 'false', 'nothing']
        self.spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        print ('finsihed AnswerConceptExtractor:init')



    def execute(self, ans_tok, ans_lemma, ans_pos, ans_tok_descendants):
        answer_concepts = self.extract_surface_level(ans_tok, ans_lemma, ans_pos, ans_tok_descendants)
        if '' in answer_concepts:
            answer_concepts.remove('')
        answer_substring = ' '.join(answer_concepts)        
        return answer_concepts, answer_substring


    def extract_surface_level(self, segment, segment_lemma, segment_pos, segment_descandants):
        answer_concepts = []
        for w, w_lemma, w_pos in zip(segment, segment_lemma, segment_pos):
                if w_pos=='INTJ' or w_pos=='PUNCT' or w_pos=='NUM' or w in self.boolean_words or w_lemma in self.boolean_words:
                        answer_concepts.append(w)
                if w in self.spacy_stopwords or w_lemma in self.spacy_stopwords:
                        answer_concepts.append(w)
                if (w_lemma+'.n' in self.vg_object_vocab or w+'.n' in self.vg_object_vocab) and w_pos=='NOUN':
                        answer_concepts.append(w)
                if (w_lemma+'.a' in self.vg_attr_vocab or w+'.a' in self.vg_attr_vocab) and (w_pos=='ADJ' or w_pos=='ADP'):
                        answer_concepts.append(w)
                if (w_lemma+'.n' in self.vg_attr_vocab or w+'.n' in self.vg_attr_vocab) and w_pos=='NOUN':
                        answer_concepts.append(w)
                if (w_lemma+'.r' in self.vg_attr_vocab or w+'.r' in self.vg_attr_vocab) and (w_pos=='ADP' or w_pos=='ADJ'):
                        answer_concepts.append(w)
                if (w_lemma+'.v' in self.vg_rel_vocab or w+'.v' in self.vg_rel_vocab) and w_pos=='VERB':
                        answer_concepts.append(w)
        return answer_concepts
