#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:44:44 2019

@author: amrita
"""
import pickle as pkl
from autocorrect import spell
from annoy import AnnoyIndex
import numpy as np
import os
from .query_concept_extractor import QueryConceptExtractor
from .query_cleaning import QueryCleaning
from options import get_options, get_option_str
from dataset.vqa_object_dataset import VQAObjectDataset
from dataset.vqa_attribute_dataset import VQAAttributeDataset
from utils import utils
import time
import nltk
from nltk.corpus import stopwords

class MapQueryConceptsToVGConcepts():
    def __init__(self, opt):
        self.opt = opt
        self.sceneparser_dataset_type = opt.sceneparser_dataset_type
        if self.sceneparser_dataset_type == 'gqa':
            self.dataset_dir = opt.gqa_dir
        elif self.sceneparser_dataset_type == 'visual_genome':
            self.dataset_dir = opt.visual_genome_dir
        self.qa_dataset_type = opt.qa_dataset_type
        print ('self.dataset_dir ', self.dataset_dir)
        self.glove_clustering_dir = opt.glove_clustering_dir
        vg_obj_annoy_index_file = self.dataset_dir+'/data/preprocessed/annoy_indices/object_annoy_index/annoy_glove_index.ann'
        vg_obj_annoy_keys_file = self.dataset_dir+'/data/preprocessed/annoy_indices/object_annoy_index/annoy_glove_words.pkl'
        vg_attr_annoy_index_file = self.dataset_dir+'/data/preprocessed/annoy_indices/attr_annoy_index/annoy_glove_index.ann'
        vg_attr_annoy_keys_file = self.dataset_dir+'/data/preprocessed/annoy_indices/attr_annoy_index/annoy_glove_words.pkl' 
        self.datadump_folder = os.path.join(os.path.join(opt.preprocessed_dump_dir, self.qa_dataset_type), get_option_str(opt))
        self.pad_object = 0
        self.pad_attribute = 0
        self.pad_relation = 0
        if not os.path.exists(self.datadump_folder+'object_vocab.pkl'):
            print ('In MapQueryConceptsToVGConcepts: Going to vqa object dataset')
            self.object_catalog_preprocessed_dir = os.path.join(self.opt.concepts_catalog_dir, self.opt.object_catalog_preprocessed_dir)
            self.vqa_object_dataset = VQAObjectDataset(None, None, None, None, None, self.object_catalog_preprocessed_dir, None, None, None, None, self.opt.gpu_ids, 0, 0, '', self.opt)
            self.object_vocab = {'<NULL>':self.pad_object}
            if self.sceneparser_dataset_type=='visual_genome':
               self.vqa_object_dataset = VQAObjectDataset(None, None, None, None, None, self.object_catalog_preprocessed_dir, None, None, None, None, self.opt.gpu_ids, 0, 0, '', self.opt)
               self.object_vocab.update({k:v+1 for k,v in self.vqa_object_dataset.vocab.items()})
            else:
               self.object_vocab.update({k:v+1 for k,v in pkl.load(open(os.path.join(self.dataset_dir, 'data/preprocessed/object_vocab.pkl'), 'rb')).items()})
            pkl.dump(self.object_vocab, open(self.datadump_folder+'/object_vocab.pkl', 'wb'))
        else:
            self.object_vocab = pkl.load(open(self.datadump_folder+'/object_vocab.pkl', 'rb'))
        self.object_vocabsize = len(set(self.object_vocab.values()))
        print ('Object vocab size ', self.object_vocabsize) 
        if not os.path.exists(self.datadump_folder+'/attribute_vocab.pkl'):
            print ('In MapQueryConceptsToVGConcepts: Going to vqa attribute dataset')
            self.attribute_catalog_preprocessed_dir = os.path.join(self.opt.concepts_catalog_dir, self.opt.attribute_catalog_preprocessed_dir)
            self.attribute_vocab = {'<NULL>':self.pad_attribute}
            if self.sceneparser_dataset_type=='visual_genome':
                self.vqa_attribute_dataset = VQAAttributeDataset(None, None, None, None, None, self.attribute_catalog_preprocessed_dir, None, None, None, None, self.opt.gpu_ids, 0, 0, '', self.opt)    
                self.attribute_vocab.update({k:v+1 for k,v in self.vqa_attribute_dataset.vocab.items()})
            else:
                self.attribute_vocab.update({k:v+1 for k,v in pkl.load(open(os.path.join(self.dataset_dir, 'data/preprocessed/attribute_vocab.pkl'), 'rb')).items()})
            pkl.dump(self.attribute_vocab, open(self.datadump_folder+'/attribute_vocab.pkl', 'wb'))
        else:
            self.attribute_vocab = pkl.load(open(self.datadump_folder+'/attribute_vocab.pkl', 'rb'))
        self.attribute_vocabsize = len(set(self.attribute_vocab.values()))
        print ('Attribute vocab size ', self.attribute_vocabsize)

        if self.sceneparser_dataset_type=='gqa':
            if not os.path.exists(self.datadump_folder+'/relation_vocab.pkl'):
                print ('In MapQueryConceptsToVGConcepts: Going to vqa relation dataset')
                self.relation_vocab = {'<NULL>':self.pad_relation}
                self.relation_vocab.update({k:v+1 for k,v in pkl.load(open(os.path.join(self.dataset_dir, 'data/preprocessed/relations_vocab.pkl'), 'rb')).items()})
            else:
                self.relation_vocab = pkl.load(open(self.datadump_folder+'/relations_vocab.pkl', 'rb'))
        self.relation_vocabsize = len(set(self.relation_vocab.values())) 
        print ('Relation vocab size ', self.relation_vocabsize)
        self.stopwords = set(stopwords.words('english'))
        self.stopwords = self.stopwords - set(self.attribute_vocab)
        self.stopwords = self.stopwords - set(self.object_vocab)
        self.query_cleaning = QueryCleaning()
        
        self.glove_emb = {x.strip().split(' ')[0]:[float(xi) for xi in x.strip().split(' ')[1:]] for x in open(self.glove_clustering_dir+'/data/glove/glove.6B.100d.txt').readlines()}
        if not (os.path.exists(vg_obj_annoy_index_file) and os.path.exists(vg_obj_annoy_keys_file)):
           self.create_indiv_annoy_index(self.object_vocab, vg_obj_annoy_index_file, vg_obj_annoy_keys_file)
        self.vg_obj_annoy_index = AnnoyIndex(100, 'euclidean')
        self.vg_obj_annoy_index.load(vg_obj_annoy_index_file)
        self.vg_obj_annoy_keys = pkl.load(open(vg_obj_annoy_keys_file, 'rb'), encoding='latin1')
        if not (os.path.exists(vg_attr_annoy_index_file) and os.path.exists(vg_attr_annoy_keys_file)):
           self.create_indiv_annoy_index(self.attribute_vocab, vg_attr_annoy_index_file, vg_attr_annoy_keys_file)
        self.vg_attr_annoy_index = AnnoyIndex(100, 'euclidean')  
        self.vg_attr_annoy_index.load(vg_attr_annoy_index_file)
        self.vg_attr_annoy_keys = pkl.load(open(vg_attr_annoy_keys_file, 'rb'), encoding='latin1')      
 
    def create_annoy_index(self, object_vocab, attribute_vocab, vg_annoy_index_file, vg_annoy_keys_file):
        vocab = set([])
        vocab.update([self.remove_postag(k.lower().strip()) for k in object_vocab])
        vocab.update([self.remove_postag(k.lower().strip()) for k in attribute_vocab])
        vocab.remove('<null>')
        print ('Going to create Annoy Index in ', vg_annoy_index_file)
        ann_index = AnnoyIndex(100, 'euclidean')
        ann_pkl = []
        i = 0
        for concept in vocab:
            glove_emb = self.get_concept_emb(concept)
            if glove_emb is not None:
               ann_index.add_item(i, glove_emb)
            ann_pkl.append(concept)
            i+=1
        ann_index.build(100)
        ann_index.save(vg_annoy_index_file)
        pkl.dump(ann_pkl, open(vg_annoy_keys_file, 'wb'))
        
    def create_indiv_annoy_index(self, vocab, vg_annoy_index_file, vg_annoy_keys_file):
        vocab = [self.remove_postag(k.lower().strip()) for k in vocab]
        vocab.remove('<null>')
        print ('Going to create Annoy Index in ', vg_annoy_index_file)
        ann_index = AnnoyIndex(100, 'euclidean')
        ann_pkl = []
        i = 0
        for concept in vocab:
            glove_emb = self.get_concept_emb(concept)
            if glove_emb is not None:
                ann_index.add_item(i, glove_emb)
            ann_pkl.append(concept)
            i+=1
        ann_index.build(100)
        ann_index.save(vg_annoy_index_file)
        pkl.dump(ann_pkl, open(vg_annoy_keys_file, 'wb'))



    def get_concept_emb(self, concept):
        concept = self.clean_concept(concept)
        concept_wo_stopwords = [x for x in concept if x not in self.stopwords]
        if len(concept_wo_stopwords)==0:
           concept_wo_stopwords = concept
        embs = []
        for concept_i in concept_wo_stopwords:
            if concept_i not in self.glove_emb:
                concept_i = spell(concept_i)
            if concept_i in self.glove_emb:
                embs.append(self.glove_emb[concept_i])
            else:
                concept_lemma = self.query_cleaning.lemmatize_word(concept_i) 
                if concept_lemma in self.glove_emb:
                   embs.append(self.glove_emb[concept_lemma])
        if len(embs)>0:
            embs = np.asarray(embs)
            emb = np.mean(embs, axis=0)
            return emb
        else:
            return None
           
    def remove_postag(self, concept):
        if '.' in concept and len(concept.split('.'))==3:
               concept = '.'.join(concept.split('.')[:-2])
        return concept
 
    def clean_concept(self, concept):
        return concept.replace('+','_').replace(' ','_').replace('-','_').replace("'s","").replace("?","").replace("/","_").replace(",","_").replace("'","").strip().split('_')
        
    def binarize(self, items, vocab):
        binarized_items = []
        for item in items:
            binarized_items.append(vocab[item])
        return np.asarray(binarized_items)    
        
    def map_concepts_to_vg_concepts(self, query_objects, query_attributes, query_relations, one_hot=False, binarize_concept_map=False, get_map=True):
        start = time.time()
        query_objects = set([self.remove_postag(x) for x in query_objects])
        query_concepts = set([])
        query_concepts.update(query_objects)
        query_concepts.update(query_attributes)
        query_concepts = set([x.lower() for x in query_concepts])
        if query_relations is not None:
            query_concepts.update(query_relations)
        if '' in query_concepts:
            raise Exception('Empty concept in query_concepts')
        if get_map:
            nn_vg_objects_map = {}
            nn_vg_attributes_map = {}
            nn_vg_objects_bin_map = {}
            nn_vg_attributes_bin_map = {}   
        nn_vg_objects = []
        nn_vg_attributes = []
        if query_relations is not None:
            nn_vg_relations_map = {}
            nn_vg_relations = []
        query_concepts_glove_emb = {}
        for concept in query_concepts:
            con_emb = self.get_concept_emb(concept)
            if con_emb is None:
                 continue
            query_concepts_glove_emb[concept] = con_emb
            if concept in query_objects:# and vg_concept in self.object_vocab:
                 vg_concept_dist = self.vg_obj_annoy_index.get_nns_by_vector(con_emb, 1, include_distances=True)
                 vg_concept = vg_concept_dist[0][0]
                 dist = vg_concept_dist[1][0]
                 if dist<=3.4:
                    vg_concept = self.vg_obj_annoy_keys[vg_concept]
                    nn_vg_objects.append(vg_concept)
                    if get_map:
                       nn_vg_objects_map[concept] = len(nn_vg_objects)-1
            if concept in query_attributes:# and vg_concept in self.attribute_vocab:
                 vg_concept_dist = self.vg_attr_annoy_index.get_nns_by_vector(con_emb, 1, include_distances=True)
                 vg_concept = vg_concept_dist[0][0]
                 dist = vg_concept_dist[1][0]
                 if dist<=3.4:
                     vg_concept = self.vg_attr_annoy_keys[vg_concept]
                     nn_vg_attributes.append(vg_concept)
                     if get_map:
                        nn_vg_attributes_map[concept] = len(nn_vg_attributes)-1
            if query_relations is not None and concept in query_relations:
                 nn_vg_relations.append(concept)
                 if get_map:
                     nn_vg_relations_map[concept] = len(nn_vg_relations)-1
        if not get_map:
            if query_relations is not None:
                return nn_vg_objects, nn_vg_attributes, nn_vg_relations
            else:
                return nn_vg_objects, nn_vg_attributes
        for concept in query_concepts:
            if binarize_concept_map:
                if concept in nn_vg_objects_map:
                    vector_objects = np.asarray([self.object_vocab[nn_vg_objects[nn_vg_objects_map[concept]]]])#self.binarize(nn_vg_objects_map[concept], self.object_vocab)
                    if not one_hot:
                       nn_vg_objects_bin_map[concept] = vector_objects
                    else:
                       nn_vg_objects_bin_map[concept] = utils.one_hot(vector_objects, self.object_vocabsize).reshape(1, self.object_vocabsize)
                if concept in nn_vg_attributes_map:
                    vector_attributes = np.asarray([self.attribute_vocab[nn_vg_attributes[nn_vg_attributes_map[concept]]]])#self.binarize(nn_vg_attributes_map[concept], self.attribute_vocab)   
                    if not one_hot:
                       nn_vg_attributes_bin_map[concept] = vector_attributes
                    else:
                       nn_vg_attributes_bin_map[concept] = utils.one_hot(vector_attributes, self.attribute_vocabsize).reshape(1, self.attribute_vocabsize)
        if binarize_concept_map:
            if len(nn_vg_objects_map)==0:
                nn_vg_objects_bin_map = self.add_dummy_concept(nn_vg_objects_bin_map, self.object_vocabsize)
            if len(nn_vg_attributes_map)==0:
                nn_vg_attributes_bin_map = self.add_dummy_concept(nn_vg_attributes_bin_map, self.attribute_vocabsize)
        if not binarize_concept_map:
           if query_relations is not None:
               return nn_vg_objects_map, nn_vg_attributes_map, nn_vg_relations_map, nn_vg_objects, nn_vg_attributes, nn_vg_relations
           else:
               return nn_vg_objects_map, nn_vg_attributes_map, nn_vg_objects, nn_vg_attributes
        else:
           if query_relations is not None:
               return nn_vg_objects_bin_map, nn_vg_attributes_bin_map, nn_vg_objects, nn_vg_attributes, nn_vg_relations #, query_concepts_glove_emb
           else:
               return nn_vg_objects_bin_map, nn_vg_attributes_bin_map, nn_vg_objects, nn_vg_attributes


    def add_dummy_concept(self, map_, size):
        map_[''] = np.zeros((1, size), dtype=np.float32)
        return map_

    def get_query_concepts_glove_emb(self, query_objects, query_attributes, query_relations):
        query_objects_glove_emb = {}
        for obj in query_objects:
            obj_emb = self.get_concept_emb(obj)
            query_objects_glove_emb[obj] = obj_emb
        query_attrs_glove_emb = {}
        for attr in query_attributes:
            attr_emb = self.get_concept_emb(attr)
            query_attrs_glove_emb[attr] = attr_emb
        query_rels_glove_emb = {}
        for rel in query_relations:
            rel_emb = self.get_concept_emb(rel)
            query_rels_glove_emb[rel] = rel_emb
        return query_objects_glove_emb, query_attrs_glove_emb, query_rels_glove_emb


        
