#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:19:36 2019

@author: amrita
"""
import pickle as pkl
import os
import numpy as np
import sys
sys.path.append('../Concept_Catalog_VQA')
from multiclass_unilabel_cce_attribute.datasets.visual_genome_attribute import VisualGenomeAttributeDataset

class VQAAttributeDataset(VisualGenomeAttributeDataset):
    def __init__(self, image_dir, preprocessed_data_dir, preprocessed_data_file, processed_vqa_dataset, processed_vqa_queries, dump_data_path, vocab_file, clusterfile,
            attributes_glove_emb_file, image_concepts_glove_emb_file, gpu_ids, cluster_classify, shuffle_data, sort_by, opt):
        super(VQAAttributeDataset, self).__init__('test', image_dir, preprocessed_data_dir, preprocessed_data_file, dump_data_path, vocab_file, clusterfile, attributes_glove_emb_file, image_concepts_glove_emb_file, gpu_ids, cluster_classify, shuffle_data, sort_by)
        self.processed_vqa_dataset = processed_vqa_dataset
        self.processed_vqa_queries = processed_vqa_queries
        self.max_query_concepts = opt.max_query_concepts        
        self.query_concepts_embed_dim = opt.query_concepts_embed_dim
    
    def get_num_classes(self):
        return super().get_num_classes()
    
    def __len__(self):
        return len(self.processed_vqa_dataset)
 
    def pad_or_clip(self, vec):
        vec_shape = np.asarray(vec).shape
        vec_len = vec_shape[0]
        if vec_len == 0:
            vec = np.zeros((self.max_query_concepts, self.query_concepts_embed_dim), dtype=np.float32)
        else:
            vec_dim = vec_shape[1]
            if vec_len < self.max_query_concepts:
                 pad_len = (self.max_query_concepts - vec_len)
                 zeros = np.zeros((pad_len, vec_dim), dtype=np.float32)
                 vec = np.concatenate([vec, zeros], axis=0)

            elif vec_len > self.max_query_concepts:
                 vec = vec[:pad_len]
        context_attention_vec = np.where(np.sum(vec, axis=1)==0, 1e-5, 1).astype(np.float32)
        return vec, context_attention_vec

    def __getitem__(self, idx):
        data = self.processed_vqa_dataset[idx]
        image_id = data['image_id']
        question_index = data['question_index']
        bbox_index = data['bbox_index']
        bbox = data['bbox']
        image_region = data['image_region']
        query_concepts_emb = {}
        query_info = self.self.processed_vqa_queries[question_index]
        query_concepts_emb.update(query_info['query_objects_emb'])
        query_concepts_emb.update(query_info['query_attributes_emb'])
        query_concept_glove_emb = np.asarray(list(query_concepts_emb.values()), np.float32)
        query_concept_glove_emb, context_attention_vector = self.pad_or_clip(query_concept_glove_emb)
        return image_id, question_index, bbox_index, bbox, image_region, query_concept_glove_emb, context_attention_vector
