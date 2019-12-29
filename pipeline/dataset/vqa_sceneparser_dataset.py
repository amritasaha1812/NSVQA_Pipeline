#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:10:23 2019

@author: amrita
"""
import json 
import torch
import os
import pickle as pkl
import numpy as np
from annoy import AnnoyIndex
from random import shuffle
from utils import utils
#from dataset.vqa_object_dataset import VQAObjectDataset
#from dataset.vqa_attribute_dataset import VQAAttributeDataset
from rule_based_program_generation.rule_based_program_generator import RuleBasedProgramGeneration
from rule_based_program_generation.program import Program, ProgramConfig
from options import get_option_str 
from scene_parsing.answer_type_extractor import AnswerTypeExtractor
from preprocess.vqa.preprocess import tokenize, encode, build_vocab, SPECIAL_TOKENS
from scene_parsing.query_categorization import QueryCategorization
from scene_parsing.query_to_template import QueryToTemplate
class VQA_SceneParser_Dataset():
    
    def __init__(self, opt, opt_executor):
        self.opt = opt
        self.opt_executor = opt_executor
        self.visual_genome_dir = opt.visual_genome_dir
        self.datadump_folder = os.path.join(os.path.join(opt.preprocessed_dump_dir, 'vqa'), get_option_str(opt))
        self.preprocessed_annotation_file = os.path.join(self.datadump_folder , self.opt.preprocessed_annotation_file)
        self.preprocessed_annotation = pkl.load(open(self.preprocessed_annotation_file, 'rb'))
        self.data_index = {}
        count = 0
        for i in self.preprocessed_annotation:
            for j in range(len(self.preprocessed_annotation[i])):
                self.data_index[count] = (i,j)
                count += 1
        self.max_bbox_per_image = self.opt_executor.max_bboxes_per_image
        self.max_attributes_per_bbox = self.opt_executor.max_attributes_per_bbox
        self.max_objects_per_bbox = self.opt_executor.max_objects_per_bbox
        self.max_objects_per_query = self.opt_executor.max_objects_per_query
        self.max_attributes_per_query = self.opt_executor.max_attributes_per_query
        self.max_relations_per_query = self.opt_executor.max_relations_per_query
        self.object_sampler_threshold = self.opt_executor.object_sampler_threshold
        self.attribute_sampler_threshold = self.opt_executor.attribute_sampler_threshold
        self.max_program_length = 5
        self.max_words_per_query = self.opt.max_words_question
        self.max_words_per_answer = self.opt.max_words_answer
        self.max_types_per_answer = self.opt.max_types_answer
        self.vg_mapping_in_query = self.opt.vg_mapping_in_query
        self.object_catalog_preprocessed_dir = os.path.join(self.opt.concepts_catalog_dir, self.opt.object_catalog_preprocessed_dir)
        self.verbose = bool(self.opt.verbose)
        self.pad_object = 0
        self.pad_attribute = 0
        self.pad_relation = 0
        if not os.path.exists(self.datadump_folder+'/object_vocab.pkl'):
            self.vqa_object_dataset = VQAObjectDataset(None, None, None, None, None, self.object_catalog_preprocessed_dir, None, None, None, None, self.opt.gpu_ids, 0, 0, '', self.opt)
            if self.verbose:
               print ('Going to vqa object dataset')
            self.object_vocab = {'<NULL>':self.pad_object}
            self.object_vocab.update({k:v+1 for k,v in self.vqa_object_dataset.vocab.items()})
            self.object_vocab_inv = {v:k for k,v in self.object_vocab.items()}
            pkl.dump(self.object_vocab, open(self.datadump_folder+'/object_vocab.pkl', 'wb'))
        else:
            self.object_vocab = pkl.load(open(self.datadump_folder+'/object_vocab.pkl', 'rb'))
        self.object_vocab_inv = {v:k for k,v in self.object_vocab.items()}
        self.attribute_catalog_preprocessed_dir = os.path.join(self.opt.concepts_catalog_dir, self.opt.attribute_catalog_preprocessed_dir)
        if not os.path.exists(self.datadump_folder+'/attribute_vocab.pkl'):       
            self.vqa_attribute_dataset = VQAAttributeDataset(None, None, None, None, None, self.attribute_catalog_preprocessed_dir, None, None, None, None, self.opt.gpu_ids, 0, 0, '', self.opt)
            if self.verbose:
               print ('Going to vqa attribute dataset')
            self.attribute_vocab = {'<NULL>':self.pad_attribute}
            self.attribute_vocab.update({k:v+1 for k,v in self.vqa_attribute_dataset.vocab.items()})
            pkl.dump(self.attribute_vocab, open(self.datadump_folder+'/attribute_vocab.pkl', 'wb'))
        else:
            self.attribute_vocab = pkl.load(open(self.datadump_folder+'/attribute_vocab.pkl', 'rb'))
        self.attribute_vocab_inv = {v:k for k,v in self.attribute_vocab.items()}
        self.vocab = json.load(open(self.datadump_folder+'/vocab.json', 'r'))
        if not self.vg_mapping_in_query:
            self.query_objects_vocab = self.vocab['query_objects_token_to_idx']
            self.query_attributes_vocab = self.vocab['query_attributes_token_to_idx']
            self.query_relations_vocab = self.vocab['query_relations_token_to_idx']
            self.pad_object = self.query_objects_vocab['<NULL>']
            self.pad_attribute = self.query_attributes_vocab['<NULL>']
            self.pad_relation = self.query_relations_vocab['<NULL>']
        self.vg_obj_annoy_index_file = self.visual_genome_dir+'/data/preprocessed/annoy_indices/object_annoy_index/annoy_glove_index.ann'
        self.vg_obj_annoy_keys_file = self.visual_genome_dir+'/data/preprocessed/annoy_indices/object_annoy_index/annoy_glove_words.pkl'
        self.vg_obj_annoy_keys = pkl.load(open(self.vg_obj_annoy_keys_file, 'rb'), encoding='latin1')
        self.vg_obj_annoy_keys_inv = {v:k for k,v in enumerate(self.vg_obj_annoy_keys)}
 
        self.vg_attr_annoy_index_file = self.visual_genome_dir+'/data/preprocessed/annoy_indices/attr_annoy_index/annoy_glove_index.ann'
        self.vg_attr_annoy_keys_file = self.visual_genome_dir+'/data/preprocessed/annoy_indices/attr_annoy_index/annoy_glove_words.pkl'
        self.vg_attr_annoy_keys = pkl.load(open(self.vg_attr_annoy_keys_file, 'rb'), encoding='latin1')
        self.vg_attr_annoy_keys_inv = {v:k for k,v in enumerate(self.vg_attr_annoy_keys)}

        self.query_concepts_embed_dim = opt.query_concepts_embed_dim
        self.vg_obj_annoy_index = AnnoyIndex(self.query_concepts_embed_dim, 'euclidean')
        self.vg_obj_annoy_index.load(self.vg_obj_annoy_index_file)

        self.vg_attr_annoy_index = AnnoyIndex(self.query_concepts_embed_dim, 'euclidean')
        self.vg_attr_annoy_index.load(self.vg_attr_annoy_index_file)
        if not self.vg_mapping_in_query:
            discard_relations = False
        else:
            discard_relations = True
        self.rule_based_program_generator = RuleBasedProgramGeneration(opt, discard_relations)
        self.answer_type_extractor = AnswerTypeExtractor()
        self.query_categorization = QueryCategorization()
        self.query_to_template = QueryToTemplate(self.opt)
    

    def load_program_data(self):
        data = pkl.load(open(self.program_annotation_file, 'rb'), encoding='latin1')
        program_annotation = {}
        for q in data['questions']:
            image_index = q['image_index']
            question_index = q['question_index']
            if image_index not in program_annotation:
                program_annotation[image_index] = {}
            if question_index not in program_annotation[image_index]:
                program_annotation[image_index][question_index] = {}
            program_annotation[image_index][question_index] = q
        return program_annotation    
        
    def __len__(self):
        return len(self.data_index)

    def clip(self, bbox):
        bbox = bbox[:self.max_bbox_per_image]
        return bbox
    
    def get_bbox_mask(self, bbox, empty_bbox):
        bbox_mask = np.asarray([x!=empty_bbox for x in bbox], dtype=np.float32)
        return bbox_mask
    
    def add_bbox_object_masks(self, bbox):
        for bbox_i in bbox:
            x = bbox_i['object_distribution']
            sorted_x = np.argsort(x)[::-1]
            sorted_x_thresholded = set(sorted_x).intersection(set(np.nonzero(x>self.object_sampler_threshold)[0]))
            if len(sorted_x_thresholded) > self.max_objects_per_bbox:
                sorted_x_thresholded = list(sorted_x_thresholded)
                shuffle(sorted_x_thresholded)
                sorted_x_thresholded = sorted_x_thresholded[:self.max_objects_per_bbox]
            sorted_x_thresholded = sorted(sorted_x_thresholded)
            bbox_i['object_ids'] = np.asarray(utils.pad_to_max(sorted_x_thresholded, self.pad_object, self.max_objects_per_bbox))
            bbox_i['object_labels'] = utils.pad_to_max([self.object_vocab_inv[i] for i in sorted_x_thresholded], '<NULL>', self.max_objects_per_bbox)
            bbox_i['object_scores'] = np.asarray(utils.pad_to_max([x[i] for i in sorted_x_thresholded], 0., self.max_objects_per_bbox))
            bbox_i['object_masks'] = np.asarray(bbox_i['object_ids']!=self.pad_object, dtype=np.float32)
        return bbox
    
    def add_bbox_attribute_masks(self, bbox):
        for bbox_i in bbox:
            x = bbox_i['attribute_distribution']
            sorted_x = np.argsort(x)[::-1]
            sorted_x_thresholded = set(sorted_x).intersection(set(np.nonzero(x>self.attribute_sampler_threshold)[0]))
            if len(sorted_x_thresholded) > self.max_attributes_per_bbox:
                sorted_x_thresholded = list(sorted_x_thresholded)
                shuffle(sorted_x_thresholded)
                sorted_x_thresholded = sorted_x_thresholded[:self.max_attributes_per_bbox]
            sorted_x_thresholded = sorted(sorted_x_thresholded)
            bbox_i['attribute_ids'] = np.asarray(utils.pad_to_max(sorted_x_thresholded, self.pad_attribute, self.max_attributes_per_bbox))
            bbox_i['attribute_labels'] = utils.pad_to_max([self.attribute_vocab_inv[i] for i in sorted_x_thresholded], '<NULL>', self.max_attributes_per_bbox)
            bbox_i['attribute_scores'] = np.asarray(utils.pad_to_max([x[i] for i in sorted_x_thresholded], 0., self.max_attributes_per_bbox))
            bbox_i['attribute_masks'] = np.asarray(bbox_i['attribute_ids']!=self.pad_object, dtype=np.float32)
        return bbox
    
    def get_embedding(self, arr, attr_type):
        embs = []
        for x in arr:
            if len(x)==0 or x=='<NULL>':
                emb = np.zeros((self.query_concepts_embed_dim), dtype=np.float32)
            else:
                if attr_type=='object':
                   emb = np.asarray(self.vg_obj_annoy_index.get_item_vector(self.vg_obj_annoy_keys_inv[x]), dtype=np.float32)
                elif attr_type=='attribute':
                   emb = np.asarray(self.vg_attr_annoy_index.get_item_vector(self.vg_attr_annoy_keys_inv[x]), dtype=np.float32)
            if len(emb.shape)==1:
                emb = np.expand_dims(emb, axis=0)
            embs.append(emb)
        embs = np.concatenate(embs, axis=0)
        return embs.tolist()
    
    def __getitem__(self, idx):
        if idx >= len(self.data_index):
           return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        image_index, question_index = self.data_index[idx]
        data = self.preprocessed_annotation[image_index][question_index]
        question_type = data['question_type']
        question = data['question']
        question_category = self.query_categorization.execute(question)
        question_template, question_concepts = self.query_to_template.execute(question)
        if self.verbose:
           print ('\nQuestion No ', idx , ': ', question)
           print ('Question Category ', question_category)
           print ('Question Template ', question_template)
        question_index = data['question_index']
        answers = [answer['answer'] for answer in data['answers']]
        answer_type = data['answer_type']
        image_id = data['image_id']
        question_id = data['question_id']
        q_tokens = tokenize(question,'query', lemmatize=True, add_start_token=True, add_end_token=True)
        q_tokens = utils.pad_or_clip(q_tokens, '<END>', self.max_words_per_query)
        q_encoded = np.asarray(encode(q_tokens,
                         self.vocab['question_token_to_idx'],
                         allow_unk=self.opt.encode_unk == 1), dtype=np.int64)
        a_tokens = [utils.pad_or_clip(tokenize(answer, 'answer', lemmatize=False, add_start_token=False, add_end_token=False),  '<END>', self.max_words_per_answer) for answer in answers]
        a_encoded = np.asarray([encode(a, self.vocab['answer_token_to_idx'], allow_unk=self.opt.encode_unk == 1) for a in a_tokens])
        a_types = [list(self.answer_type_extractor.get_answer_type(answer)) for answer in answers]
        if self.verbose:
           print ('Answer Types ', ([[self.answer_type_extractor.answer_types_inv[x] for x in a_type] for a_type in a_types]))
           print ('Answer ', answers)
        a_types = [utils.pad_or_clip(a, self.answer_type_extractor.answer_types['none'], self.max_types_per_answer) for a in a_types]
        query_objects = data['query_objects']
        query_attributes = data['query_attributes']
        query_relations = data['query_relations']
        query_to_vg_objects_map = data['query_to_vg_objects_map']
        query_to_vg_attrs_map = data['query_to_vg_attrs_map']
        query_to_vg_rels_map = data['query_to_vg_rels_map']
        if self.vg_mapping_in_query:
            q_attributes = [self.attribute_vocab[a] for a in query_attributes]
            q_objects = [self.object_vocab[o] for o in query_objects]
            q_relations = []
            q_attributes = utils.pad_or_clip(q_attributes, self.pad_attribute, self.max_attributes_per_query)
            q_objects = utils.pad_or_clip(q_objects, self.pad_object, self.max_objects_per_query)
            q_relations = utils.pad_or_clip(q_relations, self.pad_relation, self.max_relations_per_query)
            query_to_vg_objects_map = {x:i for x,i in query_to_vg_objects_map.items() if i<self.max_objects_per_query}
            query_to_vg_attrs_map = {x:i for x,i in query_to_vg_attrs_map.items() if i<self.max_attributes_per_query}
            query_to_vg_rels_map = {x:i for x,i in query_to_vg_rels_map.items() if i<self.max_relations_per_query}
        else:
            query_to_vg_objects_map = list(query_to_vg_objects_map)
            query_to_vg_attrs_map = list(query_to_vg_attrs_map)
            query_to_vg_rels_map = list(query_to_vg_rels_map)
            #query_to_vg_objects_map = list(set(query_to_vg_objects_map).intersection(question_concepts))
            #query_to_vg_attrs_map = list(set(query_to_vg_attrs_map).intersection(question_concepts))
            #query_to_vg_rels_map = list(set(query_to_vg_rels_map).intersection(question_concepts))
            query_to_vg_objects_map = utils.pad_or_clip(query_to_vg_objects_map, '<NULL>', self.max_objects_per_query)
            query_to_vg_attrs_map = utils.pad_or_clip(query_to_vg_attrs_map, '<NULL>', self.max_attributes_per_query)
            query_to_vg_rels_map = utils.pad_or_clip(query_to_vg_rels_map, '<NULL>', self.max_relations_per_query)
            q_objects = [self.query_objects_vocab[x] for x in query_to_vg_objects_map]
            q_attributes = [self.query_attributes_vocab[x] for x in query_to_vg_attrs_map]
            q_relations = [self.query_relations_vocab[x] for x in query_to_vg_rels_map]
            query_to_vg_objects_map = {x:i for i,x in enumerate(query_to_vg_objects_map) if x!='<NULL>'}
            query_to_vg_attrs_map = {x:i for i,x in enumerate(query_to_vg_attrs_map) if x!='<NULL>'}
            query_to_vg_rels_map = {x:i for i,x in enumerate(query_to_vg_rels_map) if x!='<NULL>'} 
        empty_bbox = {'attribute_distribution': np.zeros((self.query_concepts_embed_dim), dtype=np.float32), 'object_distribution': np.zeros((self.query_concepts_embed_dim), dtype=np.float32)}
        programs = []
        program = self.rule_based_program_generator.rule_based_program(question, question_category, query_to_vg_objects_map, query_to_vg_attrs_map, query_to_vg_rels_map)
        program = program[0]
        program_dict = program.to_dict()
        operator_type_sequence = utils.pad_or_clip(program_dict['operator_type_sequence'], self.rule_based_program_generator.prog_config.pad_operator, self.max_program_length)
        argument_type_sequence = utils.pad_or_clip(program_dict['argument_type_sequence'], self.rule_based_program_generator.prog_config.pad_argtype, self.max_program_length)
        target_type_sequence = utils.pad_or_clip(program_dict['target_type_sequence'], self.rule_based_program_generator.prog_config.pad_target_type, self.max_program_length)
        argument_table_index_sequence = utils.pad_or_clip(program_dict['argument_table_index_sequence'], self.rule_based_program_generator.prog_config.pad_arg_table_index, self.max_program_length)
        sequence_length = self.max_program_length#program_dict['sequence_length']
        bbox = data['bbox']
        bbox = utils.pad_or_clip(bbox, empty_bbox, self.max_bbox_per_image)
        bbox_mask = self.get_bbox_mask(bbox, empty_bbox)
        bbox = self.add_bbox_object_masks(bbox)
        bbox = self.add_bbox_attribute_masks(bbox)
        bbox_object_masks = [bbox_i['object_masks'] for bbox_i in bbox]
        bbox_attribute_masks = [bbox_i['attribute_masks'] for bbox_i in bbox]
        #print ('bbox_mask ', bbox_mask, 'bbox_object_masks ', bbox_object_masks, 'bbox_attribute_masks', bbox_attribute_masks)
        bbox_objects_emb = [self.get_embedding(bbox_i['object_labels'], 'object') for bbox_i in bbox]
        bbox_attributes_emb = [self.get_embedding(bbox_i['attribute_labels'], 'attribute') for bbox_i in bbox]
        bbox_objects = [bbox_i['object_ids'] for bbox_i in bbox]
        bbox_attributes = [bbox_i['attribute_ids'] for bbox_i in bbox]
        scene = {}
        bbox_mask = np.asarray(bbox_mask, dtype=np.float32)
        bbox_attribute_masks = np.asarray(bbox_attribute_masks, dtype=np.float32)
        bbox_object_masks = np.asarray(bbox_object_masks, dtype=np.float32)
        bbox_attributes = np.asarray(bbox_attributes, dtype=np.int64)
        bbox_objects = np.asarray(bbox_objects, dtype=np.int64)
        bbox_objects_emb = np.asarray(bbox_objects_emb, dtype=np.float32)
        bbox_attributes_emb = np.asarray(bbox_attributes_emb, dtype=np.float32)
        q_attributes = np.asarray(q_attributes, dtype=np.int64)
        q_objects = np.asarray(q_objects, dtype=np.int64)
        q_relations = np.asarray(q_relations, dtype=np.int64)
        q_encoded = np.asarray(q_encoded, dtype=np.int64)
        a_encoded = np.asarray(a_encoded, dtype=np.int64)
        a_types = np.asarray(a_types, dtype=np.int64)
        return q_encoded, a_encoded, a_types, q_attributes, q_objects, q_relations, bbox_mask, bbox_attribute_masks, bbox_attributes, bbox_attributes_emb, bbox_object_masks, bbox_objects, bbox_objects_emb, operator_type_sequence, argument_type_sequence, target_type_sequence, argument_table_index_sequence, sequence_length
        #return q_data, scene, programs
       
    
