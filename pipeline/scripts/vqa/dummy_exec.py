#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:09:04 2019

@author: amrita
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.vqa_sceneparser_dataset import VQA_SceneParser_Dataset
from options import get_options, get_option_str
from options_executor import get_options_exec
import sys
sys.path.append('../../Dynamic-Concept-Learner')
from program_induction.executors import _get_executor as get_exec
import os

def get_dataset(opt, opt_exec):
    ds = VQA_SceneParser_Dataset(opt, opt_exec)
    return ds

def get_dataloader(ds, opt):
    loader = DataLoader(dataset=ds, batch_size=5, shuffle=False)
    return loader

opt = get_options()
opt_exec = get_options_exec()
dataset = get_dataset(opt, opt_exec)
loader = get_dataloader(dataset,opt)

datadump_folder = os.path.join(os.path.join(opt.preprocessed_dump_dir, 'vqa'), get_option_str(opt))  

params = {
      'vocab_json': os.path.join(datadump_folder, 'vocab.json'),
      'device': torch.device('cpu'),
      'path_op_arg_map': 'operator_map.json',
      'count_threshold' : 0.5,
      'ex_dropout_prob' : 0.1,
      'obj_att_list_size' : 5   
}
print ('GPU available? ', torch.cuda.is_available())
exect = get_exec(**params)
print ('Length of dataset ', len(dataset))
batch_id = 0
for q_encoded, a_encoded, a_types, q_attributes, q_objects, q_relations, bbox_mask, bbox_attribute_masks, bbox_attributes, bbox_attribute_emb, bbox_object_masks, bbox_objects, bbox_object_emb, operator_type_sequence, argument_type_sequence, target_type_sequence, argument_table_index_sequence, sequence_length in loader:
    if bbox_mask is None and bbox_attribute_masks is None and bbox_object_masks is None:
          break
    ''' 
    print ('Printing All Shapes: ')
    print ('query_encoded', np.asarray(q_encoded).shape)
    print ('query_attributes', np.asarray(q_attributes).shape)
    print ('query objects', np.asarray(q_objects).shape) 
    print ('bbox mask', np.asarray(bbox_mask).shape)
    print ('bbox_attribute_masks', np.asarray(bbox_attribute_masks).shape)
    print ('bbox_object_masks', np.asarray(bbox_object_masks).shape)
    print ('bbox_attributes', np.asarray(bbox_attributes).shape)
    print ('bbox_objects', np.asarray(bbox_objects).shape)
    print ('bbox_attribute_emb', np.asarray(bbox_attribute_emb).shape)
    print ('bbox_object_emb', np.asarray(bbox_object_emb).shape)
    print ('answer_encoded', np.asarray(a_encoded).shape)
    print ('answer_types', np.asarray(a_types).shape)
    '''
    bbox_mask = torch.tensor(bbox_mask, dtype=torch.float32)
    bbox_attributes = torch.tensor(bbox_attributes, dtype=torch.float32)
    bbox_attribute_masks = torch.tensor(bbox_attribute_masks, dtype=torch.float32)
    bbox_attribute_emb = torch.tensor(bbox_attribute_emb, dtype=torch.float32)
    bbox_objects = torch.tensor(bbox_objects, dtype=torch.float32)
    bbox_object_masks = torch.tensor(bbox_object_masks, dtype=torch.float32)
    bbox_object_emb = torch.tensor(bbox_object_emb, dtype=torch.float32)
   
    #output = exect.run(q_attribute_embeds, q_object_embeds, bbox_mask, bbox_attribute_masks, bbox_attribute_emb, bbox_object_masks, bbox_object_emb, operator_type_sequence, argument_type_sequence, target_type_sequence, argument_table_index_sequence, sequence_length)
    #print ('Output: ', output)
    batch_id += 1
    
