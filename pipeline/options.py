#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:54:41 2019

@author: amrita
"""
import argparse
import os

class Options():
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--sceneparser_dataset_type', default='visual_genome', type=str, choices=['gqa', 'visual_genome'], help='choice of dataset for scene parsing')
        self.parser.add_argument('--qa_dataset_type', default='vqa', type=str, choices=['gqa', 'vqa'], help='choice of dataset for question answering')
        self.parser.add_argument('--visual_genome_dir', default='/dccstor/cssblr/amrita/VisualGenome', type=str, help='directory containing visual genome')
        self.parser.add_argument('--vqa_dir', default='/dccstor/cssblr/amrita/VQA', type=str, help='directory containing vqa')
        self.parser.add_argument('--gqa_dir', default='/dccstor/cssblr/amrita/GQA', type=str, help='directory containing gqa')
        self.parser.add_argument('--coco_dir', default='/dccstor/cssblr/amrita/coco', type=str, help='directory containing coco')
        self.parser.add_argument('--concepts_catalog_dir', default='/dccstor/cssblr/amrita/Concept_Catalog_VQA', type=str, help='directory containing catalog models')
        self.parser.add_argument('--attribute_catalog_preprocessed_dir', default='multiclass_unilabel_cce_attribute/preprocessed_data/concepts/', type=str, help='directory containing attribute catalog preprocessed data')
        self.parser.add_argument('--object_catalog_preprocessed_dir', default='multiclass_unilabel_cce_synset/preprocessed_data/concepts/', type=str, help='directory containing object catalog preprocessed data')
        self.parser.add_argument('--attribute_catalog_model_dir', default='multiclass_unilabel_cce_attribute/checkpoints/concepts/', type=str, help='directory containing attribute models')
        self.parser.add_argument('--object_catalog_model_dir', default='multiclass_unilabel_cce_synset/checkpoints/concepts/', type=str, help='directory containing object models')
        self.parser.add_argument('--preprocessed_dump_dir', default='/dccstor/cssblr/amrita/NSVQA_Pipeline/preprocessed_data/', type=str, help='directory for dumping preprocessed data')
        self.parser.add_argument('--mask_rcnn_dir', default='/dccstor/cssblr/amrita/Mask_RCNN', type=str, help='directory containing mask rcnn')
        self.parser.add_argument('--glove_clustering_dir', default='/dccstor/cssblr/amrita/GloVe_Clustering', type=str, help='directory containing glove clustering')
        self.parser.add_argument('--word2vec_googlenews', default='/dccstor/cssblr/ansarigh/DCL/data/clevr/generic/GoogleNews-vectors-negative300.bin', type=str, help='path to word2vec bin file')
        self.parser.add_argument('--vg_attr_types_file', default='attribute', type=str, help='file containing list of visual genome attributes')
        self.parser.add_argument('--vg_object_types_file', default='object', type=str, help='file containing list of visual genome objects')
        self.parser.add_argument('--vg_rel_types_file', default='relationship', type=str, help='file containing list of visual genome relations')
        self.parser.add_argument('--preprocessed_annotation_file', default='preprocessed_annotation.pkl', type=str, help='file containing preprocessed annotation dump')
        self.parser.add_argument('--program_annotation_file', default='program_annotation.pkl', type=str, help='file containing program annotation')

        self.parser.add_argument('--coco_year', default='2014', type=str, choices=['2014', '2017'], help='version of the coco dataset based on the year')
        self.parser.add_argument('--coco_split', default='train', type=str, choices=['test', 'train', 'val'], help='data split of the coco dataset')
        self.parser.add_argument('--gqa_type', default='balanced', type=str, choices=['all', 'balanced'], help='type of gqa dataset')
        self.parser.add_argument('--gqa_split', default='train', type=str, choices=['train', 'val'], help='data split of the gqa dataset')         
        self.parser.add_argument('--max_query_concepts', default=10, type=int, help='number of query concepts')
        self.parser.add_argument('--query_concepts_embed_dim', default=100, type=int, help='dimension of the query concepts embedding')
        self.parser.add_argument('--gpu_ids', default='0', type=str, help='ids of gpu to be used')

        self.parser.add_argument('--bbox_detection_type', default='gold', type=str, choices=['mask_rcnn', 'gold'], help='which kind of bbox detection to use')
        self.parser.add_argument('--object_detection_type', default='gold', type=str, choices=['motifnet', 'catalog', 'gold'], help='which kind of object detection to use')
        self.parser.add_argument('--attribute_detection_type', default='gold', type=str, choices=['motifnet', 'catalog', 'gold'], help='which kind of attribute detection to use')
        self.parser.add_argument('--vqa_dataset', default='vg_intersection_coco', type=str, choices=['coco', 'vg_intersection_coco'], help='the kind of dataset on which to do visual question answering')
        self.parser.add_argument('--gqa_dataset', default='gqa', type=str, choices=['gqa'], help='the kind of dataset on which to do visual question answering')
        self.parser.add_argument('--vg_mapping_in_query', default=0, type=int, help='whether the query objects, relations, attributes should be mapped to the visual genome vocabulary or not') 
        self.parser.add_argument('--sort_data_by_image', default=1, type=int, help='whether the data needs to be sorted by images')
        self.parser.add_argument('--input_vocab_json', type=str, help='name of the query,answer,program vocab file to be dumped')
        self.parser.add_argument('--expand_vocab', default=1, type=int, help='option (0/1) whether query,answer,program vocab should be expanded')
        self.parser.add_argument('--unk_threshold', default=1, type=int, help='frequency threshold of a token to be treated as UNK in vocabulary')
        self.parser.add_argument('--max_words_question', default=20, type=int, help='maximum number of words per question')
        self.parser.add_argument('--encode_unk', default=1, type=int, help='option whether to encode UNK in vocabulary or not')
        self.parser.add_argument('--max_answers', default=10, type=int, help='maximum number of answers considered per question')
        self.parser.add_argument('--max_words_answer', default=5, type=int, help='maximum number of words per answer')
        self.parser.add_argument('--max_types_answer', default=2, type=int, help='maximum number of types per answer')
        self.parser.add_argument('--output_h5_file', type=str, help='name of the output h5 file')
        self.parser.add_argument('--output_vocab_json', type=str, help='name of the query,answer,program vocab file to be read')
        self.parser.add_argument('--verbose',type=int, default=1, help='whether the execution should be in a verbose manner or not ')
        self.parser.add_argument('--datasize', type=str, default='toy', choices=['full', 'toy'], help='whether to use toy data for debugging or full data')      
    def parse(self):
        if not self.initialized:
            self.initialize()
            
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        for k, v in args.items():
            print ('%s: %s' % (str(k), str(v)))
        return self.opt

def get_options_parser():
    opt = Options()
    if not opt.initialized:
        opt.initialize()
    return opt.parser

def get_options():
    return Options().parse()

def get_option_str(opt):
    bbox_detection_type = opt.bbox_detection_type
    object_detection_type = opt.object_detection_type
    attribute_detection_type = opt.attribute_detection_type
    qvgmap = str(bool(opt.vg_mapping_in_query))
    sceneparser_dataset_type = opt.sceneparser_dataset_type
    qa_dataset_type = opt.qa_dataset_type
    datasize = opt.datasize
    if sceneparser_dataset_type == 'visual_genome' and qa_dataset_type == 'vqa':
        coco_year = opt.coco_year
        coco_split = opt.coco_split
        vqa_dataset = opt.vqa_dataset
        string = 'cyear_'+coco_year+'_csplit_'+coco_split+'_bbox_'+bbox_detection_type+'_obj_'+object_detection_type+'_attr_'+attribute_detection_type+'_dataset_'+vqa_dataset+'_qvgmap_'+qvgmap
    elif sceneparser_dataset_type == 'gqa' and qa_dataset_type == 'gqa':
        gqa_type = opt.gqa_type
        gqa_split = opt.gqa_split
        gqa_dataset = opt.gqa_dataset 
        string = 'gtype_'+gqa_type+'_gsplit_'+gqa_split+'_bbox_'+bbox_detection_type+'_obj_'+object_detection_type+'_attr_'+attribute_detection_type+'_dataset_'+gqa_dataset+'_qvgmap_'+qvgmap
    string = string+'_datasize_'+datasize 
    return string
