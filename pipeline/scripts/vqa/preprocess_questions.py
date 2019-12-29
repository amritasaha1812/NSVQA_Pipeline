#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:28:54 2019

@author: amrita
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse

import json
import pickle as pkl
import os

import h5py
import numpy as np

from preprocess.vqa.preprocess import tokenize, encode, build_vocab, SPECIAL_TOKENS
from rule_based_program_generation.program import Program
from utils.utils import pad_or_clip
from options import get_options, get_option_str, get_options_parser
from scene_parsing.answer_type_extractor import AnswerTypeExtractor
 
"""
Preprocessing script for VQA question files.
"""
opt = get_options()
def parse_arguments():
    preprocessed_dir = os.path.join(os.path.join(opt.preprocessed_dump_dir, 'vqa'), get_option_str(opt))
    if opt.input_vocab_json:
        opt.input_vocab_json = os.path.join(preprocessed_dir, opt.input_vocab_json)
    opt.input_preprocessed_data = os.path.join(preprocessed_dir, opt.preprocessed_annotation_file)
    if opt.output_h5_file:
        opt.output_h5_file = os.path.join(preprocessed_dir, opt.output_h5_file)
    if opt.output_vocab_json:
        opt.output_vocab_json = os.path.join(preprocessed_dir, opt.output_vocab_json)
    return opt


def main(args):
  if (args.input_vocab_json is None) and (args.output_vocab_json is None):
    print('Must give one of --input_vocab_json or --output_vocab_json')
    return

  print('Loading data')
  answer_type_extractor = AnswerTypeExtractor()
  with open(args.input_preprocessed_data, 'rb') as f:
    preprocessed_data = pkl.load(f, encoding='latin1')
  # Either create the vocab or load it from disk
  if args.input_vocab_json is None or args.expand_vocab == 1:
    print('Building vocab')
    questions = set([])
    answers = set([])
    query_objects = set([])
    query_attributes = set([])
    query_relations = set([])
    answer_types = {}
    for img_id in preprocessed_data:
       for k in range(len(preprocessed_data[img_id])):
          questions.add(preprocessed_data[img_id][k]['question'])
          answers.update([x['answer'] for x in preprocessed_data[img_id][k]['answers']])
          query_objects.update(preprocessed_data[img_id][k]['query_to_vg_objects_map'])
          query_attributes.update(preprocessed_data[img_id][k]['query_to_vg_attrs_map'])
          query_relations.update(preprocessed_data[img_id][k]['query_to_vg_rels_map'])
          for x in preprocessed_data[img_id][k]['answers']:
              answer_types.update(answer_type_extractor.get_answer_type_dict(x['answer']))
    question_token_to_idx = build_vocab(questions, 'query', lemmatize=True, add_start_token=True, add_end_token=True, min_token_count=args.unk_threshold)
    answer_token_to_idx = build_vocab(answers, 'answer', lemmatize=False, add_start_token=False, add_end_token=False)
    query_objects_token_to_idx = build_vocab(query_objects, 'query_objects', lemmatize=False, add_start_token=False, add_end_token=False, min_token_count=args.unk_threshold)
    query_attributes_token_to_idx = build_vocab(query_attributes, 'query_attributes', lemmatize=False, add_start_token=False, add_end_token=False, min_token_count=args.unk_threshold)
    query_relations_token_to_idx = build_vocab(query_relations, 'query_relations',  lemmatize=False, add_start_token=False, add_end_token=False, min_token_count=args.unk_threshold) 
    vocab = {
      'question_token_to_idx': question_token_to_idx,
      'query_objects_token_to_idx':query_objects_token_to_idx,
      'query_attributes_token_to_idx':query_attributes_token_to_idx,
      'query_relations_token_to_idx':query_relations_token_to_idx,
      'answer_token_to_idx': answer_token_to_idx,
      'answer_type_vocab': answer_types
    }


  if args.output_vocab_json is not None:
    with open(args.output_vocab_json, 'w') as f:
      json.dump(vocab, f)
  
  if not args.output_h5_file:
    return
  
  # Encode all questions and programs
  print('Encoding data')
  questions_encoded = []
  programs_encoded = []
  orig_idxs = []
  image_idxs = []
  answers = []
  orig_idx = 0
  for img_id in preprocessed_data:
       for k in range(len(preprocessed_data[img_id])):
          question = preprocessed_data[img_id][k]['question']
          orig_idxs.append(orig_idx)
          image_idxs.append(preprocessed_data[img_id][k]['image_id'])
          question_tokens = tokenize(question,'query', lemmatize=True, add_start_token=True, add_end_token=True)
          question_encoded = encode(question_tokens,
                         vocab['question_token_to_idx'],
                         allow_unk=args.encode_unk == 1)
          questions_encoded.append(question_encoded)
          if 'answers' in preprocessed_data[img_id][k]:
              answers_encoded = []
              for answer in preprocessed_data[img_id][k]['answers']:
                 answer_tokens = tokenize(answer['answer'], 'answer', lemmatize=False, add_start_token=False, add_end_token=False)
                 answer_encoded = encode(answer_tokens, vocab['answer_token_to_idx'])
                 answer_encoded = pad_or_clip(answer_encoded, SPECIAL_TOKENS.index('<NULL>'), args.max_words_answer)
              answers_encoded.append(answer_encoded)
          
              padding = [SPECIAL_TOKENS.index('<NULL>')]*args.max_words_answer
              answers_encoded = pad_or_clip(answers_encoded, padding, args.max_answers)
              answers.append(answers_encoded)
          orig_idx += 1 
  # Pad encoded questions and programs
  max_question_length = max(len(x) for x in questions_encoded)
  for qe in questions_encoded:
    while len(qe) < max_question_length:
      qe.append(vocab['question_token_to_idx']['<NULL>'])

  # Create h5 file
  print('Writing output')
  questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
  with h5py.File(args.output_h5_file, 'w') as f:
    f.create_dataset('questions', data=questions_encoded)
    f.create_dataset('image_idxs', data=np.asarray(image_idxs))
    f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))

    if len(answers) > 0:
      f.create_dataset('answers', data=np.asarray(answers))


if __name__ == '__main__':
  #opt = get_options()
  opt = parse_arguments()
  main(opt)
