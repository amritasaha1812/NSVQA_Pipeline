#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:16:27 2019

@author: amrita
"""
from scene_parsing.query_cleaning import QueryCleaning
from scene_parsing.answer_cleaning import AnswerCleaning
from evaluation.vqa.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
import spacy

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for preprocessing sequence data.
Special tokens that are in all dictionaries:
<NULL>: Extra parts of the sequence that we should ignore
<START>: Goes at the start of a sequence
<END>: Goes at the end of a sequence, before <NULL> tokens
<UNK>: Out-of-vocabulary words
"""

SPECIAL_TOKENS = [
  '<NULL>',
  '<UNK>',
  '<START>',
  '<END>']
query_cleaning = QueryCleaning()
answer_cleaning = AnswerCleaning()
vqa_eval = VQAEval(None, None)

def tokenize(s, type_of_s, lemmatize, add_start_token, add_end_token):
  s = s.replace('\n', ' ').replace('\t', ' ').strip()
  if type_of_s == 'answer':
     s = s.lower()
     s = vqa_eval.processPunctuation(s)
     s = vqa_eval.processDigitArticle(s)
  if type_of_s == 'query':
     _, tokens, pos, lemma, _ = query_cleaning.execute(s)
  elif type_of_s == 'answer':
     _, tokens, pos, lemma, _ = answer_cleaning.execute(s)
  elif type_of_s == 'query_objects' or type_of_s == 'query_attributes' or type_of_s == 'query_relations':
     tokens = [s]
  elif type_of_s == 'program':
     tokens = s.split(' ')
  if lemmatize:
    tokens = lemma
  if add_start_token:
    tokens.insert(0, '<START>')
  if add_end_token:
    tokens.append('<END>')
  return tokens


def build_vocab(sequences, type_, lemmatize, add_start_token, add_end_token, min_token_count=1):
  token_to_count = {}
  for seq in sequences:
    seq_tokens = tokenize(seq, type_, lemmatize, add_start_token, add_end_token)
    for token in seq_tokens:
      if token not in token_to_count:
        token_to_count[token] = 0
      token_to_count[token] += 1

  token_to_idx = {}
  for token in SPECIAL_TOKENS:
    if not add_start_token and token=='<START>':
       continue
    if not add_end_token and token=='<END>':
       continue
    token_to_idx[token] = len(token_to_idx)
  for token, count in sorted(token_to_count.items()):
    if count >= min_token_count:
      token_to_idx[token] = len(token_to_idx)

  return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
  seq_idx = []
  for token in seq_tokens:
    if token not in token_to_idx:
      if allow_unk:
        token = '<UNK>'
      else:
        raise KeyError('Token "%s" not in vocab' % token)
    seq_idx.append(token_to_idx[token])
  return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
  tokens = []
  for idx in seq_idx:
    tokens.append(idx_to_token[idx])
    if stop_at_end and tokens[-1] == '<END>':
      break
  if delim is None:
    return tokens
  else:
    return delim.join(tokens)
