import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
import os

import h5py
import numpy as np
import gensim
import pickle as pkl
from collections import defaultdict
from options import get_options, get_option_str, get_option_parser

"""
Script for getting pretrained word2vec embedding matrices for vqa question files.
"""

opt = get_options()
def parse_arguments():
    d = {}
    d['preprocessed_dir'] = os.path.join(os.path.join(opt.preprocessed_dump_dir, 'vqa'), get_option_str(opt))
    d['input_question_vocab_json_file'] = os.path.join(preprocessed_dir, 'vocab.json')
    d['output_question_embedding_matrix'] = os.path.join(preprocessed_dir, 'embed_q.pkl')
    d['output_answer_embedding_matrix'] = os.path.join(preprocessed_dir, 'embed_answer.pkl')
    d['input_attribute_vocab_json'] = os.path.join(preprocessed_dir, 'attribute_vocab.pkl')
    d['output_attribute_embedding_matrix'] = os.path.join(preprocessed_dir, 'embed_att.pkl')
    d['output_query_attribute_embedding_matrix'] = os.path.join(preprocessed_dir, 'embed_q_att.pkl')
    d['input_type_vocab_json'] = os.path.join(preprocessed_dir, 'object_vocab.pkl')
    d['output_type_embedding_matrix'] = os.path.join(preprocessed_dir, 'embed_type.pkl')
    d['output_query_type_embedding_matrix'] = os.path.join(preprocessed_dir, 'embed_q_type.pkl')
    d['output_query_relation_embedding_matrix'] = os.path.join(preprocessed_dir, 'embed_q_rel.pkl')
    d['path_to_word2vec'] = opt.word2vec_googlenews
    d['qvg_map'] = bool(opt.vg_mapping_in_query)
    return d 


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
    return vocab


def main(args):
    q_vocab = load_vocab(args['input_question_vocab_json'])
    att_vocab = pkl.load(open(args['input_attribute_vocab_json'], 'rb'))
    type_vocab = pkl.load(open(args['input_type_vocab_json'], 'rb'))
    
    att_id2token = defaultdict(list)
    type_id2token = defaultdict(list)
   
    for token in att_vocab:
        id = att_vocab[token]
        att_id2token[id].append(token)
    
    for token in type_vocab:
        id = type_vocab[token]
        type_id2token[id].append(token)
    
    if not args['qvg_map']:
        q_att_vocab = q_vocab['query_attributes_token_to_idx']
        q_type_vocab = q_vocab['query_objects_token_to_idx']
        q_rel_vocab = q_vocab['query_relations_token_to_idx']
        
        q_att_id2token = defaultdict(list)
        q_type_id2token = defaultdict(list)
        q_rel_id2token = defaultdict(list)
        
        for token in q_att_vocab:
            id = q_att_vocab[token]
            q_att_id2token[id].append(token)
            
        for token in q_type_vocab:
            id = q_type_vocab[token]
            q_type_id2token[id].append(token)
            
        for token in q_rel_vocab:
            id = q_rel_vocab[token]
            q_rel_id2token[id].append(token)
            
    # aggregating all tokens
    global_tokens = set(q_vocab['question_token_to_idx'].keys())
    global_tokens.update(set(q_vocab['answer_token_to_idx'].keys()))
    global_tokens.update(set(att_vocab.keys()))
    global_tokens.update(set(type_vocab.keys()))
    if not qvg_map:
        global_tokens.update(set(q_att_vocab.keys()))
        global_tokens.update(set(q_type_vocab.keys()))
        global_tokens.update(set(q_rel_vocab.keys()))
    print('| Loading gensim model')
    model = gensim.models.KeyedVectors.load_word2vec_format(args['path_to_word2vec'], binary=True)

    global_out = {}
    out_q = np.zeros([len(q_vocab['question_token_to_idx']), model.vector_size])
    out_a = np.zeros([len(q_vocab['answer_token_to_idx']), model.vector_size])
    out_att = np.zeros([len(att_id2token), model.vector_size])
    out_type = np.zeros([len(type_id2token), model.vector_size])
    if not qvg_map:
        out_q_att = np.zeros([len(q_att_id2token), model.vector_size])
        out_q_type = np.zeros([len(q_type_id2token), model.vector_size])
        out_q_rel = np.zeros([len(q_rel_id2token), model.vector_size])
    
    for token in global_tokens:
        if token in model:
            vec = model[token]
        else:
            vec = np.random.normal(0,1,model.vector_size)        
        global_out[token] = vec

    print('| processing question vocab')    
    for token in q_vocab['question_token_to_idx']:
        out_q[q_vocab['question_token_to_idx'][token]] = global_out[token]
    print('| processing answer vocab')    
    for token in q_vocab['answer_token_to_idx']:
        out_a[q_vocab['answer_token_to_idx'][token]] = global_out[token]
    
    # here we will take mean of possible token vocabularies
    print('| processing attribute vocab')
    for id in att_id2token:
        tokens = att_id2token[id]
        valid_tokens = [tok for tok in tokens if tok in model]
        if len(valid_tokens)>0:
            mat = np.stack([np.array(global_out[tok]) for tok in valid_tokens])
            out_att[id] = np.mean(mat,0)
        else:
            for tok in tokens:
                if tok in global_tokens:
                    out_att[id] = global_out[tok]
                    break
            out_att[id] = np.random.normal(0,1,model.vector_size)
        
    print('| processing type vocab')   
    for id in type_id2token:
        tokens = type_id2token[id]
        valid_tokens = [tok for tok in tokens if tok in model]
        if len(valid_tokens)>0:
            mat = np.stack([np.array(global_out[tok]) for tok in valid_tokens])
            out_type[id] = np.mean(mat,0)
        else:
            for tok in tokens:
                if tok in global_tokens:
                    out_type[id] = global_out[tok]
                    break
            out_type[id] = np.random.normal(0,1,model.vector_size)
            
    if not qvg_map: 
        print('| processing query attribute vocab')
        for id in q_att_id2token:
            tokens = q_att_id2token[id]
            valid_tokens = [tok for tok in tokens if tok in model]
            if len(valid_tokens)>0:
                mat = np.stack([np.array(global_out[tok]) for tok in valid_tokens])
                out_q_att[id] = np.mean(mat,0)
            else:
                for tok in tokens:
                    if tok in global_tokens:
                        out_q_att[id] = global_out[tok]
                        break
                out_q_att[id] = np.random.normal(0,1,model.vector_size)
            
        print('| processing query type vocab')   
        for id in q_type_id2token:
            tokens = q_type_id2token[id]
            valid_tokens = [tok for tok in tokens if tok in model]
            if len(valid_tokens)>0:
                mat = np.stack([np.array(global_out[tok]) for tok in valid_tokens])
                out_q_type[id] = np.mean(mat,0)
            else:
                for tok in tokens:
                    if tok in global_tokens:
                        out_q_type[id] = global_out[tok]
                        break
                out_q_type[id] = np.random.normal(0,1,model.vector_size)
                
        print('| processing query rel vocab')   
        for id in q_rel_id2token:
            tokens = q_rel_id2token[id]
            valid_tokens = [tok for tok in tokens if tok in model]
            if len(valid_tokens)>0:
                mat = np.stack([np.array(global_out[tok]) for tok in valid_tokens])
                out_q_rel[id] = np.mean(mat,0)
            else:
                for tok in tokens:
                    if tok in global_tokens:
                        out_q_rel[id] = global_out[tok]
                        break
                out_q_rel[id] = np.random.normal(0,1,model.vector_size)
            
    print('| saving stuff')
    with open(args['output_question_embedding_matrix'],'wb') as fp:
        pkl.dump(out_q, fp)
    with open(args['output_answer_embedding_matrix'],'wb') as fp:
        pkl.dump(out_a, fp)        
    with open(args['output_attribute_embedding_matrix'],'wb') as fp:
        pkl.dump(out_att, fp)
    with open(args['output_type_embedding_matrix'],'wb') as fp:
        pkl.dump(out_type, fp)
    
    if not qvg_map:
        with open(args['output_query_attribute_embedding_matrix'], 'wb') as fp:
            pkl.dump(out_q_att, fp)
        with open(args['output_query_type_embedding_matrix'], 'wb') as fp:
            pkl.dump(out_q_type, fp)
        with open(args['output_query_relation_embedding_matrix'], 'wb') as fp:
            pkl.dump(out_q_rel, fp)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    args = vars(args)
    print('| options')
    for k, v in args.items():
        print('%s: %s' % (str(k), str(v)))
    print('| Processing completed successfully')

