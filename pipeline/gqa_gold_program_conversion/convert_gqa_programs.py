#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:54:50 2019

@author: amrita
"""
import re
import json
import pickle as pkl
import os
import spacy
from pattern.en import singularize
from gqa_vocab.gqa_concept_vocab import GQAConceptVocabulary

class ConvertGQAProgramToCustom():
    
    def __init__(self, opt):
        self.gqa_concept_vocab = GQAConceptVocabulary(opt)
        self.gqa_objects = self.gqa_concept_vocab.load_gqa_concepts('objects')
        self.gqa_attributes = self.gqa_concept_vocab.load_gqa_concepts('attributes')
        self.gqa_relations = self.gqa_concept_vocab.load_gqa_concepts('relations')
        self.nlp = spacy.load("en_core_web_sm")    
        self.spatial_positions = ['top', 'bottom', 'left', 'right', 'center', 'middle']
        self.gender_pronouns = ['he', 'she', 'it', 'they', 'we', 'her', 'his']
        self.property_types = ["activity", "age", "appearance", "arrangement", "brightness", "cleanliness", "cleanliness", "clothes", "color", "company", "density", "depth", "digital", "face expression", "fatness", "fertility", "fit", "flavor", "food", "fruit", "hardness", "height", "length", "liquid", "location", "material", "nature", "occasion", "opaqness", "orientation", "pattern", "place", "pose", "position", "quantity", "race", "realism", "room", "shape", "signal", "size", "sport", "sportActivity", "state", "strength", "temperature_state", "texture", "tone", "way of cut", "weather", "width", "gender", "weight", "direction", "object", "cake slice", "name", "-", "_", "hposition", "thickness", "this", "scene", "furniture", "fast food", "clothing", "watercraft", "type", "over"]
        self.simplifications  = {'dressed in':'wearing', 'looking for':'looking_for', 'looking at':'looking_at', 'looking':'looking_at', 'pointing':'pointing_at', 'in the middle of':'in_between', 'in the center of':'in_between', 'in mirror':'reflected_in', 'light blue':'light_blue'} 
        print ('finished loading gqa objects, relations, attributes')
        
    def get_argument_types(self, arguments):
        argtypes = []
        for arg in arguments:
            if '|' in arg:
                arguments_i = [self.get_argument_types([x]) for x in arg.split('|')]
                if all([x=='object' for x in arguments_i]):
                    argtype = ['object']*len(arguments_i)
                elif all([x=='relation' for x in arguments_i]):
                    argtype = ['relation']*len(arguments_i)
                elif all([x=='attribute' for x in arguments_i]):
                    argtype = ['attribute']*len(arguments_i)
                else:
                    argtype = arguments_i
            else:  
                arg_parsed = self.nlp(arg)
                arg_lemma = ' '.join([tok.lemma_ for tok in arg_parsed])
                arg_sing = ' '.join([singularize(tok.text) for tok in arg_parsed])
                if arg in self.gqa_objects or arg_lemma in self.gqa_objects or arg_sing in self.gqa_objects:
                    argtype = 'object'
                elif arg in self.gqa_attributes or arg_lemma in self.gqa_attributes or arg_sing in self.gqa_attributes:
                    argtype = 'attribute'
                elif arg in self.gqa_relations or arg_lemma in self.gqa_relations or arg_sing in self.gqa_relations:
                    argtype = 'relation'
                elif arg in ['s', 'o']:
                    argtype = 'scene'
                elif arg in self.spatial_positions:
                    argtype = 'spatial_positions'
                elif arg in self.gender_pronouns:
                    argtype = 'gender_pronouns'
                elif arg in self.property_types:
                    argtype = 'property_type'
                elif arg.startswith('not'):
                   arg = arg.replace('not', '', 1).strip()
                   argtype = self.get_argument_types([arg])         
                   if argtype != 'none':
                      if type(argtype)==list:
                          argtype = '_'.join(argtype)
                      argtype = 'not_'+argtype
                   else:
                      print ('cannot find arg ', arg)
                      argtype = 'none'
                elif arg.startswith('same '):
                    arg = arg.replace('same ', '', 1).strip()
                    argtype = self.get_argument_types([arg])
                    if argtype != 'none':
                       if type(argtype)==list:
                          argtype = '_'.join(argtype)
                       argtype = 'same_'+argtype   
                    else:
                       print ('cannot find arg ', arg)
                       argtype = 'none'
                else:
                    print ('cannot find arg ', arg)
                    argtype = 'none'
            if type(argtype) == 'list':
                argtypes.extend(argtype)
            else:    
                argtypes.append(argtype)
        return argtypes
               

    def postprocess_none_arguments(self, arguments, argument_types):
          new_args = []
          new_arg_types = []
          for arg, arg_type in zip(arguments, argument_types):
             arg_orig = arg
             if arg_type=='none':
                for k,v in self.simplifications.items():
                    if k in arg:
                      arg = arg.replace(k, v)
                      break
                if arg.isdigit():
                    continue
                types = []
                words = []
                last_type = None
                if arg in self.gqa_objects:
                    types = ['object']
                    words = [arg]
                elif arg in self.gqa_relations:
                    types = ['relation']
                    words = [arg]
                elif arg in self.gqa_attributes:
                    types = ['attribute']
                    words = [arg]
                else:
                  for word in arg.split(' '):
                    if word in self.gqa_objects or word.replace('_',' ') in self.gqa_objects:
                       word = word.replace('_',' ')
                       if last_type=='object':
                          words[-1] = words[-1] +' '+word
                       else:
                          types.append('object')
                          words.append(word)
                       last_type = 'object'
                    elif word in self.gqa_relations or word.replace('_',' ') in self.gqa_relations:
                       word = word.replace('_',' ')
                       if last_type=='relation':
                          words[-1] = words[-1] +' '+word
                       else:
                          types.append('relation')
                          words.append(word)
                       last_type = 'relation'
                    elif word in self.gqa_attributes or word.replace('_',' ') in self.gqa_attributes:
                       word = word.replace('_',' ')
                       if last_type=='attribute':
                          words[-1] = words[-1]+' '+word
                       else:
                          types.append('attribute')
                          words.append(word)
                       last_type = 'attribute'
                if len(types)>0:
                    print (arg_orig , '--->', arg)
                new_args.extend(words)
                new_arg_types.extend(types)
             else:
                new_args.append(arg)
                new_arg_types.append(arg_type)
          return new_args, new_arg_types		 

    def convert_program(self, program):
        converted_program = []
        converted_i = 0
        i_converted_i_map = {'none':'scene_none'}
        for line_i,line in enumerate(program):
            operation = line['operation']
            argument = [re.sub(r"[0-9\-]*\)", "", re.sub(r"\([0-9\-]*", "", re.sub(r"\([0-9\-]*\)", "", x))).strip() for x in line['argument'].replace('?','').split(',')]
            if '' in argument:
                argument.remove('')
            for x in argument:
                if x.isdigit() and len(x)>4:
                    argument.remove(x)
            argument_types = self.get_argument_types(argument)
            if len(line['dependencies'])==0:
                line['dependencies'] = ['none']
            if not all([x in i_converted_i_map for x in line['dependencies']]):
                print ('dependency not found in map')
                print ('i_converted_i_map', i_converted_i_map)
                print ('dependencies', line['dependencies'])
            if operation.endswith("rel") or operation=='relate':
                converted_operation = operation
                converted_arguments = []
                converted_argument_types = []
                dependencies = line['dependencies']
                for i in range(len(argument)):
                    arg = argument[i]
                    arg_type = argument_types[i]
                    if arg=='s' or arg=='o':
                        converted_arguments.extend([i_converted_i_map[d] for d in dependencies])
                        converted_argument_types.extend(['scene']*len(dependencies))
                    else:
                        converted_arguments.append(arg)
                        converted_argument_types.append(arg_type)
                if operation.startswith('verify'):
                    converted_operation = 'verify_rel'
                elif operation.startswith('choose'):
                    converted_operation = 'choose_rel'
            else:        
                if operation == 'select':
                    converted_operation = "filter_object"
                    dependencies = line['dependencies']
                    converted_arguments = [i_converted_i_map[d] for d in dependencies]+argument
                    converted_argument_types = ['scene']*len(dependencies)+argument_types
                elif operation.startswith("filter"):
                    if set(argument_types)==set(['attribute']):
                        converted_operation = "filter_attribute"
                    elif set(argument_types)==set(['object']):
                        converted_operation = "filter_object"
                    property_type = '_'.join(operation.split(' ')[1:])
                    if len(property_type)=='':
                        property_type = None
                    dependencies = line['dependencies']
                    if property_type:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies]+[property_type] + argument
                        converted_argument_types = ['scene']*len(dependencies)+['property_type']+argument_types
                    else:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies] + argument
                        converted_argument_types = ['scene']*len(dependencies)+argument_types
                elif operation.startswith("choose"):
                    converted_operation = "choose"
                    property_type = '_'.join(operation.split(' ')[1:])
                    if len(property_type)=='':
                        property_type = None
                    dependencies = line['dependencies']
                    if property_type:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies]+[property_type] + argument
                        converted_argument_types = ['scene']*len(dependencies)+['property_type']+argument_types
                    else:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies] + argument
                        converted_argument_types = ['scene']*len(dependencies)+argument_types
                elif operation.startswith("verify"):
                    converted_operation = "verify"
                    property_type = '_'.join(operation.split(' ')[1:])
                    if len(property_type)=='':
                        property_type = None
                    dependencies = line['dependencies']
                    if property_type:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies]+[property_type] + argument
                        converted_argument_types = ['scene']*len(dependencies)+['property_type']+argument_types
                    else:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies] + argument
                        converted_argument_types = ['scene']*len(dependencies)+argument_types
                else:
                    converted_operation = operation
                    property_type = '_'.join(operation.split(' ')[1:])
                    if len(property_type)=='':
                        property_type = None
                    dependencies = line['dependencies']
                    if property_type:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies]+[property_type] + argument
                        converted_argument_types = ['scene']*len(dependencies)+['property_type']+argument_types
                    else:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies] + argument
                        converted_argument_types = ['scene']*len(dependencies)+argument_types
            converted_arguments, converted_argument_types = self.postprocess_none_arguments(converted_arguments, converted_argument_types)
            sorted_converted_argument_types_inds = np.argsort(np.asarray(converted_argument_types))
            sorted_converted_argument_types = [converted_argument_types[i] for i in sorted_converted_argument_types_inds]
            sorted_converted_arguments = [converted_arguments[i] for i in sorted_converted_argument_types_inds]
            converted_output = {'operation':converted_operation, 'arguments': sorted_converted_arguments, 'argument_types': sorted_converted_argument_types}
            converted_program.append(converted_output)
            i_converted_i_map[line_i] = 'scene_'+str(converted_i)
            converted_i += 1
        return converted_program

if __name__=="__main__":
    program_converter = ConvertGQAProgramToCustom()
    new_data = program_converter.convert_all_programs()            
    pkl.dump(new_data, open('querytype_query_converted_programs_list_1.pkl', 'wb'))        
    
    
