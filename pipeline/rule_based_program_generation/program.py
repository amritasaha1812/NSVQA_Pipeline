#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 20:19:59 2019

@author: amrita
"""
import json
from utils.utils import pad_to_max

class ProgramConfig():
    
    def __init__(self, operator_map_file="operator_map.json", disable_relations=False):
        self.operator_prototype, self.op_to_id, self.argtype_to_id = self.read_operator_map(operator_map_file, disable_relations)
        self.id_to_op = {v:k for k,v in self.op_to_id.items()}
        self.id_to_argtype = {v:k for k,v in self.argtype_to_id.items()}
        self.pad_operator = self.op_to_id['NONE']
        self.pad_argtype = [self.argtype_to_id['NONE']]*self.max_num_arguments
        self.pad_target_type = self.argtype_to_id['NONE']
        self.pad_arg_table_index = [0]*self.max_num_arguments
        self.max_program_length = 5
        
    def read_operator_map(self, file, disable_relations):
        print ('going to read ', file)
        ops = json.load(open(file))
        print ('read ', file)
        prototype = {k:v for k,v in ops['Prototype'].items() if not disable_relations or 'relation' not in k}
        op_to_id = {k:v for k,v in ops['OpToId'].items() if not disable_relations or 'relation' not in k}
        argtype_to_id = {k:v for k,v in ops['ArgumentTypetoId'].items() if not disable_relations or 'relation' not in k}
        num_arguments = [len(x["input"]) for x in prototype.values()]
        self.max_num_arguments = max(num_arguments)
        print ('self.max_num_arguments ', self.max_num_arguments)
        prototype = {k:{"input":pad_to_max(v["input"], "NONE", self.max_num_arguments), "output":v["output"]} for k,v in prototype.items()}
        return prototype, op_to_id, argtype_to_id

    
class Program():

    def __init__(self, prog_config, object_memory, attribute_memory, relation_memory, disable_relations):
        self.prog_config = prog_config
        self.disable_relations = disable_relations
        self.operator = []
        self.operator_types = []
        self.argument_types = []
        self.target_types = []
        self.arg_table_indices = []
        self.arg_values = []
        self.target_table_indices = []
        self.object_memory = object_memory
        self.attribute_memory = attribute_memory
        self.relation_memory = relation_memory
        self.memory = {}
        self.memory[self.prog_config.argtype_to_id['object']] = object_memory
        self.memory[self.prog_config.argtype_to_id['attribute']] = attribute_memory
        if not self.disable_relations:
            self.memory[self.prog_config.argtype_to_id['relation']] = relation_memory
        for k,v in self.prog_config.argtype_to_id.items():
            if v not in self.memory:
                self.memory[v] = []
        self.memory[self.prog_config.argtype_to_id['scene']] = ["scene1"]    
        self.memory[self.prog_config.argtype_to_id['NONE']] = ["NONE"]
    def get_operator_type(self, operator):
        return self.prog_config.op_to_id[operator]
    
    def get_argument_type(self, operator):
        return [self.prog_config.argtype_to_id[x] for x in self.prog_config.operator_prototype[operator]["input"]]
    
    def get_target_type(self, operator):
        return self.prog_config.argtype_to_id[self.prog_config.operator_prototype[operator]["output"]]
    
    def add_line_of_code(self, operator, arg_table_index, value_inputs):
        operator_type = self.get_operator_type(operator)
        argument_type = self.get_argument_type(operator)
        target_type = self.get_target_type(operator)
        self.operator.append(operator)
        self.operator_types.append(operator_type)
        self.argument_types.append(argument_type)
        self.target_types.append(target_type)
        self.arg_values.append(value_inputs)
        target_table_index = len(self.memory[target_type])
        self.memory[target_type].append(self.prog_config.id_to_argtype[target_type]+str(len(self.memory[target_type])+1))
        arg_table_index = pad_to_max(arg_table_index, 0, self.prog_config.max_num_arguments)
        if len(arg_table_index)>self.prog_config.max_num_arguments:
            print ('(arg_table_index)', arg_table_index)
            raise Exception('len(arg_table_index)>self.prog_config.max_num_arguments')
        #print (self.prog_config.id_to_op[operator_type],'--->', [self.prog_config.id_to_argtype[x]+'('+str(y)+')' for x,y in zip(argument_type, arg_table_index)])
        self.arg_table_indices.append(arg_table_index)
        self.target_table_indices.append(target_table_index)

    def print_loc(self, i):
        operator = self.prog_config.id_to_op[self.operator_types[i]]
        #print ('operator ', operator, 'self.argument_types ', self.argument_types, 'self.arg_table_indices ', self.arg_table_indices)
        arguments = [str(self.memory[self.argument_types[i][j]][self.arg_table_indices[i][j]]) for j in range(len(self.argument_types[i]))]
        target = self.memory[self.target_types[i]][self.target_table_indices[i]]
        return (target+ ' = '+operator+'('+', '.join(arguments)+')')
        
    def print_program(self):
        program_length = len(self.operator_types)
        program = ''
        for i in range(program_length):
            program += self.print_loc(i).strip()+'\n'
        return program
    
    def to_str(self):
        return ' '.join([self.operator[i]+' '+','.join(self.arg_values[i]) for i in range(len(self.operator))])

    def to_dict(self):
        d = {}
        d['operator_type_sequence'] = self.operator_types
        d['argument_type_sequence'] = self.argument_types
        d['target_type_sequence'] = self.target_types
        d['argument_table_index_sequence'] = self.arg_table_indices
        d['sequence_length'] = len(self.operator)
        return d

