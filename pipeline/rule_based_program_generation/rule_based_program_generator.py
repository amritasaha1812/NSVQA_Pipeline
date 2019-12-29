#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:06:29 2019

@author: amrita
"""
#import os
#import sys
#sys.path.insert(0, os.path.abspath('.'))
from scene_parsing.query_cleaning import QueryCleaning
from scene_parsing.query_concept_extractor import QueryConceptExtractor
from options import get_options
from .program import Program, ProgramConfig

class RuleBasedProgramGeneration():
    def __init__(self, opt, disable_relations=False):
        self.verbose = bool(opt.verbose)
        self.query_cleaning = QueryCleaning()
        print ('finished QueryCleaning:init')
        self.query_concept_extractor = QueryConceptExtractor(opt)
        print ('finished QueryConceptExtractor:init')
        self.middle_verb = set(['is', 'was', 'are', 'were', 'would', 'should', 'shall', 'do', 'did', 'does', 'has', 'have', 'kind', 'can', 'be', 'will', 'what', 'why', 'how'])
        print ('finished RuleBasedProgramGeneration:init')    
        self.disable_relations = disable_relations
        self.prog_config = ProgramConfig(disable_relations=disable_relations)
        
        
    def extract_from_segment(self, segment, segment_lemma, segment_pos, segment_descendants, query_objs, query_attrs, query_rels):
        objects, attributes, relations = self.query_concept_extractor.execute(segment, segment_lemma, segment_pos, segment_descendants)
        objects = [x for x in objects if x in query_objs]
        attributes = [x for x in attributes if x in query_attrs]
        relations = [x for x in relations if x in query_rels]
        return objects, attributes, relations
    
    def get_children_of_root(self, query_parsed):
        immediate_children = {token.text:token.children for token in query_parsed}
        children_of_root = [immediate_children[token.text] for token in query_parsed if token.dep_=='ROOT'][0]
        return children_of_root
    
    def parse_query(self, m, query_toks, query_lemma, query_pos, query_tok_descendants, children_of_root, query_to_vg_objects_map, query_to_vg_attrs_map, query_to_vg_rels_map):
        m_index = query_toks.index(m)
        first_toks = query_toks[:m_index]
        first = query_lemma[:m_index]
        first_pos = query_pos[:m_index]
        first_descendants = query_tok_descendants[:m_index]
        last_toks = query_toks[m_index+1:]
        last = query_lemma[m_index+1:]
        last_pos = query_pos[m_index+1:]
        last_descendants = query_tok_descendants[m_index+1:]
        objects_in_first = []
        attributes_in_first = []
        relations_in_first = []
        objects_in_last = []
        attributes_in_last = []
        relations_in_last = []
        first_toks_desc_map = {x:y for x,y in zip(first_toks, first_descendants)}
        last_toks_desc_map = {x:y for x,y in zip(last_toks, last_descendants)}
        if len(first)>0:
                objects_in_first, attributes_in_first, relations_in_first = self.extract_from_segment(first_toks, first, first_pos, first_descendants, query_to_vg_objects_map, query_to_vg_attrs_map, query_to_vg_rels_map)    
        if len(last)>0:
                objects_in_last, attributes_in_last, relations_in_last =  self.extract_from_segment(last_toks, last, last_pos, last_descendants, query_to_vg_objects_map, query_to_vg_attrs_map, query_to_vg_rels_map)
                target_object = []
                target_attribute = []
                target_relation = []
                if len(objects_in_first)==0 and len(attributes_in_first)==0 and len(relations_in_first)==0:
                        for c in children_of_root:
                                c = c.text
                                if c in objects_in_last and len(last_toks_desc_map[c])>0 and len(target_object)==0:
                                        objects_in_last.remove(c)
                                        target_object.append(c)
                                if c in attributes_in_last and len(last_toks_desc_map[c])>0 and len(target_attribute)==0:
                                        attributes_in_last.remove(c)
                                        target_attribute.append(c)
                                if c in relations_in_last and len(last_toks_desc_map[c])>0 and len(target_relation)==0:
                                        relations_in_last.remove(c)
                                        target_relation.append(c)
                else:
                        for x in objects_in_first:
                            if len(first_toks_desc_map[x])>0 and len(target_object)==0:
                               target_object.append(x)
                            else:
                               objects_in_last.append(x)
                        for x in attributes_in_first:
                            if len(first_toks_desc_map[x])>0 and len(target_attribute)==0:
                               target_attribute.append(x)
                            else:
                               attributes_in_last.append(x)
                        for x in relations_in_first:
                            if len(first_toks_desc_map[x])>0 and len(target_relation)==0:
                               target_relation.append(x)
                            else:
                               relations_in_last.append(x) 
        if len(target_relation)>0:
               objects_in_last.extend(target_object)
               attributes_in_last.extend(target_attribute)
               target_attribute = []
               target_object = []
        elif len(target_attribute)>0 and len(target_object)>0:
               attributes_in_last.extend(target_attribute)
               target_attribute = []
        return target_object, target_attribute, target_relation, objects_in_last, attributes_in_last, relations_in_last              
    
    
    def load_memories(self, objects_in_last, attributes_in_last, relations_in_last, target_object, target_attribute, target_relation, query_to_vg_objects_map, query_to_vg_attrs_map, query_to_vg_rels_map):
        
        object_memory = [-1]*len(query_to_vg_objects_map)
        for k,v in query_to_vg_objects_map.items():
            object_memory[v] = k
        attribute_memory = [-1]*len(query_to_vg_attrs_map)
        for k,v in query_to_vg_attrs_map.items():
            attribute_memory[v] = k
        relation_memory = [-1]*len(query_to_vg_rels_map)
        for k,v in query_to_vg_rels_map.items():
            relation_memory[v] = k
        return object_memory, attribute_memory, relation_memory       
 
    def rule_based_program(self, query, query_category, query_to_vg_objects_map=None, query_to_vg_attrs_map=None, query_to_vg_rels_map=None):
        query = query.lower()
        if not any([' '+m+' ' in ' '+query.strip()+' ' for m in self.middle_verb]):
            query = 'is it '+query.strip()
        '''
        if query_to_vg_objects_map:
           print ('Objects :: ', [k+'('+str(v)+')' for k,v in query_to_vg_objects_map.items()])
        if query_to_vg_attrs_map:
           print ('Attributes :: ', [k+'('+str(v)+')' for k,v in query_to_vg_attrs_map.items()])
        if query_to_vg_rels_map:
           print ('Relations :: ', [k+'('+str(v)+')' for k,v in query_to_vg_rels_map.items()])
        '''
        query_parsed, query_toks, query_pos, query_lemma, query_tok_descendants = self.query_cleaning.execute(query)
        children_of_root = self.get_children_of_root(query_parsed)
        programs = []
        for m in self.middle_verb:
            if m not in query_toks:
                continue    
            target_object, target_attribute, target_relation, objects_in_last, attributes_in_last, relations_in_last = self.parse_query(m, query_toks, query_lemma, query_pos, query_tok_descendants, children_of_root, query_to_vg_objects_map, query_to_vg_attrs_map, query_to_vg_rels_map)
            if self.disable_relations:
                 target_relation = set([])
                 relations_in_last = set([])
            object_memory, attribute_memory, relation_memory = self.load_memories(objects_in_last, attributes_in_last, relations_in_last, target_object, target_attribute, target_relation, query_to_vg_objects_map, query_to_vg_attrs_map, query_to_vg_rels_map)
            program = Program(self.prog_config, object_memory, attribute_memory, relation_memory, self.disable_relations)
            count =1
            for o in objects_in_last:
                    operator = "filter_object"
                    arg_table_index = [ object_memory.index(o)]#, (count-1) ]
                    target_table_index = count
                    program.add_line_of_code(operator, arg_table_index, [o])
                    count+=1
            obj_var_start = 1
            obj_var_end = count-1
            '''if len(objects_in_last)>1 and obj_var_start<count:
                    for index in range(obj_var_start, count):
                        operator = "intersection"
                        arg_table_index = [ index, index+1 ]
                        target_table_index = count
                        program.add_line_of_code(operator, arg_table_index, [])
                        count += 1 
                    obj_var_end = count
            '''
            attr_var_start = count
            for a in attributes_in_last:
                    operator = "filter_attribute"
                    arg_table_index = [ attribute_memory.index(a)]#, (count-1) ]
                    target_table_index = count
                    program.add_line_of_code(operator, arg_table_index, [a])
                    count+=1
            attr_var_end = count-1
            '''if len(attributes_in_last)>1:
                    program.append('Obj'+str(count)+' = Intersection([Obj'+str(attr_var_start)+', ..., Obj'+str(count-1)+'])')
                    attr_var_end = count
                    count+=1
            '''
            for r in relations_in_last:
                    operator = "filter_relation"
                    arg_table_index = [ relation_memory.index(r)]#, (count-1)]
                    target_table_index = count
                    program.add_line_of_code(operator, arg_table_index, [r])
                    count+=1
            if len(target_attribute)>0 and len(target_object)>0:
                    for a in target_attribute:
                            operator = "filter_attribute"
                            arg_table_index = [ attribute_memory.index(a)]#, (count-1)]
                            target_table_index = count
                            program.add_line_of_code(operator, arg_table_index, [a])
                            count += 1
                    to_return = []
                    for o in target_object:
                            if query_category=='logical':
                                operator = 'get_object'
                            elif query_category=='quantitative':
                                operator = 'count_object'
                            elif query_category=='boolean':
                                operator = 'exists_object'
                            arg_table_index = [ object_memory.index(o)]#, (count-1)]
                            target_table_index = count
                            program.add_line_of_code(operator, arg_table_index, [o])
                            count+=1
                    if len(target_relation)>0:
                            for r in target_relation:
                                 if query_category=='logical':
                                     operator = 'get_relation'
                                 elif query_category=='quantitative':
                                     operator = 'count_relation'
                                 elif query_category=='boolean':
                                     operator = 'exists_relation'
                                 arg_table_index = [ relation_memory.index(r)]#, (count-1) ]
                                 target_table_index = count
                                 program.add_line_of_code(operator, arg_table_index, [r])
                                 count+=1
            elif len(target_attribute)>0 and len(target_object)==0:
                    to_return = []
                    for a in target_attribute:
                            if query_category=='logical':
                                operator = 'get_attribute'
                            elif query_category=='quantitative':
                                operator = 'count_attribute'
                            elif query_category=='boolean':
                                operator = 'exists_attribute'
                            arg_table_index = [ attribute_memory.index(a)]#, (count-1) ]
                            target_table_index = count
                            program.add_line_of_code(operator, arg_table_index, [a])
                            count+=1
                    if len(target_relation)>0:
                            for r in target_relation:
                                 if query_category=='logical':
                                     operator = 'get_relation'
                                 elif query_category=='quantitative':
                                     operator = 'count_relation'
                                 elif query_category=='boolean':
                                     operator = 'exists_relation'
                                 arg_table_index = [ relation_memory.index(r)]#, (count-1) ]
                                 target_table_index = count
                                 program.add_line_of_code(operator, arg_table_index, [r])
                                 count+=1
            elif len(target_attribute)==0 and len(target_object)>0:
                    for o in target_object:
                            if query_category=='logical':
                                operator = 'get_object'
                            elif query_category=='quantitative':
                                operator = 'count_object'
                            elif query_category=='boolean':
                                operator = 'exists_object'
                            arg_table_index = [ object_memory.index(o)]#, (count-1) ]
                            target_table_index = count
                            program.add_line_of_code(operator, arg_table_index, [o])
                            count+=1
                    if len(target_relation)>0:
                            for r in target_relation:
                                 if query_category=='logical':
                                     operator = 'get_relation'
                                 elif query_category=='quantitative':
                                     operator = 'count_relation'
                                 elif query_category=='boolean':
                                     operator = 'exists_relation'
                                 arg_table_index = [ relation_memory.index(r)]#, (count-1) ]
                                 target_table_index = count
                                 program.add_line_of_code(operator, arg_table_index, [r])
                                 count+=1
            elif len(target_attribute)==0 and len(target_object)==0 and len(target_relation)>0:
                    for r in target_relation:
                            if query_category=='logical':
                                 operator = 'get_relation'
                            elif query_category=='quantitative':
                                 operator = 'count_relation'
                            elif query_category=='boolean':
                                 operator = 'exists_relation'
                            arg_table_index = [ relation_memory.index(r)]#, (count-1) ]
                            target_table_index = count
                            program.add_line_of_code(operator, arg_table_index, [r])
                            count+=1
            elif len(target_attribute)==0 and len(target_object)==0 and len(target_relation)==0:
                    if query_category=='logical':
                         operator = 'get'
                    elif query_category=='quantitative':
                         operator = 'count'
                    elif query_category=='boolean':
                         operator = 'exists'
                    arg_table_index = [ ]#(count-1) ]
                    target_table_index = count
                    program.add_line_of_code(operator, arg_table_index, [])
                    count+=1
            if self.verbose:
                print (program.print_program())
            programs.append(program)
            break
        return programs

if __name__=="__main__":
    opt = get_options()
    rule_based_program_generator = RuleBasedProgramGeneration(opt)
    query = 'What is the man playing with ?'
    programs = rule_based_program_generator.rule_based_program(query)
    for i, program in enumerate(programs):
        print ('Rule Based Program (', i, ') :: \n', program.print_program())#'\n\n'.join(['\n'.join(programs_str[i]) for i in len(programs_str)]))

