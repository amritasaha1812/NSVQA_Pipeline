#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:51:28 2019

@author: amrita
"""
import numpy as np

def one_hot(vector, size):
    onehot = np.zeros((size), dtype=np.float32)
    if vector.shape[0]>0:
        onehot[vector] = 1.
    return onehot

def normalize(vector, ord):
    norm_ = np.linalg.norm(vector, ord)
    if norm_ == 0:
      return softmax(vector, axis=0)
    else:
      return vector/np.linalg.norm(vector, ord)

def pad_to_max(llist, value, size):
    llist_old = llist.copy()
    ll = [value]*(size-len(llist))
    llist.extend(ll)
    return llist

def pad_or_clip(llist, value, size):
    if len(llist)>=size:
        return llist[:size]
    else:
        return pad_to_max(llist, value, size)

def softmax(mat, axis):
    mat = np.array(mat)
    e = np.exp(mat)
    sum_e = np.expand_dims(np.sum(e, axis=axis), axis=-1)
    dist = e/sum_e
    return dist
