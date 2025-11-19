# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:52:02 2025

@author: SPalliy
"""




import pandas as pd

import numpy as np
from numpy import ndarray


from typing import Tuple, List, Dict



class RNNNode():
    
    def forward(self, x_in:ndarray, h_in:ndarray, params: Dict[str, Dict[str, ndarray]]) -> Tuple[ndarray]:
        
        self.z = np.column_stack([x_in, h_in])
        
        h_int = np.dot(self.z, params['w_h']['value']) + params['b_h']['value']
        
        self.h_out = self.tanh(h_int)
        
        self.x_out = np.dot(self.h_out, params['w_o']['value']) + params['b_o']['value']
        
        return self.x_out, self.h_out
    
    
    
    def backward(self, x_output_grad:ndarray, h_output_grad:ndarray, params: Dict[str, Dict[str, ndarray]]) -> Tuple[ndarray]:
        
        #h_out : W*z  + b
        
        
        h_int_grad = np.dot(params['w_o']['value'], x_output_grad)
        
        params['w_o']['deriv'] += np.dot(h_output_grad.T, x_output_grad) #should be h_out not the grad
        params['b_o']['deriv'] += h_output_grad.sum(axis=0)  #should be x_output_grad

        h_int_grad = self.dtanh(h_int_grad)
        h_int_grad += h_output_grad
        
        dz = np.dot(params['w_h']['value'], h_int_grad)
    
        params['w_h']['deriv'] += np.dot(self.z.T, h_int_grad)
        params['b_h']['deriv'] += h_int_grad.sum(axis=0)
        
      

class RNNLayer():
    
    def __init__(self, hidden_size:int, output_size:int, weight_scale:float=None):

        self.hidden_size = hidden_size     
        self.h_start = np.zeros(1,hidden_size)
        self.output_size = output_size
        self.first = True
        self.weight_scale = weight_scale
        
        
    def _init_params(self, input_: ndarray):
        self.vocab_size = input_.shape[2]
        
        self.params = {}

        self.params['w_h'] = {}
        self.params['w_o'] = {}
        self.params['b_h'] = {}
        self.params['b_o'] = {}
        
        
        self.params['w_h']['value'] = np.random.normal(size = (self.vocab_size, self.hidden_size))
        self.params['b_h']['value'] = np.random.normal(size = (1, self.hidden_size))

        self.params['w_o']['value'] = np.random.normal(size = (self.hidden_size, self.output_size))    
        self.params['b_o']['value'] = np.random.normal(size = (1, self.output_size))


        self.params['w_h']['deriv'] = np.zeros_like(self.params['w_h']['value'])
        self.params['b_h']['deriv'] = np.zeros_like(self.params['b_h']['value'])
    
        self.params['w_o']['deriv'] = np.zeros_like(self.params['w_o']['value'])
        self.params['b_o']['deriv'] = np.zeros_like(self.params['b_o']['value'])

        self.cells = [RNNNode() for x in range(input_.shape[1])]


    def _clear_gradients(self):        
        for key in self.params.keys():
            self.params[key]['deriv'] = np.ones_like(self.params[key]['deriv'])


    def forward(self, x_seq_in:ndarray):
        
        seq_len = x_seq_in.shape[1]
        batch_size = x_seq_in.shape[0]
        
        h_in = np.copy(self.h_start)
        h_in = np.repeat(h_in, batch_size, axis=0)

        x_out = np.zeros(batch_size, seq_len, self.output_size)
        
        for t in range(seq_len):
            x_in = x_seq_in[:, t:, ]
            
            y_out, h_in = self.cells[t].forward(x_in, h_in, self.params)
            
            x_out[:, t:, ] = y_out

        self.h_start = h_in.mean(axis=0, keepdims=True)

        return x_out


    
    def backward(self, x_seq_out_grad:ndarray):
         
        seq_len = x_seq_out_grad.shape[1]
        batch_size = x_seq_out_grad.shape[0]
        h_int_grad = np.zeros(batch_size, self.hidden_size)
        
        x_seq_in_grad = np.zeros(batch_size, seq_len, self.vocab_size)
        
        for t in reversed(seq_len):
           x_out_grad = x_seq_out_grad[:, t:, ]
           
           grad_out, h_int_grad = self.cells[t].backward(x_out_grad,h_int_grad,self.params)

           x_seq_in_grad[:, t, :] = grad_out


        return x_seq_in_grad
        


def RNNModel():
    
    def __init__(self, layers:List[RNNLayer],sequence_length: int,vocab_size:int, loss: Loss):
        
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.layers = layers
        self.loss = loss
        
        for layer in self.layers:
            setattr(layer, 'sequence_length', sequence_length)

        
        
    def forward(self, x_batch:ndarray):
        
        for layer in self.layers:
             x_batch = layer.forward(x_batch) 
                          
        return x_batch
        


    def backward(self, loss_grad:ndarray):
        
        for layer in reversed(self.layers):
             loss_grad = layer.forward(loss_grad) 
                          
        return loss_grad
    
    
def RNNTrainer():

    def __init__()    


        