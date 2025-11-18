



import pandas as pd
import numpy as np

from numpy import ndarray

from typing import List, Tuple, Dict




x = np.array([1,2,3])
h = np.array([4,5,6])







class RNN_Node():
    
    def forward(self, x_in:ndarray, h_in:ndarray, param: Dict[str, Dict[str, ndarray ]]) -> Tuple[ndarray]:
        
        self.z  = np.column_stack((x_in, h_in))
        
        h = np.dot(self.z, param['w_h']['value']) + param['b_h']['value']
        
        self.h_out = self.tanh(h)
        
        self.x_out = np.dot(self.h_out, param['w_o']['value']) + param['b_o']['value']
        
        return self.x_out, self.h_out
        

def backward(self, x_output_grad:ndarray, h_output_grad:ndarray, param:Dict[str, Dict[str, ndarray]]) -> Tuple[ndarray]:
    
        param['w_o'] += np.dot(self.h_out, h_output_grad.T)
        param['b_o'] += np.sum(x_output_grad, axis=0)
        
        dh = np.dot(x_output_grad, param['w_h']['value'].T)
        dh += h_output_grad
        
        dh_int = dh * self.dtanh(self.h_in)

        param['w_h'] += np.dot(self.z.T, dh_int.T)
        param['b_h'] += np.sum(dh_int, axis=0)
        
        dz = np.dot(dh_int, param['w_h']['value'].T)

        x_in_grad = dz[:, :self.x_in.shape[1]]
        h_in_grad = dz[:, self.x_in.shape[1]:]

        return x_in_grad, h_in_grad
        
        






