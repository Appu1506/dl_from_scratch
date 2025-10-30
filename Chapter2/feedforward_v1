import numpy as np
import pandas as pd

class Operation():
    
    def __init__(self):
        pass
    

    def forward(self, input_) -> np.ndarray:
       self.input_ = input_
       self.output = self._output()
       return self.output
   
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
       self.input_grad = self._input_grad(output_grad)
       return self.input_grad
   
    
    def _output(self):
       raise NotImplementedError()
       
       
    def _input_grad(self,output_grad: np.ndarray) -> np.ndarray:
       raise NotImplementedError()
       
       
       
       
class ParamOperation(Operation):
    
    def __init__(self, param:np.ndarray) -> np.ndarray:
        super().__init__()
        self.param = param
            
    
    def backward(self, output_grad: np.ndarray)->np.ndarray:
       self.input_grad = self._input_grad(output_grad)
       self.param_grad = self._param_grad(output_grad)

       return self.input_grad


    def _param_grad(self,output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
        
            

class WeightMultiply(ParamOperation):
    
    def __init__(self, W:np.ndarray):
        super().__init__(W)

    def _output(self) -> np.ndarray:
        return np.dot(self.input_, self.param)


    def _input_grad(self, output_grad:np.ndarray) -> np.ndarray:
        
        return np.dot(output_grad, self.param.T)
        

    def _param_grad(self, output_grad:np.ndarray) -> np.ndarray:
        
        return np.dot(self.input_.T, output_grad)






class BiasAdd(ParamOperation):
    
    def __init__(self, B:np.ndarray):
        super().__init__(B)
    
    
    def _output(self) -> np.ndarray:
        return self.input_ + self.param


    def _input_grad(self, output_grad:np.ndarray) -> np.ndarray:        
        return output_grad * np.ones_like(self.input_)
        

    def _param_grad(self, output_grad:np.ndarray) -> np.ndarray:
        
        param_grad = np.ones_like(self.param) * output_grad
    
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])                      
 
                   

    

class Layer():
    
    def __init__(self, neurons:int):

        self.operations = []
        self.params = []  
        self.param_grads = []
        self.neurons = neurons
        self.First = True
        

    def setup_layer(self, input_:np.ndarray):
        raise NotImplementedError()


    def forward(self, input_:np.ndarray) -> np.ndarray:
           
        if self.First is True:
            self.setup_layer(input_)
            self.First=False
 
        self.input_ = input_
     
        for operation in self.operations:
            input_ = operation.forward(input_)

        self.output = input_
                        
        return self.output



    def backward(self, output_grad:np.ndarray) -> np.ndarray:
        
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
            
        input_grad = output_grad
        
        self._param_grad()            
        return input_grad
            
            
    def _param_grad(self):
        
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation._param_grad)
                

        return self.param_grads





class Sigmoid(Operation):
    
    def __init__(self):
        pass
    
    
    def _output(self) -> np.ndarray:
        
        return 1/(1 + np.exp(-1 * self.input_))
        
       
    def _input_grad(self, output_grad:np.ndarray) -> np.ndarray:        
        sigmoid_backward = self.output * (1-self.output)
        input_grad = sigmoid_backward * output_grad

        return input_grad
        
        
    

                
class Dense(Layer):

    def __init__(self, neurons:int, activation: Operation = Sigmoid()):
        
        super().__init__(neurons)
        self.activation = activation        
        
    def setup_layer(self, input_):
        
        self.params = []
        
        self.params.append(np.random.rand(input_.shape[1], self.neurons))
        self.params.append(np.random.rand(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]), BiasAdd(self.params[1]), self.activation]        

        return None

    




x = np.random.randn(2, 3)  # 2 samples, 3 features


dense = Dense(1)           # 2 neurons

out = dense.forward(x)



dense.params[0]


_intermediate = x
for op in dense.operations:
    _intermediate = op.forward(_intermediate)
    print(f"{op.__class__.__name__} output:\n", _intermediate)


print("Input:\n", x)
print("Output:\n", out)

    

y_true = np.ones_like(out)  # dummy target
loss_grad = out - y_true
    
_intermediate = loss_grad

for op in dense.operations:
    _intermediate = op.backward(_intermediate)
    print(f"{op.__class__.__name__} output:\n", _intermediate)
    



dense.operations[0].param = np.zeros_like(dense.operations[0].param)  # WeightMultiply
dense.operations[1].param = np.zeros_like(dense.operations[1].param)  # BiasAdd

# Forward pass
out = dense.forward(x)



        
        
       
       
