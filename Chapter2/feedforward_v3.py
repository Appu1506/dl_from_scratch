

"""
Operations
forward -> output
backward - > inputgrad
output
inputgrad


paramops
backward -> paramgrad, input grad
inputgrad
paramgrad


weightadd
output
inputgrad
paramgrad



biasadd
output
inputgrad
paramgrad


layer
setuplayer
forward
backward
paramgrad


dense
setuplayer

"""


import numpy as np
import pandas as pd


from numpy import ndarray
from typing import List, Tuple

class Operation:
    
    def __init__(self):
        pass
        
        
    def forward(self, input_:ndarray) -> ndarray:
        self.input_ = input_
        self.output = self._output()        
        return self.output


    def backward(self, output_grad:ndarray) -> ndarray:
        self.input_grad = self._input_grad(output_grad)
        
        return self.input_grad
        
        
    def _output(self) -> ndarray:        
        raise NotImplementedError()

        
    def _input_grad(self,output_grad:ndarray) -> ndarray:        
        raise NotImplementedError()


class ParamOperation(Operation):
    
    def __init__(self, param:ndarray):
        super().__init__()
        self.param = param
        
        
    def backward(self, output_grad) -> ndarray:
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)
        
        return self.input_grad                
        
    def _param_grad(self, output_grad) -> ndarray:
        raise NotImplementedError()
        

class WeightMultiply(ParamOperation):
    
    def __init__(self, W:ndarray):
        super().__init__(W)
        
        
    def _output(self) -> ndarray:
        return np.dot(self.input_, self.param)
        
        
    def _input_grad(self, output_grad) -> ndarray:
        return np.dot(output_grad, self.param.T)        


    def _param_grad(self, output_grad) -> ndarray:
        return np.dot(self.input_.T, output_grad)        
        

class BiasAdd(ParamOperation):
    
    def __init__(self, B:ndarray):
        super().__init__(B)
        
        
    def _output(self) -> ndarray:
        return self.input_ + self.param
        
        
    def _input_grad(self, output_grad) -> ndarray:
        return output_grad * np.ones_like(self.input_)        


    def _param_grad(self, output_grad) -> ndarray:
        grad  = np.ones_like(self.param) * output_grad        
        return np.sum(grad, axis=0).reshape(1, grad.shape[1])            
    
    
class Sigmoid(Operation):

    def __init__(self):
        super().__init__()
        
     
    def _output(self) -> ndarray:
        return 1/(1 + np.exp(-1 * self.input_))
        
    def _input_grad(self, output_grad:ndarray) -> ndarray:
        sigmoid_grad = self.output * (1-self.output)
        return sigmoid_grad * output_grad
     

class Layer():
    
    def __init__(self, neurons:int):
        self.neurons = neurons    
        self.operations = []
        self.param_grad = [] 
        self.first  = True


    def _setup_layer(self, input_):
        raise NotImplementedError()

        
    def forward(self, input_:ndarray)-> ndarray:
        
        self.input_ = input_
        
        if self.first is True:
            self._setup_layer(input_)
            self.first = False
        
        for operation in self.operations:
           input_ = operation.forward(input_)
        
        self.output = input_
        return self.output        
    

    def backward(self, output_grad:ndarray)-> ndarray:
    
        for operation in reversed(self.operations):
           output_grad = operation.backward(output_grad)
        
        input_grad = output_grad
        self._param_grad()
        return input_grad 


    def _param_grad(self)-> ndarray:
        
        self.param_grads = []

        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grad.append(operation._param_grad)                

        return self.param_grads


class Dense(Layer):
    
    def __init__(self, neurons:int, activation:Sigmoid):
        super().__init__(neurons)
        self.activation = activation            

    def _setup_layer(self, input_:ndarray):
        
        self.params = []
        
        self.params.append(np.random.rand(input_.shape[1], self.neurons))
        self.params.append(np.random.rand(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]), BiasAdd(self.params[1]), self.activation]


class Loss():
    
    def __init__(self):
        pass
        
    def forward(self, prediction:ndarray, target:ndarray) -> ndarray:
        self.prediction = prediction
        self.target = target        
            
        loss_value = self._output()
        return loss_value
    
    def backward(self) -> ndarray:
        self.loss_grad  = self._loss_grad()
        return self.loss_grad

    def _output(self):
        raise NotImplementedError()        

    def _loss_grad(self):
        raise NotImplementedError()        


class MeanSquaredError(Loss):
    
    def __init__(self):
        super().__init__()
            
    def _output(self):
        return np.sum(np.power(self.prediction-self.target, 2))/self.prediction.shape[0]

    def _loss_grad(self):
        return (2 * (self.prediction-self.target))/self.prediction.shape[0]


class Optimizer():
    
    def __init__(self, lr=float):
        self.lr = lr
    
    def step(self):
        raise NotImplementedError()        
        


class SGD(Optimizer):
    
    def __init__(self, lr=0.01):
        super().__init__(lr)
        
        
    def step(self):
        for param, param_grad in zip(self.net.params(),self.net.param_grads()):
            param_grad -= param_grad * self.lr            
            
            
class NeuralNetwork(Dense):
           
    def __init__(self, loss:Loss, layers:List[Layer]):
        self.loss = loss
        self.layers = layers
            

    def forward(self, x_batch:ndarray) -> ndarray:
        x_out  = x_batch

        for layer in self.layers:
            x_out = layer.forward(x_out)
            
        return x_out


    def backward(self, loss_grad:ndarray) -> None:

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(loss_grad)
            
        return None


    def params(self):
        for layer in self.layers:
            yield from layer.params


    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads



    def train_batch(self, x_batch:ndarray, y_batch:ndarray) -> ndarray:
        
        prediction = self.forward(x_batch)
        loss = self.loss.forward(prediction, y_batch)            
        
        self.backward(self.loss.backward())
        
        return loss




def permute_data(x_batch, y_batch):
    
     index = np.random.permutation(x_batch.shape[0])
     return x_batch[index], y_batch[index]

class Trainer():
    
    def __init__(self, net = NeuralNetwork, optim = Optimizer):
        self.net = net
        self.optim = optim
        setattr(self.optim, 'net', self.net)


    def generate_batch(self, X:ndarray, Y:ndarray, size:int=32) -> Tuple[ndarray]:
        N = X.shape[0]

        for i in range(0, N, size):
            x_batch, y_batch = X[i:i+size], Y[i:i+size]        
        
        yield x_batch, y_batch
        
        
    
    def fit(self, x_train:ndarray, y_train:ndarray, x_test:ndarray, y_test:ndarray, 
            eval_every:int=10, batch_size:int=32, restart:bool=True, epochs:int=10) -> None:
        
    
        if restart:
            for layer in self.net.layers:
                layer.first=True
            
        for e in range(epochs):            
            x_train, y_train = permute_data(x_train, y_train) 
            
            batch_generator = self.generate_batch(x_train, y_train, batch_size)
            
            for i, (x_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(x_batch, y_batch)
                self.optim.step()

            if (e+1) % eval_every == 0:
                 test_preds = self.net.forward(x_test)
                 loss = self.net.loss.forward(test_preds, y_test)
                 print(f"Validation loss after {e+1} epochs is {loss:.3f}")
    
    
def mae(y_true: ndarray, y_pred: ndarray):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: ndarray, y_pred: ndarray):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def eval_regression_model(model: NeuralNetwork,
                          X_test: ndarray,
                          y_test: ndarray):
    
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(mae(preds, y_test)))
    print()
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))   







boston = pd.read_csv(r"C:\Users\SPalliy\Downloads\ML\BostonHousing.csv")

data = boston.drop("MEDV", axis=1).values     

target = boston["MEDV"].values                


features = boston.drop("MEDV", axis=1).columns # same as boston.feature_names



from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)





def to_2d_np(a: ndarray, 
          type: str="col") -> ndarray:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
    "Input tensors must be 1 dimensional"
    
    if type == "col":        
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

# make target 2d array
y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)



nn = NeuralNetwork(
    layers=[Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError()
)


lr = NeuralNetwork(
    layers=[Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError())



dl = NeuralNetwork(
    layers=[Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError())
    
            
trainer = Trainer(nn, SGD(lr=0.001))

trainer = Trainer(lr, SGD(lr=0.001))

trainer = Trainer(dl, SGD(lr=0.001))


trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10)


print()
eval_regression_model(nn, X_test, y_test)









