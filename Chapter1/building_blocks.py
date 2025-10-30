import numpy as np
import math

from typing import Callable

from typing import List


x = np.array([[1,2,3],[4,5,6]])


w = np.array([2,3,4])



Array_Function = Callable[[np.ndarray], np.ndarray]


Chain = List[Array_Function]




def func1(x: np.ndarray) -> np.ndarray:
    
    y = x + 1
    
    return y



def square(x: np.ndarray) -> np.ndarray:
    
    y = np.power(x,2)
    
    return y


def exp(x: np.ndarray) -> np.ndarray:
    
    y = np.exp(x)
    
    return y


def deriv(func:Callable[[np.ndarray],np.ndarray],
          input_:np.ndarray, delta:float=0.001) -> np.ndarray:
    
    der = (func(input_+0.001) - func(input_-0.001))/(2*0.001)
    
    return der


def chain_length_2(chain:Chain, x:np.ndarray)->np.ndarray:

    assert len(chain) == 2
    
    f1 = chain[0]
    f2 = chain[1]

    y1 = f1(x)

    dydx = deriv(f1, x)
    dy2dx = deriv(f2, y1)
    
    #print(dydx,dy2dx)
    
    print(dydx*dy2dx)




    




def chain_length_3(chain:Chain, x:np.ndarray)->np.ndarray:
    
    assert len(chain)==3
    
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    
    y1 = f1(x)
    y2 = f2(y1)
    
    print(y1)
    print(y2)
    
    dydx1 = deriv(f1, x)
    dydx2 = deriv(f2, y1)

    print(dydx1, dydx2)
    
    dydx3 = deriv(f3, y2)*dydx2*dydx1
    
    print(dydx3)
    
    
    

from typing import Dict, Tuple


def forward_linear_regression(X_batch: np.ndarray,y_batch: np.ndarray,weights: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, np.ndarray]]:

    
    N = np.dot(X_batch,weights['W'])
    
    P = N + weights['B']    
    
    loss = np.mean(np.power(y_batch - P, 2))
    
    forward_info: Dict[str, ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch
    
    return forward_info, loss
       

X_batch = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
])   

y_batch = np.array([
    [10.0],
    [20.0]
])   

weights = {
    "W": np.array([[2.0],   
                   [1.0]])
    ,
    "B": np.array([[0.5]])  
}    
    

print(forward_linear_regression(X_batch,y_batch,weights))


def loss_gradient_linear_regressions(forward_info: Dict[str, np.ndarray],weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    
    dLdP = -2*(forward_info['y']-forward_info['P']) 
    
    dPdN =  np.ones_like(forward_info['N'])

    dPdB = np.ones_like(weights['B'])

    dLdN = dLdP * dPdN

    dNdW = np.transpose(forward_info['X'], (1, 0))

    dLdW = np.dot(dNdW, dLdN)
    dLdB = (dLdP * dPdB).sum(axis=0)
    
    loss_gradients: Dict[str, ndarray] = {}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB
    
    return loss_gradients





def init_weights(n_in: int) -> Dict[str, np.ndarray]:
    '''
    Initialize weights on first forward pass of model.
    '''
    
    weights: Dict[str, np.ndarray] = {}
    W = np.random.randn(n_in, 1)
    B = np.random.randn(1, 1)
    
    weights['W'] = W
    weights['B'] = B

    return weights


Batch = Tuple[np.ndarray, np.ndarray]

def generate_batch(X: np.ndarray, 
                   y: np.ndarray,
                   start: int = 0,
                   batch_size: int = 10) -> Batch:
    '''
    Generate batch from X and y, given a start position
    '''
    assert X.ndim == y.ndim == 2, \
    "X and Y must be 2 dimensional"

    if start+batch_size > X.shape[0]:
        batch_size = X.shape[0] - start
    
    X_batch, y_batch = X[start:start+batch_size], y[start:start+batch_size]
    
    return X_batch, y_batch



def permute_data(X: np.ndarray, y: np.ndarray):
    '''
    Permute X and y, using the same permutation, along axis=0
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def train(X: np.ndarray, 
          y: np.ndarray, 
          n_iter: int = 1000,
          learning_rate: float = 0.001,
          batch_size: int = 100,
          return_losses: bool = False, 
          return_weights: bool = False, 
          seed: int = 1) -> None:
    '''
    Train model for a certain number of epochs.
    '''
    if seed:
        np.random.seed(seed)
    start = 0

    # Initialize weights
    weights = init_weights(X.shape[1])

    # Permute data
    X, y = permute_data(X, y)
    
    if return_losses:
        losses = []

    for i in range(n_iter):

        # Generate batch
        if start >= X.shape[0]:
            X, y = permute_data(X, y)
            start = 0
        
        X_batch, y_batch = generate_batch(X, y, start, batch_size)
        start += batch_size
    
        # Train net using generated batch
        forward_info, loss = forward_linear_regression(X_batch, y_batch, weights)

        if return_losses:
            losses.append(loss)

        loss_grads = loss_gradient_linear_regressions(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

    if return_weights:
        return losses, weights
    
    return None

    
        

train_info = train(X, y,learning_rate = 0.001,batch_size=23,return_weights=True,return_losses=True,seed=80718)


def predict(X: np.ndarray,
     weights: Dict[str, np.ndarray]):
     '''
     Generate predictions from the step-by-step linear regression model.
     '''
     N = np.dot(X, weights['W'])
     return N + weights['B']

preds = predict(X, weights) # weights = train_info[0]



preds = predict(X, train_info[-1]) # weights = train_info[0]


def mae(preds: np.ndarray, actuals: np.ndarray):
     '''
     Compute mean absolute error.
     '''
     return np.mean(np.abs(preds - actuals))



def rmse(preds: np.ndarray, actuals: np.ndarray):
     '''
     Compute root mean squared error.
     '''
     return np.sqrt(np.mean(np.power(preds - actuals, 2)))








