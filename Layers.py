import numpy as np

class Layer:
    ''' Base class for MLP Layers, defines just the boilerplate for derived classes'''

    def __init__(self) -> None:
        pass

    def forward(self,input):
        return input
    
    def backward(self,input,grad_output):
        # dloss/dinput = dloss/dlayer * dlayer/dinput

        return np.dot(grad_output,np.eye(input.shape[1]))
    

class Dense(Layer):
    def __init__(self,input_dimension,output_dimension,learning_rate = 0.01):
        ''' Input : Input vector , output_dimension : num of ouptputs
        Formula : Y = WX , where X is expressed in d*n and W in output* features'''

        self.learning_rate = learning_rate
        self.weights = np.random.randn(output_dimension,input_dimension)          #Later needs to be replaced by Kaiming Initialization
        self.biases = np.random.randn(output_dimension).reshape(-1,1)


    def forward(self, input):
        return np.dot(self.weights,input) + self.biases
    
    def backward(self, input, grad_output):         #need to update function to check the overall weight calculations
        # print(f'Self weights shape:{self.weights.shape}')
        gradient_wrt_input = np.dot(self.weights.T,grad_output)

        # grad_wrt_weight = np.dot(input,grad_output.T)
        grad_wrt_weight = np.dot(grad_output,input.T)
        # print(f'Grad op {grad_output}')
        grad_wrt_bias = grad_output.mean(axis=0)*input.shape[0]
       
        self.weights = self.weights - self.learning_rate*grad_wrt_weight
        self.biases = self.biases - self.learning_rate*grad_wrt_bias

        # print(self.weights)
        return gradient_wrt_input
    