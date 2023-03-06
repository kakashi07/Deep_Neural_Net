from Layers import *

class RELU(Layer):
    def __init__(self) -> None:
        super().__init__()
        #No initializations needed for RELU Layer
        pass

    def forward(self,input):
        return np.maximum(0,input)
    
    def gradient(self,input):
        input[input<=0] = 0
        input[input > 0] = 1

        return input
    
    def backward(self, input, grad_output):
        ''' i.e. dl/dx = dl/dlayer * dlayer/dx --> dl/dlayer is grad_output and is known, dl/dlayer is the gradient of that layer with the input'''
        # return grad_output * self.gradient(input)
        # return grad_output * (input>0)
        return np.multiply(grad_output,self.gradient(input))

class Sigmoid(Layer):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self,input):
        return 1/(1+np.exp(input))
    
    def gradient(self,input):
        return input*(1-input)
    
    def backward(self, input, grad_output):
        return grad_output * self.gradient(input)
