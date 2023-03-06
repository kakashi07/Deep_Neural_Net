from Layers import *
    
class CrossEntropy(Layer):
    def __init__(self):
        super().__init__()
        pass

    def loss(self,predicted_value,true_value):
        loss = np.sum(predicted_value*np.log(true_value))
        return loss

class HingeLoss(Layer):
    pass

class MSE(Layer):
    def __init__(self) :
        super().__init__()
        pass

    def loss(self,predicted_value,true_value):
        loss = np.mean((predicted_value-true_value)**2)
        return loss
    
    def gradient(self,predicted_value,true_value):
        # assert len(predicted_value)==len(true_value), "Vectors length mismatch for gradient calculation for : Y and Yhat"
        return np.array(2*(1/max(predicted_value.shape))*np.sum(predicted_value-true_value)).reshape(-1,1)
    
    def backward(self, predicted_value, true_value):
        ''' Since it is the last layer, the backward takes only the true and predicted values'''
        return self.gradient(predicted_value,true_value)


class Test_Loss(Layer):
    def __init__(self) :
        super().__init__()
        pass

    def loss(self,predicted_value,true_value):
        loss = (np.square(predicted_value-true_value))/2
        return loss
    
    def gradient(self,predicted_value,true_value):
        return predicted_value - true_value

    def backward(self, predicted_value, true_value):
        ''' Since it is the last layer, the backward takes only the true and predicted values'''
        return self.gradient(predicted_value,true_value)

