import Layers as Layer
import numpy as np
import loss_functions as Loss
import Activations as activations


def softmax_crossentropy_with_logits(logits,reference_answers):
    # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
    logits_for_answers = logits[np.arange(len(logits)),reference_answers]
    
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
    
    return xentropy
def grad_softmax_crossentropy_with_logits(logits,reference_answers):
    # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)),reference_answers] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- ones_for_answers + softmax) / logits.shape[0]



x = np.array(np.random.random([1000,4])).T      #X should be expressed in d*n
y = np.random.randint(0,2,1000)
network = []

layer_1 = Layer.Dense(x.shape[0],4)
network.append(layer_1)
layer_2 = activations.RELU()
network.append(layer_2)
layer_3 = Layer.Dense(4,2)
network.append(layer_3)
network.append(activations.RELU())

layer_4 = Loss.MSE()
layer_5 = Loss.Test_Loss()

# network.append(layer_3)

activations = []
input = x   #For first layer, input is X

for index,layer in enumerate(network):
    # print(f' Doing {index+1} layer')
    activations.append(layer.forward(input))
    input = activations[-1]         #latest append is the input to new layer
    



logits = activations[-1]
# loss = softmax_crossentropy_with_logits(logits,y)
# loss_grad = grad_softmax_crossentropy_with_logits(logits,y)

loss = layer_5.loss(logits,y)
loss_grad = layer_5.gradient(logits,y)
# print(loss_grad.shape)

# print(loss_grad.shape)

activations_list = [x] + activations
for reversed_layer_index in reversed(range(len(network))):
    print(f'Backpropagating through {reversed_layer_index} layer')
    # print(activations_list[reversed_layer_index].shape)
    layer = network[reversed_layer_index]
    print(layer.__class__)
    print('\n')
    loss_grad= layer.backward(activations_list[reversed_layer_index],loss_grad)



print(logits)
# print(np.mean(loss_grad,axis=0))