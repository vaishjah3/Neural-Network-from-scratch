'''import numpy as np
#This defines a basic structure of a neuron somewhere in between a Network Network
inputs=[1, 2, 3, 4]
weight=[0.2 ,0.8, 0.5, 0.6]
bias=2

output= inputs[0]*weight[0]+inputs[1]*weight[1]+inputs[2]*weight[2]+bias
print(output)

#Coding a neuron layer with 4 inputs fully connected to 3 outputs



inputs=[1, 2, 2.5, 3.4]

weight1=[0.2, 0.8, 0.76, 0.9]
weight2=[1, 2.5, 3.6, 2.1]
weight3=[1.7, 2.3, 3.2, 1.3]

bias1=1
bias2=2
bias3=0.5 


output=[inputs[0]*weight1[0]+inputs[1]*weight1[1]+inputs[2]*weight1[2]+inputs[3]*weight1[3]+bias1,
        inputs[0]*weight2[0]+inputs[1]*weight2[1]+inputs[2]*weight2[2]+inputs[3]*weight2[3]+bias2,
inputs[0]*weight2[0]+inputs[1]*weight2[1]+inputs[2]*weight2[2]+inputs[3]*weight2[3]+bias3]

X=[[1, 2, 2.5, 3.4], [0.2, 0.3, -0.5, 0.7], [1.2, -0.75, 0.89, 0.54]] 
weights=[[0.2, 0.8, 0.76, 0.9], [1, 2.5, 3.6, 2.1], [1.7, 2.3, 3.2, 1.3]]
bias=[1,2, 0.5]

weights2=[[-0.3, 0.4, 0.73], 
          [0.9, -5.2, 1.6], 
          [0.3, 0.3, 3.1]]
bias2=[-1.5, 2.3, -1.5]



#inputs=np.transpose(inputs)
layer1_outputs=np.dot(X, np.array(weights).T )+bias
layer2_outputs=np.dot(layer1_outputs, np.array(weights2).T )+bias2'''

import numpy as np

np.random.seed(0)


X=[[1, 2, 2.5, 3.4], 
   [0.2, 0.3, -0.5, 0.7], 
   [1.2, -0.75, 0.89, 0.54]] 

class Layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights= 0.01 * np.random.randn(n_inputs, n_neurons)
        self.bias=np.zeros((1, n_neurons))
    def compute_forward_prop(self, inputs):
        self.output=np.dot(inputs, self.weights)+self.bias

Layer1=Layer_dense(4,5)
Layer2=Layer_dense(5,2)

Layer1.compute_forward_prop(X)
Layer2.compute_forward_prop(Layer1.output)
print(Layer2.output)




