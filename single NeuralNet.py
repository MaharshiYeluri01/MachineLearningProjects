from numpy import exp,random,dot,array
training_set_inputs=array([[1,1,1],[1,0,0],[0,1,1],[1,0,1]])
training_set_outputs=([[1,0,1,1]]).T
random.seed(15)
synaptic_weights=2*random.random(3,1)-1
for i in range (1000):
       output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
       synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
       print(caliculated_output,synaptic_weights)

print( 1 / (1 + exp(-(dot(array([0,0,1]), synaptic_weights)))))
                           
