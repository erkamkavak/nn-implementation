import math
import numpy as np
import random
from typing import List

class ActivationFunction: 
    def __init__(self) -> None:
        pass

class ReLU(ActivationFunction): 
    def __init__(self) -> None:
        super().__init__()  


class DataPoint: 
    def __init__(self, inputs, expected_outputs) -> None:
        self.inputs = inputs
        self.expected_outputs = expected_outputs

class Node: 
    def __init__(self, weights, bias) -> None:
        self.bias = bias
        # Weights for a node is the weights of all edges that are coming to this node. 
        self.weights = weights
    
    def calculate_output(self, inputs): 
        self.inputs = inputs
        self.weighted_input = self.calculate_weighted_input()
        self.output = self.output_function(self.weighted_input)
        return self.output
    
    def calculate_weighted_input(self): 
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    def output_function(self, weighted_input): 
        """
        A node in neural network should behave just like neurons, it should have a 
        threshold to have a positive output. Output function is an activation function for this purpose.
        Activation function like this helps the neural network to change smoothly based on the
        weights and biases. 
        """
        return 1 / (1 + np.exp(-weighted_input))
    
    def loss(self, expected_output): 
        return 1/2 * (expected_output - self.output) ** 2
        # return np.nan_to_num(-expected_output*np.log(self.output)-(1-expected_output)*np.log(1-self.output))
    
    def output_pd_weighted_input(self): 
        """
        Partial derivative of output wrt total input
        """
        return self.output * (1 - self.output)
    
    def loss_pd_output(self, expected_output): 
        """
        Partial derivative of node loss wrt output
        """
        return (self.output - expected_output)
    
    def loss_pd_weighted_input(self, expected_output : float):
        """
        Partial derivative of node loss wrt total input. 
        d(node_loss) / d(weighted_input) = d(node_loss) / d(output) * d(output) / d(weighted_input)
        """
        return self.loss_pd_output(expected_output) * self.output_pd_weighted_input()
    
    def weighted_input_pd_weight(self, index):
        """
        weighted_input[index] = weights[index] * inputs[index]
        d(weighted_input(index)) / d(weights[index]) = inputs[index]
        """
        return self.inputs[index] 
    
    def apply_gradient(self, gradient, learn_rate): 
        """
        Apply gradients function tries to minimize the loss of a node by 
        applying the gradient vector to the weights and bias.
        Gradient is the partial derivative of the loss wrt the total input for this node.  
        """
        loss_pd_weighted_input = gradient
        for weight_index in range(len(self.weights)): 
            loss_pd_weight = loss_pd_weighted_input * self.weighted_input_pd_weight(weight_index)
            self.weights[weight_index] -= learn_rate * loss_pd_weight
        # weighted_input = weights * inputs + bias -> d(weighted_input) / d(bias) = 1 
        loss_pd_bias = loss_pd_weighted_input * 1
        self.bias -= learn_rate * loss_pd_bias

        

class Layer: 
    def __init__(self, num_nodes_in : int, num_nodes : int) -> None:
        # num_nodes_in is the number of the nodes of the previous layer(all of these 
        # nodes will be coming in to the nodes of this layer)
        self.num_nodes = num_nodes
        self.nodes = []
        for i in range(self.num_nodes): 
            weights = np.random.randn(num_nodes_in) / np.sqrt(num_nodes_in)
            bias = np.random.randn(1)[0]
            self.nodes.append(Node(weights, bias))

    def calculate_outputs(self, inputs): 
        outputs = np.zeros(self.num_nodes)
        for node_index in range(self.num_nodes):
            node : Node = self.nodes[node_index]
            outputs[node_index] = node.calculate_output(inputs)
        return outputs
    
    def get_outputs(self): 
        outputs = np.zeros(self.num_nodes)
        for node_index in range(self.num_nodes): 
            node : Node = self.nodes[node_index]
            outputs[node_index] = node.output
        return outputs
    
    def apply_gradients(self, gradients, learn_rate): 
        """
        Applies the apply_gradient function to all its nodes. 
        Gradients is an array of loss_pd_weighted_inputs for each node in the layer. 
        """
        for node_index in range(self.num_nodes): 
            node : Node = self.nodes[node_index]
            node.apply_gradient(gradients[node_index], learn_rate)


class NeuralNetwork: 
    def __init__(self, layer_sizes) -> None:
        self.layers = []
        # first layer size is the size of the input layer. We don't need an input layer in our 
        # network so we don't create it. 
        for i in range(1, len(layer_sizes)): 
            self.layers.append(Layer(layer_sizes[i - 1], layer_sizes[i]))
    
    def calculate_outputs(self, inputs): 
        for layer_index in range(len(self.layers)): 
            layer : Layer = self.layers[layer_index]
            inputs = layer.calculate_outputs(inputs)
        return inputs
    
    def calculate_accuracy(self, data_points : List[DataPoint]): 
        accuracy = 0
        for i in range(len(data_points)):
            data_point : DataPoint = data_points[i] 
            output = np.argmax(self.calculate_outputs(data_point.inputs))
            expected = np.argmax(data_point.expected_outputs)
            if output == expected: accuracy += 1
        return accuracy / len(data_points)
    
    def calculate_total_loss(self, data_points : List[DataPoint]): 
        total_loss = 0
        for data_point in data_points: 
            self.calculate_outputs(data_point.inputs)
            output_layer : Layer = self.layers[len(self.layers) - 1]
            loss = 0
            for node_index in range(output_layer.num_nodes): 
                node : Node = output_layer.nodes[node_index]
                loss += node.loss(data_point.expected_outputs[node_index])
            total_loss += loss / len(data_points)
        return total_loss
    
    def calculate_all_gradients(self, expected_outputs): 
        """
            The gradient we want to calculate for a node is the partial derivative of 
            loss of the node wrt total inputs of the node. We can then use this value to 
            find partial derivative of loss wrt to the weights and bias and minimize the loss
            by changing the weights and bias. 

            For the output layer we can directly calculate d(loss) / d(weighted_input).
            But for the hidden layers we can't directly calculate it. We can calculate is 
            using the 
        """
        # gradients = [[]] * len(self.layers) 
        gradients = [np.zeros(layer.num_nodes) for layer in self.layers]
        
        # First calculate gradients of the output layer. 
        output_layer : Layer = self.layers[-1]
        loss_pd_weighted_input_array = np.zeros(output_layer.num_nodes) 
        for node_index in range(output_layer.num_nodes): 
            node : Node = output_layer.nodes[node_index]
            loss_pd_weighted_input_array[node_index] = node.loss_pd_weighted_input(expected_outputs[node_index])
        gradients[-1] = loss_pd_weighted_input_array

        # Then calculate gradients of the hidden layers using the gradients of next layer. 
        for layer_index in range(len(self.layers) - 2, -1, -1): 
            layer : Layer = self.layers[layer_index]
            loss_pd_weighted_input_array = np.zeros(layer.num_nodes)
            for node_index in range(layer.num_nodes): 
                node : Node = layer.nodes[node_index]
                loss_pd_output_sums = 0
                next_layer : Layer = self.layers[layer_index + 1]
                for o in range(next_layer.num_nodes): 
                    loss_pd_output_sums += gradients[layer_index + 1][o] \
                                            * next_layer.nodes[o].weights[node_index]
                loss_pd_weighted_input_array[node_index] = loss_pd_output_sums * node.output_pd_weighted_input()
            gradients[layer_index] = loss_pd_weighted_input_array
        return gradients
    
    def apply_all_gradients(self, gradients, learn_rate): 
        for layer_index in range(len(self.layers)):
            layer : Layer = self.layers[layer_index] 
            layer.apply_gradients(gradients[layer_index], learn_rate)
    
    def learn(self, training_data : DataPoint, learn_rate : float): 
        self.calculate_outputs(training_data.inputs)
        gradients = self.calculate_all_gradients(training_data.expected_outputs)
        self.apply_all_gradients(gradients, learn_rate)
        return gradients

    def train(self, training_datas : List[DataPoint], epochs, mini_batch_size):
        learn_rate = 3.0
        for i in range(epochs): 
            random.shuffle(training_datas)
            mini_batches = [
                training_datas[k:k+mini_batch_size]
                for k in range(0, len(training_datas), mini_batch_size)
            ]
            for mini_batch_data in mini_batches:
                gradients = [np.zeros(layer.num_nodes) for layer in self.layers] 
                for training_data in mini_batch_data: 
                    self.calculate_outputs(training_data.inputs)
                    new_gradients = self.calculate_all_gradients(training_data.expected_outputs)
                    gradients = [grad + new_grad for (grad, new_grad) in zip(gradients, new_gradients)] 
                gradients = [grad / mini_batch_size for grad in gradients]
                self.apply_all_gradients(gradients, learn_rate)
                start_ind = np.random.randint(0, m - 100)
                total_loss = neural_network.calculate_total_loss(training_datas[start_ind:start_ind + 100])
                print(f"Total loss : {total_loss}")
                accuracy = neural_network.calculate_accuracy(training_datas[start_ind:start_ind + 100])
                print(f"Current accuracy : {accuracy}")




def create_output_layer(output, output_layer_size): 
    """
    Creates the output layer as an array from the given output layer size and the index of 
    the output.  
    """
    vec = np.zeros(output_layer_size)
    vec[output] = 1.0
    return vec


import pandas as pd
data = pd.read_csv('./train.csv')
data = np.array(data)
m, n = data.shape

input_layer_size = 784
output_layer_size = 10
layer_sizes = [input_layer_size, 30, output_layer_size]
neural_network = NeuralNetwork(layer_sizes)

training_datas = []
for i in range(m): 
    training_input = data[i][1:n]
    training_input = training_input / 255.
    training_output = data[i][0]
    training_output = create_output_layer(training_output, output_layer_size)
    training_datas.append(DataPoint(training_input, training_output))

neural_network.train(training_datas, 30, 10)