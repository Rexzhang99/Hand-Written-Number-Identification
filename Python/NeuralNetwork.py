import numpy as np
import pickle
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

with open("data/pkl/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

image_size = 28  # width and length
no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)

@np.vectorize
def tanh(x):
    return np.tanh(x)

@np.vectorize
def relu(x):
    return np.maximum(0,x)


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)


class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate,
                 bias=None
                 ):

        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes

        self.no_of_hidden_nodes = no_of_hidden_nodes

        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """
        A method to initialize the weight matrices
        of the neural network with optional
        bias nodes"""

        bias_node = 1 if self.bias else 0

        rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes,
                          self.no_of_in_nodes + bias_node))

        rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_normal(mean=0,
                             sd=1,
                             low=-rad,
                             upp=rad)
        self.who = X.rvs((self.no_of_out_nodes,
                          self.no_of_hidden_nodes + bias_node))

    def train_single(self, input_vector, target_vector):
        """
        input_vector and target_vector can be tuple,
        list or ndarray
        """

        bias_node = 1 if self.bias else 0
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector,
                                           [self.bias]))

        output_vectors = []
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.wih,
                                input_vector)
        output_hidden = activation_function(output_vector1)

        if self.bias:
            output_hidden = np.concatenate((output_hidden,
                                            [[self.bias]]))

        output_vector2 = np.dot(self.who,
                                output_hidden)
        output_network = activation_function(output_vector2)

        output_errors = target_vector - output_network
        # update the weights:
        tmp = output_errors * output_network * (1.0 - output_network)
        tmp = self.learning_rate * np.dot(tmp,
                                          output_hidden.T)
        self.who += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T,
                               output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1, :]
        else:
            x = np.dot(tmp, input_vector.T)
        self.wih += self.learning_rate * x

    def train(self, data_array,
              labels_one_hot_array,
              epochs=1,
              intermediate_results=False):
        intermediate_weights = []
        for epoch in range(epochs):
            for i in range(len(data_array)):
                self.train_single(data_array[i],
                                  labels_one_hot_array[i])
            if intermediate_results:
                intermediate_weights.append((self.wih.copy(),
                                             self.who.copy()))
        return intermediate_weights

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray

        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector,
                                           [self.bias]))
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.wih,
                               input_vector)
        output_vector = activation_function(output_vector)

        if self.bias:
            output_vector = np.concatenate((output_vector,
                                            [[self.bias]]))

        output_vector = np.dot(self.who,
                               output_vector)
        output_vector = activation_function(output_vector)

        return output_vector

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs

    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm


epochs = 1
# with open("nist_tests.csv", "w") as fh_out:
#     for hidden_nodes in [20, 50, 100, 120, 150]:
#         for learning_rate in [0.01, 0.05, 0.1, 0.2]:
#             for bias in [None, 0.5]:
#                 network = NeuralNetwork(no_of_in_nodes=image_pixels,
#                                         no_of_out_nodes=10,
#                                         no_of_hidden_nodes=hidden_nodes,
#                                         learning_rate=learning_rate,
#                                         bias=bias)
#                 weights = network.train(train_imgs,
#                                         train_labels_one_hot,
#                                         epochs=epochs,
#                                         intermediate_results=True)
#                 for epoch in range(epochs):
#                     print("*", end="")
#                     network.wih = weights[epoch][0]
#                     network.who = weights[epoch][1]
#                     train_corrects, train_wrongs = network.evaluate(train_imgs,
#                                                                     train_labels)
#
#                     test_corrects, test_wrongs = network.evaluate(test_imgs,
#                                                                   test_labels)
#                     outstr = str(hidden_nodes) + " " + str(learning_rate) + " " + str(bias)
#                     outstr += " " + str(epoch) + " "
#                     outstr += str(train_corrects / (train_corrects + train_wrongs)) + " "
#                     outstr += str(train_wrongs / (train_corrects + train_wrongs)) + " "
#                     outstr += str(test_corrects / (test_corrects + test_wrongs)) + " "
#                     outstr += str(test_wrongs / (test_corrects + test_wrongs))
#
#                     fh_out.write(outstr + "\n")
#                     fh_out.flush()

np.random.seed(seed=233423)

# activation_function = sigmoid
# activation_function = tanh
activation_function = relu

network = NeuralNetwork(no_of_in_nodes=image_pixels,
                        no_of_out_nodes=10,
                        no_of_hidden_nodes=120,
                        learning_rate=0.1,
                        bias=None)

weights = network.train(train_imgs,
                        train_labels_one_hot,
                        epochs=epochs,
                        intermediate_results=True)

train_cm = network.confusion_matrix(test_imgs, test_labels)
print(train_cm)

test_corrects, test_wrongs = network.evaluate(test_imgs, test_labels)
accuracy = test_corrects / (test_corrects + test_wrongs)
print(accuracy)
